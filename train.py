import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import deepspeed
import numpy as np
import torch
import tqdm
from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from util.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from util.dataset import Dataset, ValDataset, collate_fn
import shutil
import time
from ds_config import init_ds  
from args import parse_args
from model.llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


def main(args):
    args = parse_args(args)

    # init log
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if os.path.exists(args.log_dir) == False:
        os.makedirs(args.log_dir)
    logger = SummaryWriter(args.log_dir)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side='right',
        use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[MODULE]")
    args.module_token_idx = tokenizer("[MODULE]", add_special_tokens=False)['input_ids'][0]

    if args.conv_type == "llava_v1.5":
        replace_llama_attn_with_flash_attn()
        args.use_mm_start_end = False

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "module_token_idx": args.module_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    
    model = LISAForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.device)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    train_dataset = Dataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        args.vocab_path,
        args.neg_sample_method,
        args.neg_sample_rate,
        args.combine_sample_rate,
        precision=args.precision,
        image_size=args.image_size,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        vqa_data=args.vqa_data,
        figure_seg_data=args.figure_seg_data,
    )

    args.steps_per_epoch = len(train_dataset) // (args.batch_size* args.grad_accumulation_steps* world_size)

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
            args.val_data_type
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
        test_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.test_dataset,
            args.image_size,
            args.val_data_type
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
    
    ds_config = init_ds(args)
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=int(args.device.split(":")[1]),
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=int(args.device.split(":")[1]),
            ),
        )

        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=int(args.device.split(":")[1]),
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou, eval_loss, best_loss = 0.0, 0.0, 0.0, 100.0

    if args.eval_only:
        if args.val_data_type == "figure_seg":
            giou, ciou = validate(val_loader, model_engine, 0, logger, args)
        elif args.val_data_type == "atrr_vqa" or args.val_data_type == "everything_vqa" or args.val_data_type == "mask_vqa":
            eval_loss = vqa_validate(val_loader, model_engine, args)
            print(eval_loss)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_epoch = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            logger,
            train_iter,
            args,
            tokenizer
        )

        if args.no_eval == False:
            if args.val_data_type == "figure_seg":
                giou, ciou = validate(val_loader, model_engine, epoch, logger, args)
                _, _ = validate(test_loader, model_engine, epoch, logger, args)
                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou
            elif args.val_data_type == "atrr_vqa" or args.val_data_type == "everything_vqa" or args.val_data_type == "mask_vqa":
                eval_loss = vqa_validate(val_loader, model_engine, args)
                is_best = eval_loss < best_loss
                best_loss = min(eval_loss, best_loss)

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.val_data_type == "figure_seg":
                if int(args.device.split(":")[1]) == 0:
                    torch.save(
                        {"epoch": epoch},
                        os.path.join(
                            args.log_dir,
                            "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                                best_score, cur_ciou
                            ),
                        ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    logger,
    train_iter,
    args,
    tokenizer
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if int(args.device.split(":")[1]) == 0:
                progress.display(global_step + 1)
                logger.add_scalar("train/loss", losses.avg, global_step)
                logger.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                logger.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                logger.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                logger.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                logger.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                logger.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if int(args.device.split(":")[1]) == 0:
                logger.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def vqa_validate(val_loader, model_engine, args):
    model_engine.eval()
    eval_loss = 0.0
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        with torch.no_grad():
            input_dict["inference"] = False
            output_dict = model_engine(**input_dict)
            ce_loss = output_dict["ce_loss"]
            eval_loss += ce_loss.item()
    eval_loss /= len(val_loader)
    print("eval_loss: ", eval_loss)
    return eval_loss


def validate(val_loader, model_engine, epoch, logger, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if int(args.device.split(":")[1]) == 0:
        logger.add_scalar("val/giou", giou, epoch)
        logger.add_scalar("val/giou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])

