import argparse


def parse_args(args):
    parser = argparse.ArgumentParser(description="Every Part Matters")
    parser.add_argument("--device", default="cuda:0", type=str)
    # /data_share/model_hub/llava/llava-v1.5-13b
    # /data_share/model_hub/llava/llava-v1.6-vicuna-13b
    # /data_share/model_hub/llava/llava-v1.5-7b
    parser.add_argument("--base_model", default="")
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    # dir path of the pretrained model for vision tower clip-vit-large-patch14
    parser.add_argument("--vision-tower", default="", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # "figure_seg||atrr_vqa||everything_vqa||mask_vqa"
    parser.add_argument("--dataset", default="figure_seg", type=str)
    # 8,2
    parser.add_argument("--sample_rates", default="8", type=str)
    parser.add_argument("--vqa_data", default="scigraphqa_instruct_2k", type=str)
    parser.add_argument("--val_dataset", default="val", type=str)
    parser.add_argument("--figure_seg_data", default="train", type=str)
    parser.add_argument("--test_dataset", default="test", type=str)

    # module_vocab.json
    parser.add_argument("--vocab_path", default="", type=str)
    parser.add_argument("--neg_sample_rate", default=1, type=int)
    parser.add_argument("--neg_sample_method", default="random", type=str)
    parser.add_argument("--combine_sample_rate", default=1, type=int)
    # "figure_seg||atrr_vqa||everything_vqa||mask_vqa"
    parser.add_argument("--val_data_type", default="", type=str)

    parser.add_argument("--dataset_dir", default="", type=str)
    parser.add_argument("--log_base_dir", default="", type=str)
    parser.add_argument("--exp_name", default="epm", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument(
        "--batch_size", default=8, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    # dir path of the pretrained model for mask decoder sam_vit_h_4b8939.pth
    parser.add_argument("--vision_pretrained", default="", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1.5",
        type=str,
        choices=["llava_v1.5", "llava_v1.6"],
    )
    parser.add_argument("--local_rank", default=0, type=int)
    return parser.parse_args(args)
