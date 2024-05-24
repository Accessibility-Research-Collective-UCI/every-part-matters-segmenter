import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from util.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, question_templates)
from model.llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


def parse_args(args):
    parser = argparse.ArgumentParser(description="EPM chat")
    parser.add_argument("--version", default="/data_share/model_hub/llava/llava-v1.5-13b")
    parser.add_argument("--vis_save_path", default="/home/llmtrainer/Multimodal/EPM/vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--vision_pretrained", default="/home/llmtrainer/Multimodal/LISA-main/sam_model/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument(
        "--vision-tower", default="/home/llmtrainer/LLM/irlab-llm/LISA/LISA/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1.5",
        type=str,
        choices=["llava_v1", "llava_llama_2", "llava_v1.5"],
    )
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    # pytorch_model_fu_n1_e1.bin
    parser.add_argument("--weight", default="/home/llmtrainer/Multimodal/EPM/checkpoint/pytorch_model_fu.bin", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    return parser.parse_args(args)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

args = parse_args(sys.argv[1:])
# Create model
tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[MODULE]")
args.seg_token_idx = tokenizer("[MODULE]", add_special_tokens=False).input_ids[0]

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
        "module_token_idx": args.seg_token_idx,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "vision_pretrained": args.vision_pretrained,
        "use_cache": False,
    }

torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half
model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)
model.get_model().initialize_lisa_modules(model.get_model().config)

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
    # model.print_trainable_parameters()

model.resize_token_embeddings(len(tokenizer))
state_dict = torch.load(args.weight, map_location="cpu")
model.load_state_dict(state_dict, strict=True)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
    vision_tower = model.get_model().get_vision_tower()
    model.model.vision_tower = None
    import deepspeed

    model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
    model = model_engine.module
    model.model.vision_tower = vision_tower.half().cuda()
elif args.precision == "fp32":
    model = model.float().cuda()

vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank)

clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)
# model.config.use_cache = False

def attribute_api(image_path, name):
    model.eval()

    

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    prompt_list = [
        "<image>\nWhat is the absoulte position of the '%s' in the image?" % name,
        "<image>\nWhat is the relative position of the '%s' in the image?" % name,
        "<image>\nWhat is the function of the '%s' in the image?" % name
    ]
    output_dict = {
        "absolute_position": "",
        "relative_position": "",
        "function": ""
    }
    for prompt in prompt_list:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        if args.use_mm_start_end:
            replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        if "absoulte position" in prompt:
            if "This module does not exist in the image." in text_output:
                output_dict["absolute_position"] = ""
            else:
                try:
                    output_dict["absolute_position"] = text_output.split("in the image is ")[1].replace("</s>","").strip()
                except:
                    output_dict["absolute_position"] = ""
        elif "relative position" in prompt:
            if "This module does not exist in the image." in text_output:
                output_dict["relative_position"] = ""
            else:
                try:
                    output_dict["relative_position"] = text_output.split("in the image is ")[1].replace("</s>","").strip()
                except:
                    output_dict["relative_position"] = ""
        elif "function" in prompt:
            if "This module does not exist in the image." in text_output:
                output_dict["function"] = ""
            else:
                try:
                    output_dict["function"] = text_output.split("in the image is ")[1].replace("</s>","").strip()
                except:
                    output_dict["function"] = ""
        
    # print(output_dict)
    
    return output_dict