import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import numpy as np
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .data_processing import get_nagative_mask
from .utils import DEFAULT_IMAGE_TOKEN, atrr_vqa_templates, atrr_vqa_templates_neg


def preprocess_multimodal(source):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source


class ATRRVQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        vocab_path,
        neg_sample_method,
        neg_sample_rate,
        precision: str = "fp32",
        image_size: int = 224,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.vocab_path = vocab_path
        self.neg_sample_method = neg_sample_method
        self.neg_sample_rate = neg_sample_rate
        DATA_DIR = os.path.join(base_image_dir, "figure_seg_checked")
        self.vqa_image_root = os.path.join(base_image_dir, "figure_seg_checked")

        data_file = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
        self.vqa_data = []
        for file in data_file:
            with open(os.path.join(DATA_DIR, file)) as f:
                org_data = json.load(f)
                image = cv2.imread(org_data["origin_image"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                conversantions = atrr_vqa_templates(org_data)
                _, neg_ = get_nagative_mask(os.path.join(DATA_DIR, file), self.vocab_path, image, self.neg_sample_rate, self.neg_sample_method)
                neg_conversations = []
                for neg in neg_:
                    neg_conversations += atrr_vqa_templates_neg(neg)
                random.shuffle(conversantions)
                random.shuffle(neg_conversations)
                for conv in conversantions:
                    conversantion = []
                    conversantion.append(conv[0])
                    conversantion.append(conv[1])
                    conversantion[0]["value"] = "<image>\n" + conversantion[0]["value"]
                    self.vqa_data += [{
                        "id": org_data["origin_image"],
                        "image": org_data["origin_image"],
                        "conversations": conversantion
                    }
                    ]
                for conv in neg_conversations:
                    conversantion = []
                    conversantion.append(conv[0])
                    conversantion.append(conv[1])
                    conversantion[0]["value"] = "<image>\n" + conversantion[0]["value"]
                    self.vqa_data += [{
                        "id": org_data["origin_image"],
                        "image": org_data["origin_image"],
                        "conversations": conversantion
                    }
                    ]
                
        print("vqa_data: ", len(self.vqa_data))
    
    def __len__(self):
        return len(self.vqa_data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        if idx == -1:
            idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = item["image"]
        image = cv2.imread(image_path)
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
       