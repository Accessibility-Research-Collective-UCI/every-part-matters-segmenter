import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json, get_nagative_mask
from .utils import DEFAULT_IMAGE_TOKEN, ANSWER_LIST, NEGATIVE_ANSWER_LIST, question_templates


class FigureSegDataset(torch.utils.data.Dataset):
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
        combine_sample_rate,
        precision: str = "fp32",
        image_size: int = 224,
        exclude_val=False,
        figure_seg_data="figure_seg",
    ):
        self.exclude_val = exclude_val
        self.figure_seg_data = figure_seg_data

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.answer_list = ANSWER_LIST

        seg_images = []
        seg_images += glob.glob(os.path.join(base_image_dir, self.figure_seg_data, "*.png"))
         
        jsons = [path.replace(".png", ".json") for path in seg_images]
        images = [json.loads(open(path, "r").read())["origin_image"] for path in jsons]
        self.figure_seg_data = (images, jsons)
        self.images = images

        self.vocab_path = vocab_path
        self.neg_sample_method = neg_sample_method
        self.neg_sample_rate = neg_sample_rate
        self.combine_sample_rate = combine_sample_rate

        print("FigureSegDataset: len(images) =", len(images))
    
    def __len__(self):
        return len(self.images)
    

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
        images, jsons = self.figure_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents = get_mask_from_json(json_path, image, data_type="figure_seg")
        neg_mask, neg_modules = get_nagative_mask(json_path, self.vocab_path, image, self.neg_sample_rate, self.neg_sample_method)

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []

        all_classes = {'name': sents["name"], 
                       'function': sents["function"], 
                       "relative position": sents["position"][1],
                       "absolute position": sents["position"][0]
                       }
        # sample: must have 'name' class
        name_only = {k: all_classes[k] for k in ['name']}
        name_rel_pos = {k: all_classes[k] for k in ['name', 'relative position']}
        name_abs_pos = {k: all_classes[k] for k in ['name', 'absolute position']}
        name_func = {k: all_classes[k] for k in ['name', 'function']}
        name_rel_abs_pos = {k: all_classes[k] for k in ['name', 'relative position', 'absolute position']}
        name_func_rel_pos = {k: all_classes[k] for k in ['name', 'function', 'relative position']}
        name_func_abs_pos = {k: all_classes[k] for k in ['name', 'function', 'absolute position']}
        name_func_rel_abs_pos = {k: all_classes[k] for k in ['name', 'function', 'relative position', 'absolute position']}

        sampled_classes = [name_only, name_rel_pos, name_abs_pos, name_func, name_rel_abs_pos, name_func_rel_pos, name_func_abs_pos, name_func_rel_abs_pos]

        # sampled_classes = [one_class, two_classes, three_classes, all_classes]
        # sampled_classes = [{
        #     "name": sents["name"],
        #     "function": sents["function"],
        #     "relative position": sents["position"][1],
        #     "absolute position": sents["position"][0]
        # }]
        sampled_classes = random.sample(sampled_classes, self.combine_sample_rate)
        sampled_sents = []
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for classes in sampled_classes:
            # question = random.choice(question_templates(classes))
            question = question_templates(classes)[0]
            questions.append(question)
            sampled_sents.append(question)

            # answers.append(random.choice(ANSWER_LIST))
            answers.append(ANSWER_LIST[0])

            
            # conv = conversation_lib.default_conversation.copy()

            # i = 0
            # while i < len(questions):
            #     conv.messages = []
            #     conv.append_message(conv.roles[0], questions[i])
            #     conv.append_message(conv.roles[1], answers[i])
            #     conversations.append(conv.get_prompt())
            #     i += 1
        
        for classes in neg_modules:
            question = question_templates(classes)[0]
            questions.append(question)
            sampled_sents.append(question)

            answers.append(NEGATIVE_ANSWER_LIST[0])

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        sampled_inds = list(range(len(sampled_sents)))
        sampled_sents = np.vectorize(questions.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_classes))
        ]
        sampled_masks += [
            (neg_mask == 1).astype(np.float32) for _ in range(len(neg_modules))
        ]
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
        )


