import torch
import numpy as np
import glob
import os
import cv2
from transformers import CLIPImageProcessor
from .data_processing import get_mask_from_json
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .figure_seg_dataset import FigureSegDataset
from .vqa_dataset import VQADataset
from .atrr_vqa_dataset import ATRRVQADataset

# from .everything_vqa_dataset import EVERYTHINGVQADataset
# from .mask_vqa_dataset import MASKVQADataset
from .utils import (
    DEFAULT_IMAGE_TOKEN,
    ANSWER_LIST,
    question_templates,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
    atrr_vqa_templates,
    every_vqa_templates,
    mask_vqa_templates,
)
from model.llava.mm_utils import tokenizer_image_token
import torch.nn.functional as F
import json


class Dataset(torch.utils.data.Dataset):
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
        dataset="scigraphqa||figure_seg",
        sample_rate=[3, 1],
        vqa_data="scigraphqa_instruct_290k",
        figure_seg_data="figure_seg",
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "scigraphqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        precision,
                        image_size,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "figure_seg":
                self.all_datasets.append(
                    FigureSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        vocab_path,
                        neg_sample_method,
                        neg_sample_rate,
                        combine_sample_rate,
                        precision,
                        image_size,
                        exclude_val,
                        figure_seg_data,
                    )
                )
            elif dataset == "atrr_vqa":
                self.all_datasets.append(
                    ATRRVQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        vocab_path,
                        neg_sample_method,
                        neg_sample_rate,
                        precision,
                        image_size,
                    )
                )
            elif dataset == "everything_vqa":
                self.all_datasets.append(
                    EVERYTHINGVQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        precision,
                        image_size,
                    )
                )
            elif dataset == "mask_vqa":
                self.all_datasets.append(
                    MASKVQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        precision,
                        image_size,
                    )
                )

    def __len__(self):
        return sum([len(data) for data in self.all_datasets])

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        if len(self.all_datasets) == 1:
            return *data[idx], inference
        return *data[-1], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        val_data_type="figure_seg",
    ):
        self.images = glob.glob(os.path.join(base_image_dir, val_dataset, "*.png"))
        self.data_type = val_data_type
        self.ds = val_dataset
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.vqa = []
        if self.data_type == "atrr_vqa":
            for image_path in self.images:
                json_path = image_path.replace(".png", ".json")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask_json, sampled_sents = get_mask_from_json(
                    json_path, image, data_type=self.data_type
                )
                for sent in atrr_vqa_templates(sampled_sents):
                    self.vqa.append(
                        {
                            "id": image_path,
                            "image": image_path,
                            "conversations": sent,
                            "mask": mask_json,
                            "sent": sampled_sents,
                        }
                    )
        elif self.data_type == "everything_vqa":
            DATA_PATH = "/home/llmtrainer/Multimodal/EPM/dataset/figure_ocr_val.json"
            with open(DATA_PATH, "r") as f:
                org_data = [json.loads(line) for line in f.readlines()]
                for data in org_data:
                    image_path = data["image_path"]
                    json_path = (
                        "/home/llmtrainer/Multimodal/EPM/dataset/figure_seg_checked/"
                        + image_path.replace(".png", ".json")
                    )
                    if not os.path.exists(json_path):
                        continue
                    with open(json_path, "r") as f:
                        json_data = json.load(f)
                        image_path = json_data["origin_image"]
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mask_json, sampled_sents = get_mask_from_json(
                            json_path, image, data_type=self.data_type
                        )
                        for sent in every_vqa_templates(data):
                            self.vqa.append(
                                {
                                    "id": image_path,
                                    "image": image_path,
                                    "conversations": sent,
                                    "mask": mask_json,
                                    "sent": data["modules"],
                                }
                            )
        elif self.data_type == "mask_vqa":
            for image_path in self.images:
                json_path = image_path.replace(".png", ".json")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask_json, sampled_sents = get_mask_from_json(
                    json_path, image, data_type=self.data_type
                )
                self.vqa.append(
                    {
                        "id": image_path,
                        "image": image_path,
                        "conversations": mask_vqa_templates(sampled_sents),
                        "mask": mask_json,
                        "sent": sampled_sents,
                    }
                )

    def __len__(self):
        if self.data_type == "atrr_vqa" or self.data_type == "everything_vqa":
            return len(self.vqa)
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
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        if self.data_type == "figure_seg":
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".png", ".json")
            mask_json, sampled_sents = get_mask_from_json(
                json_path, image, data_type=self.data_type
            )
            prompt = question_templates(
                {
                    "name": sampled_sents["name"],
                    "function": sampled_sents["function"],
                    "relative position": sampled_sents["position"][1],
                    "absolute position": sampled_sents["position"][0],
                }
            )

            conv.append_message(conv.roles[0], prompt[0])
            conv.append_message(conv.roles[1], ANSWER_LIST[0])
            conversations.append(conv.get_prompt())

        elif (
            self.data_type == "atrr_vqa"
            or self.data_type == "everything_vqa"
            or self.data_type == "mask_vqa"
        ):
            # print(self.vqa[idx]['conversations'])
            conv.append_message(
                conv.roles[0], "<image>\n" + self.vqa[idx]["conversations"][0]["value"]
            )
            conv.append_message(
                conv.roles[1], self.vqa[idx]["conversations"][1]["value"]
            )
            # print(conv.get_prompt())
            conversations.append(conv.get_prompt())
            image_path = self.vqa[idx]["image"]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_json = self.vqa[idx]["mask"]
            sampled_sents = self.vqa[idx]["sent"]

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if (
            self.data_type == "atrr_vqa"
            or self.data_type == "everything_vqa"
            or self.data_type == "mask_vqa"
        ):
            masks = torch.from_numpy(mask_json)
        else:
            masks = [mask_json]

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1.5", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1.5":
        sep = conv.sep + conv.roles[1] + ": "
    elif conv_type == "llava_llama2":
        sep = "[/INST] "
    else:
        raise NotImplementedError

    for conversation, target in zip(conversation_list, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = len(tokenizer.encode(conversation))

        # print(conversation)

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            parts[0] = parts[0].strip()

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
            cur_len += 1

        target[cur_len:] = IGNORE_INDEX

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }
