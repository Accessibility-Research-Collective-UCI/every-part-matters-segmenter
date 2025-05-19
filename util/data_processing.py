import glob
import json
import os
import random
import cv2
import numpy as np
import Levenshtein

random.seed(42)


def get_mask_from_json(json_path, img, data_type="figure_seg"):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]

    height, width = img.shape[:2]

    if data_type == "figure_seg" or data_type == "atrr_vqa":
        comments = {
            "name": anno["name"],
            "function": anno["function"],
            "position": anno["position"],
        }
        if data_type == "atrr_vqa":
            mask = np.empty(shape=(0, height, width), dtype=np.uint8)
            return mask, comments
    elif data_type == "mask_vqa":
        comments = {
            "name": anno["name"],
        }
        mask = np.empty(shape=(0, height, width), dtype=np.uint8)
        return mask, comments
    else:
        comments = {}
        mask = np.empty(shape=(0, height, width), dtype=np.uint8)
        return mask, comments

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)

        for point in points:
            sep_mask = np.zeros((height, width), dtype=np.uint8)
            try:
                cv2.polylines(sep_mask, np.array([point], dtype=np.int32), True, 1, 1)
                cv2.fillPoly(sep_mask, np.array([point], dtype=np.int32), 1)
            except:
                print("error in: ", json_path)
                continue
            tmp_mask = tmp_mask + sep_mask

        # tmp_mask中>=1的值设为1
        tmp_mask = (tmp_mask >= 1).astype(np.uint8)

        tmp_area = tmp_mask.sum()
        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]
        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        for point in points:
            sep_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(
                sep_mask, np.array([point], dtype=np.int32), True, label_value, 1
            )
            cv2.fillPoly(sep_mask, np.array([point], dtype=np.int32), label_value)
            mask = mask + sep_mask

    if label_value == 1:
        mask = (mask >= 1).astype(np.uint8)
    else:
        mask = (mask >= 255).astype(np.uint8)

    return mask, comments


def get_nagative_mask(json_path, vocab_path, img, rate, method):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())
    with open(vocab_path, "r") as r:
        vocab = json.loads(r.read())
    name = anno["name"]
    image = json_path.split("/")[-1].replace(".json", ".png")
    height, width = img.shape[:2]
    vocab_dict = {line["image"]: line["module"] for line in vocab}
    vocab_list = []
    for line in vocab:
        if line["module"]:
            vocab_list += line["module"]
    vocab_list = list(set(vocab_list))
    if vocab_dict[image]:
        module_vocab = vocab_dict[image]
    else:
        module_vocab = []
    neg_modules = sample_negative_module(
        name, [w for w in vocab_list if w not in module_vocab and w != ""], rate, method
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    return mask, neg_modules


def sample_negative_module(name, vocab_pool, rate, method):
    if method == "random":
        sample_modules = random.sample(vocab_pool, rate)
    else:
        similarity_list = []
        for word in vocab_pool:
            similarity_list.append((word, calculate_similarity(name, word, method)))
        sample_modules = sorted(similarity_list, key=lambda x: x[-1])[-rate:]
    sample_modules = [
        {"name": m, "function": "", "relative position": "", "absolute position": ""}
        for m in sample_modules
    ]
    return sample_modules


def calculate_similarity(word1, word2, method="Levenshtein"):
    if method == "Levenshtein":
        distance = Levenshtein.distance(word1, word2)
        similarity = 1 - (distance / max(len(word1), len(word2)))
    return similarity


if __name__ == "__main__":
    data_dir = "./train"
    vis_dir = "./vis"

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    json_path_list = sorted(glob.glob(data_dir + "/*.json"))
    for json_path in json_path_list:
        img_path = json_path.replace(".json", ".jpg")
        img = cv2.imread(img_path)[:, :, ::-1]

        # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
        mask, comments, is_sentence = get_mask_from_json(json_path, img)

        ## visualization. Green for target, and red for ignore.
        valid_mask = (mask == 1).astype(np.float32)[:, :, None]
        ignore_mask = (mask == 255).astype(np.float32)[:, :, None]
        vis_img = img * (1 - valid_mask) * (1 - ignore_mask) + (
            (np.array([0, 255, 0]) * 0.6 + img * 0.4) * valid_mask
            + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
        )
        vis_img = np.concatenate([img, vis_img], 1)
        vis_path = os.path.join(
            vis_dir, json_path.split("/")[-1].replace(".json", ".jpg")
        )
        cv2.imwrite(vis_path, vis_img[:, :, ::-1])
        print("Visualization has been saved to: ", vis_path)
