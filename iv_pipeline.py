import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import zipfile
import cv2
import numpy as np
from plot import PlotPrompt
import torch
import json
from tqdm import tqdm
from segment import segment_api
from figure_understanding.attribute_api import attribute_api
from enum import Enum

DEVICE = "cuda:0"
output_path = "vis_output/"

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        print(self.sum, self.count, self.avg)

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )
        # print(total)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def trans_polygon_to_mask(points, image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    # print(height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    for point in points:
        if len(point) == 0:
            continue
        sep_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(sep_mask, np.array([point], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(sep_mask, np.array([point], dtype=np.int32), 1)
        mask = mask + sep_mask
    mask = (mask >= 1).astype(np.int32)
    return mask


def metric(pred_masks, gold_masks):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    def intersectionAndUnionGPU(output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert output.dim() in [1, 2, 3]
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
        area_output = torch.histc(output, bins=K, min=0, max=K - 1)
        area_target = torch.histc(target, bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target
    intersection, union, acc_iou = 0.0, 0.0, 0.0
    for pred_mask_i, gold_mask_i in zip(pred_masks, gold_masks):
        pred_mask_i = torch.from_numpy(pred_mask_i.astype(np.int32)).cuda()
        gold_mask_i = torch.from_numpy(gold_mask_i).cuda()
        intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_mask_i.contiguous().clone(), gold_mask_i.contiguous(), 2, ignore_index=255
            )
        intersection += intersection_i
        union += union_i
        acc_iou += intersection_i / (union_i + 1e-5)
        if intersection_i[1] > union_i[1]:
            print("error")
            print(intersection_i, union_i)
            print(torch.where(pred_mask_i == 1))
            exit()
        acc_iou[union_i == 0] += 1.0  # no-object target
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
    acc_iou = acc_iou.cpu().numpy() / len(gold_masks)
    intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=len(gold_masks))
    
    # intersection_meter.all_reduce()
    # union_meter.all_reduce()
    # acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

# ocr + ner + segment
class Pipeline():
    def __init__(self, ocr="epm", ner="bert", segment="epm", attribute="epm") -> None:
        self.ocr_type = ocr
        self.ocr_model = self.inint_ocr(self.ocr_type)
        self.ner_model = ner
        self.segment_type = segment
        self.sam_model = self.inint_segment(self.segment_type)
        self.attribute = attribute

    
    def inint_ocr(self, ocr_model):
        if ocr_model == "epm":
            from ocr import ocr_api
            model = ocr_api.ocr_api
        elif ocr_model == "paddle_ocr":
            from paddleocr import PaddleOCR
            model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        elif ocr_model == "gpt4" or ocr_model == "mplug" or ocr_model == "ideal":
            model = None

        return model
    
    def inint_segment(self, segment_model):
        if segment_model == "epm":
            from segment import segment_api
            model =  segment_api.segment_vote_api
        else:
            model = None
        return model
    
    def forward(self, image_path, text, gold_modules, gold_check_modules=[]):

        # step 1 ocr
        if self.ocr_type == "epm":
            modules = self.ocr_model(text, image_path)
        elif self.ocr_type == "paddle_ocr":
            modules = []
            result = self.ocr_model.ocr(image_path, cls=True)
            for idx in range(len(result)):
                res = result[idx]
                if res is None:
                    continue
                for line in res:
                    # if line[1][1] > 0.8:
                    modules.append(line[1][0])
        elif self.ocr_type == "gpt4":
            ocr_file = open("ocr/figure_ocr_gpt4.json", "r", encoding="utf-8")
            ocr_data = json.loads(ocr_file.read())
            modules = ocr_data[image_path]
        elif self.ocr_type == "mplug":
            ocr_file = open("ocr/figure_ocr_mplug.json", "r", encoding="utf-8")
            ocr_data = json.loads(ocr_file.read())
            modules = ocr_data[image_path]
        elif self.ocr_type == "ideal":
            modules = gold_modules
        modules = list(set(modules))
        print("OCR: ", modules)
        
        # step 2 ner
        if self.ner_model == "bert":
            from ner.bert_ner.inference import ner
            words = ner(text)
        elif self.ner_model == "gpt4":
            ner_file = open("ner/gpt_ner.json", "r", encoding="utf-8")
            ner_data = json.loads(ner_file.read())
            words = ner_data[image_path]
            words = words[1:-1].split(",")
            words = [w.strip() for w in words]
        elif self.ner_model == "ideal":
            words = gold_check_modules
        words = list(set(words))
        print("NER: ", words)
        
        # step 3 segment
        checked_words = []
        for word in words:
            for mod in modules:
                if word.lower() == mod.lower():
                    checked_words.append(mod)
                    break
        missed_words = [w for w in modules if w not in checked_words]
        print("Missed words: ", missed_words)

        checked_modules = {}
        missed_modules = {}

        
        for word in missed_words:
            if self.segment_type == "epm":
                if self.attribute == "epm":
                    attr = attribute_api(image_path, word)
                elif self.attribute == "gpt4":
                    try:
                        attr = self.gpt4_attribute[image_path][word]
                    except:
                        attr = {"absolute_position": "", "relative_position": "", "function": ""}
                elif self.attribute == "llava":
                    for l in self.llava_attribute:
                        if l["image_path"] == image_path and l["name"] == word:
                            attr = l
                            break
                elif self.attribute == "mplug":
                    for l in self.mplug_attribute:
                        if l["image_path"] == image_path and l["name"] == word:
                            attr = l
                            break
                _, exist = segment_api.segment_api(name=word, image_path=image_path, output_path=output_path+image_path.split("/")[-1].replace(".png", "_")+word+".png")
                if exist:
                    missed_mask, exist = self.sam_model(name=word, image_path=image_path,  output_path=output_path+image_path.split("/")[-1].replace(".png", "_")+word+".png", absolute_position=attr["absolute_position"], relative_position=attr["relative_position"], function=attr["function"])
                    missed_modules[word] = missed_mask
                    
                else:
                    raise ValueError("Invalid strategy")        
            
        
        return words, modules, missed_modules, checked_modules

    
test_dir = "dataset/iv_test/"
test_file_list = os.listdir(test_dir)
test_data = [json.loads(open(test_dir + f, "r").read()) for f in test_file_list if f.endswith(".json")]
json_list = [f for f in test_file_list if f.endswith(".json")]


def test_iv(segment="epm", ocr="gpt4", ner="gpt4", attribute="epm"):
    import time
    pipeline = Pipeline(segment=segment, ocr=ocr, ner=ner, attribute=attribute)
    pred_masks = []
    gold_masks = []
    acc, pre, gold = 0, 0, 0
    
    total_time = 0
    for data in tqdm(test_data):
        start_time = time.time()
        image_path = data["origin_image"]
        text = " ".join(data["paragraph"])
        missed_modules = list(set(data["missed_modules"]))
        modules = list(set(data["modules"]))
        checked_modules = [m for m in modules if m not in missed_modules]
        _, _, pred_missed_modules, pred_checked_modules = pipeline.forward(image_path, text, modules, checked_modules)

        gold_points = data["shapes"][0]["points"]
        gold += len(missed_modules)
        pre += len(pred_missed_modules)

        for m in missed_modules:
            gold_mask = trans_polygon_to_mask(gold_points[m], image_path)
            if m not in pred_missed_modules:
                pred_masks.append(np.zeros(gold_mask.shape).astype(np.int32))
                gold_masks.append(gold_mask)
            else:
                pred_masks.append(pred_missed_modules[m])
                gold_masks.append(gold_mask)
                acc += 1
        end_time = time.time()
        total_time += end_time - start_time
    print("average time: ", total_time / len(test_data))
    metric(pred_masks, gold_masks)
    print(acc, pre, gold)
    print("precision: {:.2f}, recall {:.2f} f1: {:.2f}".format(acc / pre, acc / gold, 2 * acc / (pre + gold)))


test_iv(segment="epm", ocr="gpt4", ner="gpt4", attribute="epm")