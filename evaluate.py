import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
from enum import Enum
import numpy as np
import cv2
from tqdm import tqdm
import sys

test_dir = "./dataset/figure_seg/test/"
test_file_list = os.listdir(test_dir)
test_data = [json.loads(open(test_dir + f, "r").read()) for f in test_file_list if f.endswith(".json")]
json_list = [f for f in test_file_list if f.endswith(".json")]

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



def test_epm():
    from segment.segment_api import segment_vote_api, segment_api
    import time
    from figure_understanding.attribute_api import attribute_api

    pred_masks, gold_masks = [], []
    pred_pos_masks, gold_pos_masks = [], []
    total_time = 0
    pos_time = 0
    for data in tqdm(test_data):
        
        image_path = data["origin_image"]
        gold_points = data["shapes"][0]["points"]
        gold_points = data["shapes"][0]["points"]
        gold_mask = trans_polygon_to_mask(gold_points, image_path)

        # predict positive module
        attr = attribute_api(image_path, data["name"])
        attr["name"] = data["name"]
        attr["image_path"] = image_path
        attr["output_path"] = "./vis_output/epm_{}.jpg".format(data["name"].replace("/", ""))
        start_time = time.time()
        pred_mask, exist = segment_api(image_path=attr["image_path"], name=attr["name"], output_path=attr["output_path"], absolute_position="", relative_position="", function="")
        if exist:
            pred_mask, _ = segment_vote_api(image_path=attr["image_path"], name=attr["name"], output_path=attr["output_path"], absolute_position=attr["absolute_position"], relative_position=attr["relative_position"], function=attr["function"])
        end_time = time.time()
        total_time += end_time - start_time
        pos_time += end_time - start_time
        pred_masks.append(pred_mask.astype(np.int32))
        gold_masks.append(gold_mask)
        pred_pos_masks.append(pred_mask.astype(np.int32))
        gold_pos_masks.append(gold_mask)
        
        # predict negative module
        attr = attribute_api(image_path, data["neg_module"])
        attr["name"] = data["neg_module"]
        attr["image_path"] = image_path
        attr["output_path"] = "./vis_output/epm_neg_{}.jpg".format(data["neg_module"].replace("/", ""))
        start_time = time.time()
        pre_mask_neg, exist = segment_api(image_path=attr["image_path"], name=attr["name"], output_path=attr["output_path"], absolute_position="", relative_position="", function="")
        if exist:
            # if don't want to use vote, set the function to segment_api
            # function, absolute_position, relative_position can be set to ""
            pre_mask_neg, _ = segment_vote_api(image_path=attr["image_path"], name=attr["name"], output_path=attr["output_path"], absolute_position=attr["absolute_position"], relative_position=attr["relative_position"], function=attr["function"])
        end_time = time.time()

        total_time += end_time - start_time
        pred_masks.append(pre_mask_neg.astype(np.int32))
        gold_masks.append(np.zeros_like(gold_mask).astype(np.int32))
        
        
    print("total time: ", total_time)
    print("average time: ", total_time / (2 * len(test_data)))
    print("pos time: ", pos_time)
    metric(pred_masks, gold_masks)
    metric(pred_pos_masks,gold_pos_masks)

test_epm()