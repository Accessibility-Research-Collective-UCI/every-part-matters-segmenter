from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
# calculate loss (ignore_index)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


QUESTION_PROMPT = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the corresponding module from the figure based on the given attributes: {attributes}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Using the provided attributes—{attributes}—extract the relevant module from the figure.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify and isolate the module within the figure by referencing the specified characteristics: {attributes}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Apply the attributes {attributes} to segment the appropriate module from the figure.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Utilize the defined attributes—namely, {attributes}—to delineate the target module from the figure."
]

ANSWER_LIST = [
    "[MODULE] is the module that has been segmented.",
    "The module subjected to segmentation is [MODULE].",
    "[MODULE].",
    "The segmented portion corresponds to [MODULE].",
    "Segmentation has been applied to [MODULE]."
]

NEGATIVE_ANSWER_LIST = [
    "[MODULE] is the module that has been segmented. There is no corresponding module on figure."
]


def question_templates(classes):
    attr = {
        "name": "unknown",
        "function": "unknown",
        "relative position": "unknown",
        "absolute position": "unknown"
    }
    for cls in classes.keys():
        if classes[cls] != "":
            attr[cls] = classes[cls]
    attributes = ""
    for k, v in attr.items():
        if v != "unknown":
            if v[-1] == ".":
                v = v[:-1]
            v = v[0].lower() + v[1:]
            attributes += f"{k}:{v}, "
    attributes = attributes[:-2]
    return [template.format(attributes=attributes) for template in QUESTION_PROMPT]


def atrr_vqa_templates(data):
    function = []
    ab_position = [
        {
            "from": "human",
            "value": "What is the absolute position of the '%s' in the image?" % data["name"]
        },
        {
            "from": "gpt",
            "value": "The absolute position of the '%s' in the image is %s" % (data["name"], data["position"][0][0].lower() + data["position"][0][1:])
        }
    ]
    rel_position = [
        {
            "from": "human",
            "value": "What is the relative position of the '%s' in the image?" % data["name"]
        },
        {
            "from": "gpt",
            "value": "The relative position of the '%s' in the image is %s" % (data["name"], data["position"][1][0].lower() + data["position"][1][1:])
        }
    ]
    if len(data["function"]) > 0:
        function = [
            {
                "from": "human",
                "value": "What is the function of the '%s' in the image?" % data["name"]
            },
            {
                "from": "gpt",
                "value": "The function of the '%s' in the image is %s" % (data["name"], data["function"][0].lower() + data["function"][1:])
            }
        ]
    if len(function) > 0:
        return [ab_position, rel_position, function]
    else:
        return [ab_position, rel_position]


def atrr_vqa_templates_neg(data):
    function = []
    ab_position = [
        {
            "from": "human",
            "value": "What is the absolute position of the '%s' in the image?" % data["name"]
        },
        {
            "from": "gpt",
            "value": "This module does not exist in the image."
        }
    ]
    rel_position = [
        {
            "from": "human",
            "value": "What is the relative position of the '%s' in the image?" % data["name"]
        },
        {
            "from": "gpt",
            "value": "This module does not exist in the image."
        }
    ]
    function = [
            {
                "from": "human",
                "value": "What is the function of the '%s' in the image?" % data["name"]
            },
            {
                "from": "gpt",
                "value": "This module does not exist in the image."
            }
    ]
    return [ab_position, rel_position, function]

def every_vqa_templates(data):
    template = [
        {
            "from": "human",
            "value": "Tell me the names of all the modules in the figure.",
        },
        {
            "from": "gpt",
            "value": "%s" %  "[" + ", ".join(data["modules"]) + "]."
        }
    ]
    return [template]


def mask_vqa_templates(data):
    template = [
        {
            "from": "human",
            "value": "Tell me the name of the module masked in the figure."
        },
        {
            "from": "gpt",
            "value": "%s" % data["name"]
        }
    ]
    return template

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

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
