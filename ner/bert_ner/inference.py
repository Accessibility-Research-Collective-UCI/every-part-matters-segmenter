import sys
sys.path.append("/home/llmtrainer/Multimodal/EPM/ner/bert_ner")
from transformers import BertTokenizer, AdamW, BertConfig
from ner.bert_ner.model import BertCrf
import torch
from ner.bert_ner.util import bpe_encode_one, tag_to_entity
from nltk import sent_tokenize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# path to base model (SciBert)
BASE_MODEL = "/data_share/model_hub/scibert"
DEVICE = "cuda:0"
CHECKPOINT = "/home/llmtrainer/Multimodal/EPM/ner/bert_ner/checkpoint/bert_crf.pth"
LOG = "/home/llmtrainer/Multimodal/EPM/ner/bert_ner/log/ner.log"

# entity_to_idx = {'application': 1, 'method': 2, 'pronoun': 3, 'symbol': 4, 'other': 5, 'task': 6, 'None': 0}
entity_to_idx = {'Term': 1, 'None': 0}
tag_to_idx = {'O': 0, "B-Term": 1, "I-Term": 2}

# tag_to_idx = {'O': 0, 'B-application': 1, 'I-application': 2, 'B-method': 3, 'I-method': 4, 'B-pronoun': 5, 'I-pronoun': 6, 'B-symbol': 7, 'I-symbol': 8, 'B-other': 9, 'I-other': 10, 'B-task': 11, 'I-task': 12}

configs = BertConfig.from_pretrained(BASE_MODEL, num_labels=len(tag_to_idx))
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
model = BertCrf.from_pretrained(BASE_MODEL, config=configs)
model.to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])

def ner(text):
    model.eval()
    entities = []
    for sent in sent_tokenize(text):
        encoding, attention_mask, sentence, start2idx, end2idx, bert_tokens = bpe_encode_one(sent, tokenizer)
        encoding = encoding.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        outputs = model(input_ids=encoding, attention_mask=attention_mask)
        logits = outputs[0]

        pre_tag = model.crf.decode(logits, attention_mask)
        pre_tag = pre_tag.tolist()
        pre_entity = tag_to_entity(pre_tag[0][0], tag_to_idx, entity_to_idx)

        for entity_idx in pre_entity:
            # print(entity_idx)
            # print(start2idx)
            try:
                start_id = start2idx.index(entity_idx[0])
                end_id = end2idx.index(entity_idx[1]) + 1
                if entity_idx[2] == 3:
                    continue
                entities.append(" ".join(sentence[start_id: end_id]))
            except:
                continue
    
    return entities

if __name__ == "__main__":
    ner("The VoiceTRAN Communicator is composed of a number of servers that interact with each other through the Hub as shown in Figure 1. The Hub is used as a centralized message router through which servers can communicate with one another. Frames containing keys and values are emitted by each server. They are routed by the hub and received by a secondary server based on rules defined in the Hub script.")




