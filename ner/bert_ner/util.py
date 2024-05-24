import logging
import torch
from configs import train_config
from nltk import word_tokenize


# write log
def log_writer(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# attention mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    return atten_masks


def bpe_encode_one(text, tokenizer):
    sentence = word_tokenize(text)
    start2idx = []
    end2idx = []
    bert_tokens = []
    bert_tokens += [tokenizer.cls_token]

    for token in sentence:
        start2idx.append(len(bert_tokens))
        sub_tokens = tokenizer.tokenize(token)
        if len(sub_tokens) == 0:
            sub_tokens = [token]
        bert_tokens += sub_tokens
        end2idx.append(len(bert_tokens))
    
    bert_tokens += [tokenizer.sep_token]
    input_id = tokenizer.convert_tokens_to_ids(bert_tokens)
    encoding = torch.tensor([input_id])
    attention_mask = torch.tensor(attention_masks([input_id]))
    return encoding, attention_mask, sentence, start2idx, end2idx, bert_tokens




# BPE algorithm
def bpe_encode(batch_sentences, batch_entities, tokenizer, tag_to_idx):
    input_ids = []
    max_len = 0
    batch_bert_tokens = []
    new_batch_entities = []
    for num, sentence in enumerate(batch_sentences):
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens += [tokenizer.cls_token]

        for token in sentence:
            start2idx.append(len(bert_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) == 0:
                sub_tokens = [token]
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens))

        bert_tokens += [tokenizer.sep_token]
        batch_bert_tokens.append(bert_tokens)

        input_id = tokenizer.convert_tokens_to_ids(bert_tokens)
        max_len = max(max_len, len(input_id))
        input_ids.append(input_id)

        new_entities = [[start2idx[e[0]], end2idx[e[1] - 1], e[2]] for e in batch_entities[num]]
        new_batch_entities.append(new_entities)

    new_input_ids = []
    for input_id in input_ids:
        new_input_id = input_id + [tokenizer.pad_token_id for _ in range(max_len - len(input_id))]
        new_input_ids.append(new_input_id)

    encoding = torch.tensor(new_input_ids)
    gold_tags = token_to_tag(new_input_ids, new_batch_entities, tag_to_idx)
    return encoding, new_input_ids, gold_tags


def token_to_tag(input_ids, entities, tag_to_idx):
    tags = torch.tensor([])
    for num, inputs in enumerate(input_ids):
        tag = torch.zeros(len(inputs))
        for e in entities[num]:
            e[2] = "Term"
            tag[e[0]] = tag_to_idx['B-' + e[2]]
            for loc in range(e[0] + 1, e[1]):
                tag[loc] = tag_to_idx['I-' + e[2]]
        tags = torch.cat([tags, tag.unsqueeze(0)])
    tags = tags.long()
    return tags


def tag_to_entity(tag, tag_to_idx, entity_to_idx):
    entity = []
    tag_bio = []
    for t in tag:
        for i, j in tag_to_idx.items():
            if t == j:
                tag_bio.append(i)
    start = end = 0
    types = 0
    for n, t in enumerate(tag_bio):
        entity_tag = t.replace('B-', '').replace('I-', '')
        if 'B' in t:
            if types != 0:
                entity.append([start, end + 1, types])
            types = entity_to_idx[entity_tag]
            start = end = n
        elif 'I' in t and n != 0 and entity_tag in tag_bio[n - 1]:
            end += 1
        else:
            if types != 0:
                entity.append([start, end + 1, types])
            types = 0
    if types != 0:
        entity.append([start, end + 1, types])
    return entity


def entity_to_tag(input_ids, entities, tag_to_idx, entity_to_idx):
    tags = torch.tensor([])
    entity_to_idx_inv = {value: key for key, value in entity_to_idx.items()}
    for num, inputs in enumerate(input_ids):
        tag = torch.zeros(len(inputs))
        for e in entities[num]:
            tag[e[0]] = tag_to_idx['B-' + entity_to_idx_inv[e[2]]]
            for loc in range(e[0] + 1, e[1]):
                tag[loc] = tag_to_idx['I-' + entity_to_idx_inv[e[2]]]
        tags = torch.cat([tags, tag.unsqueeze(0)])
    tags = tags.long()
    return tags