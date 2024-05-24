import json
import os

data_dir = '/home/llmtrainer/Multimodal/EPM/ner/bert_ner/dataset'


def data_loader(path):
    dataset = []
    entity_type = []
    tag = []
    file = open(path, encoding='utf-8')
    res = file.readlines()
    all_idx = 0
    for dic in res:
        jdata = json.loads(dic)
        sentence = jdata['sentence']
        entities = jdata['ner']
        entities = [e for e in entities if e[2] != "pronoun"]
        for e in entities:
            e[0] = e[0]
            e[1] = e[1] + 1
            # if e[2] not in entity_type:
            #     entity_type.append(e[2])
        line = {'sentence': sentence, 'entity': entities, 'id': all_idx}
        all_idx += 1
        dataset.append(line)
    
    entity_type = ["Term"]
    tag.append('O')
    for t in entity_type:
        tag.append('B-' + t)
        tag.append('I-' + t)
    
    entity_to_idx = {i: no + 1 for no, i in enumerate(entity_type)}
    tag_to_idx = {i: no for no, i in enumerate(tag)}
    file.close()
    entity_to_idx['None'] = 0
    return dataset, entity_to_idx, tag_to_idx


def train_data(batch_size, data_mode):
    train_dataset, entity_to_idx, tag_to_idx = data_loader(os.path.join(data_dir, data_mode, 'train.json'))
    last_batch_num = len(train_dataset) % batch_size
    if last_batch_num > 0:
        train_dataset += [train_dataset[ids] for ids in range(batch_size - last_batch_num)]
    return train_dataset, entity_to_idx, tag_to_idx


def eval_data(mode, data_mode):
    if mode == 'dev':
        dataset, _, _ = data_loader(os.path.join(data_dir, data_mode, 'dev.json'))
    else:
        dataset, _, _ = data_loader(os.path.join(data_dir, data_mode, 'test.json'))
    return dataset