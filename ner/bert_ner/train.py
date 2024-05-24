import sys
sys.path.append("/home/llmtrainer/Multimodal/EPM/ner/bert_ner")
from transformers import BertTokenizer, AdamW, BertConfig
import transformers
import torch
import data_loader
from model import BertCrf
from util import log_writer, bpe_encode, attention_masks, tag_to_entity
import torch.nn as nn
from configs import train_config
from tqdm import tqdm
import random
import numpy as np


#  train
def train(train_mode, data_mode):

    set_seed(train_config['SEED'])

    # initialize model and data
    logger = log_writer('/home/llmtrainer/Multimodal/EPM/ner/bert_ner/log/bert_crf.log')
    tokenizer = BertTokenizer.from_pretrained('/data_share/model_hub/scibert')
    train_dataset, entity_to_idx, tag_to_idx = data_loader.train_data(train_config['BATCH_SIZE'], data_mode)
    print(entity_to_idx)
    print(tag_to_idx)
    configs = BertConfig.from_pretrained('/data_share/model_hub/scibert', num_labels=len(tag_to_idx))
    model = BertCrf.from_pretrained('/data_share/model_hub/scibert', config=configs)
    model.to(train_config['DEVICE'])

    # Prepare optimizer and schedule (linear warmup and decay)
    learning_rate = train_config['LEARNING_RATE']
    crf_learning_rate = train_config['CRF_LEARNING_RATE']
    lstm_learning_rate = train_config['LSTM_LEARNING_RATE']
    weight_decay = train_config['WEIGHT_DECAY']
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    lstm_param_optimizer = list(model.lstm.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': learning_rate},

        {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lstm_learning_rate},
        {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': lstm_learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': learning_rate}
    ]
    batch_size = train_config['BATCH_SIZE']
    total_batch_count = int(len(train_dataset) / batch_size)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=train_config['LR_WARMUP'] *
                                                                              total_batch_count * train_config['EPOCH'],
                                                             num_training_steps=total_batch_count * train_config[
                                                                 'EPOCH'])
    

    # load existing model
    start_epoch = -1
    f_max = 0
    if train_mode == 'C':
        checkpoint = torch.load('/home/llmtrainer/Multimodal/EPM/ner/bert_ner/checkpoint/bert_crf.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    # train model
    for epoch in range(start_epoch + 1, train_config['EPOCH']):
        logger.info('--------Epoch: %d--------' % epoch)
        for batch_num in tqdm(range(total_batch_count)):
            model.train()
            model.zero_grad()
            batch_train_data = train_dataset[
                                   batch_num * train_config['BATCH_SIZE']: (batch_num + 1) * train_config['BATCH_SIZE']]
            batch_sentences = [line['sentence'] for line in batch_train_data]
            batch_entities = [line['entity'] for line in batch_train_data]

            encoding, input_ids, gold_tags = bpe_encode(batch_sentences, batch_entities, tokenizer, tag_to_idx)
            encoding = encoding.to(train_config['DEVICE'])
            attention_mask = torch.tensor(attention_masks(input_ids)).to(train_config['DEVICE'])
            gold_tags = gold_tags.to(train_config['DEVICE'])

            outputs = model(input_ids=encoding, attention_mask=attention_mask, labels=gold_tags)
            loss = outputs[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # log
        print('Epoch: %d Finished' % epoch)
        r, p, f = evaluation(model, mode='dev', logger=logger, tokenizer=tokenizer,
                                 data_mode=data_mode, entity_to_idx=entity_to_idx, tag_to_idx=tag_to_idx)
        print('dev-Entity: Recall - %f ; Precision - %f ; F-measure - %f' % (r, p, f))
        if f > f_max:
            f_max = f
            r, p, f = evaluation(model, mode='test', logger=logger, tokenizer=tokenizer,
                                     data_mode=data_mode, entity_to_idx=entity_to_idx, tag_to_idx=tag_to_idx)
            print('test-Entity: Recall - %f ; Precision - %f ; F-measure - %f' % (r, p, f))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                         'scheduler': scheduler.state_dict()}
            torch.save(state, '/home/llmtrainer/Multimodal/EPM/ner/bert_ner/checkpoint/bert_crf.pth')
            torch.save(model, '/home/llmtrainer/Multimodal/EPM/ner/bert_ner/checkpoint/bert_crf.pkl')


# evaluation
def evaluation(model, mode, logger, tokenizer, data_mode, entity_to_idx, tag_to_idx):
    dataset = data_loader.eval_data(mode, data_mode)
    model.eval()
    with torch.no_grad():
        pre_et = 0
        gold_et = 0
        acc_et = 0
        for data in dataset:
            sentence = data['sentence']
            entity = data['entity']
            encoding, input_ids, gold_tags = bpe_encode([sentence], [entity], tokenizer, tag_to_idx)
            encoding = encoding.to(train_config['DEVICE'])
            attention_mask = torch.tensor(attention_masks(input_ids)).to(train_config['DEVICE'])
            gold_tags = gold_tags.to(train_config['DEVICE'])
            outputs = model(input_ids=encoding, attention_mask=attention_mask, labels=gold_tags)
            _, logits = outputs[:2]
            pre_tag = model.crf.decode(logits, attention_mask)
            pre_tag = pre_tag.tolist()
            pre_entity = tag_to_entity(pre_tag[0][0], tag_to_idx, entity_to_idx)

            gold_tags = gold_tags.tolist()
            gold_entity = tag_to_entity(gold_tags[0], tag_to_idx, entity_to_idx)
            gold_et += len(gold_entity)

            for e in pre_entity:
                pre_et += 1
                if e in gold_entity:
                    acc_et += 1

        recall = acc_et / gold_et
        if pre_et > 0:
            precision = acc_et / pre_et
            if recall > 0:
                f_measure = 2 / (1 / precision + 1 / recall)
            else:
                f_measure = 0
        else:
            precision = 0
            f_measure = 0
        recall = round(recall * 100, 2)
        precision = round(precision * 100, 2)
        f_measure = round(f_measure * 100, 2)
        print("acc_ent: %d, pre_ent: %d, gold_ent: %d" % (acc_et, pre_et, gold_et))
        logger.info(mode + '-Entity: Recall - %s ; Precision - %s ; F-measure - %s' % (str(recall), str(precision)
                                                                                       , str(f_measure)))
    return recall, precision, f_measure


def predict(data_mode):
    train_dataset, entity_to_idx, tag_to_idx = data_loader.train_data(train_config['BATCH_SIZE'], data_mode)
    configs = BertConfig.from_pretrained('/data_share/model_hub/scibert', num_labels=len(tag_to_idx))
    model = BertCrf.from_pretrained('/data_share/model_hub/scibert', config=configs)
    model.to(train_config['DEVICE'])
    checkpoint = torch.load('/home/llmtrainer/Multimodal/EPM/ner/bert_ner/checkpoint/bert_crf.pth')
    model.load_state_dict(checkpoint['model'])
    logger = log_writer('/home/llmtrainer/Multimodal/EPM/ner/bert_ner/log/bert_crf.log')
    tokenizer = BertTokenizer.from_pretrained('/data_share/model_hub/scibert')

    r, p, f = evaluation(model, mode='test', logger=logger, tokenizer=tokenizer,
                                     data_mode=data_mode, entity_to_idx=entity_to_idx, tag_to_idx=tag_to_idx)
    print('test-Entity: Recall - %f ; Precision - %f ; F-measure - %f' % (r, p, f))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    #  train_mode: C-train based on previous model / I-new train
    #  data_mode: IV_WCL / IV_SYM / IV_AI
    train(train_mode='I', data_mode='IV_AI')
    # predict("IV_AI")