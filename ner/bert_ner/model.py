from transformers import BertModel, BertPreTrainedModel
from crf import CRF
import torch.nn as nn


class BertCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1, bidirectional=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output.permute(1, 0, 2))
        sequence_output = sequence_output.permute(1, 0, 2)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,) + outputs
        return outputs  # (loss), scores