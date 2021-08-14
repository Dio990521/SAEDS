"""
Modified bert for multi-class classification. Source Hugginface, modified by Chenyang, Dec, 2019
"""

from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from module.binary_decoder import BinaryDecoder
from module.bert_encoder import BertEncoder


class BinaryBertClassifier(nn.Module):
    def __init__(self, hidden_dim, num_label, args):
        super(BinaryBertClassifier, self).__init__()
        self.encoder = None
        self.decoder = BinaryDecoder(hidden_dim, num_label)
        self.args = args
        self.dropout = None

    def init_encoder(self, BERT_model):
        self.encoder = BertEncoder.from_pretrained(BERT_model)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        outputs = self.encoder(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        pooled_output = outputs[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pred = self.decoder(pooled_output)

        return pred
