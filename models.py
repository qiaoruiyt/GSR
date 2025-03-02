import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertModel
from transformers import BertForSequenceClassification, BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, model, num_classes) -> None:
        super(BertClassifier, self).__init__()
        self.featurizer = BertFeaturizer.from_pretrained(model)
        self.classifier = torch.nn.Linear(self.featurizer.d_out, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        out = self.classifier(x)
        return out


class BertFeaturizer(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1] # get pooled output
        return outputs


class DistilBertClassifier(nn.Module):
    def __init__(self, model, num_classes) -> None:
        super(DistilBertClassifier, self).__init__()
        self.featurizer = DistilBertFeaturizer.from_pretrained(model)
        self.classifier = torch.nn.Linear(self.featurizer.d_out, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        out = self.classifier(x)
        return out


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size
        
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output
