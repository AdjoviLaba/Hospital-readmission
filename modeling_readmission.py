import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertConfig, BertModel

class BertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model for classification tasks.
    This class overrides the huggingface implementation to add specific functionality
    for readmission prediction.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_all_encoded_layers=True,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_all_encoded_layers,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # For binary classification with sigmoid
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # For multi-class classification with softmax
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output