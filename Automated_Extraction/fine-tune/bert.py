# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import numpy as np
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer
import argparse

"""
Note: In this script, "training" refers to fine-tuning a pre-trained BERT model,
not training from scratch. The pre-trained BERT weights are loaded and adapted
for cybersecurity NER tasks.
"""
#  Replace "your_project_directory_path" with your actual project directory path

# Command line arguments for flexibility
parser = argparse.ArgumentParser()
parser.add_argument('--mitre_ics_data_dir', type=str, default='your_project_directory_path/Automated_Extraction/Datasets/Mitre_ICS/', 
                    help='Directory containing MITRE ICS dataset')
parser.add_argument('--mitre_enterprise_data_dir', type=str, default='your_project_directory_path/Automated_Extraction/Datasets/Mitre_ENTREPRISE/', 
                    help='Directory containing MITRE Enterprise dataset')
parser.add_argument('--paper_data_dir', type=str, default='your_project_directory_path/Automated_Extraction/Datasets/Paper_report/', 
                    help='Directory containing Paper report dataset')
parser.add_argument('--output_dir', type=str, default='your_project_directory_path/Automated_Extraction/models/bert_crf/', 
                    help='Output directory for model')
parser.add_argument('--bert_model', type=str, default='bert-base-cased', 
                    help='Pre-trained BERT model to use')
parser.add_argument('--max_seq_length', type=int, default=256, 
                    help='Maximum sequence length')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, 
                    help='Learning rate')
parser.add_argument('--num_train_epochs', type=int, default=10, 
                    help='Number of training epochs')

args = parser.parse_args()

# Configuration
max_seq_length = args.max_seq_length
output_dir = args.output_dir
batch_size = args.batch_size
gradient_accumulation_steps = 1
total_train_epochs = args.num_train_epochs

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_data_dir(data_path=None):
    """Get data directory path"""
    if data_path and os.path.exists(data_path):
        return data_path
    else:
        raise Exception(f'Data path not found: {data_path}')

class InputExample(object):
    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels: list of label sequence of the sentence
        """
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 2:
                        continue
                    word = pieces[0]
                    label = pieces[-1]  # Last element is the label
                    words.append(word)
                    ner_labels.append(label)
                if words:  # Only add non-empty sentences
                    out_lists.append([words, [], [], ner_labels])  # Keep format consistent
        return out_lists

class CybersecurityNER_DataProcessor(DataProcessor):
    """
    Data processor for cybersecurity NER with 7 labels
    """
    def __init__(self):
        # Your 7 labels based on the sample data
        self._label_types = [
            'X',           # For padding/subword tokens
            '[CLS]',       # Start token
            '[SEP]',       # End token  
            'O',           # Outside/Other
            'B-TARGET_ASSET',   # Beginning of target asset
            'I-TARGET_ASSET',   # Inside target asset
            'B-PRECON',         # Beginning of precondition
            'I-PRECON',         # Inside precondition
            'B-MITIGATION',     # Beginning of mitigation (if you have this)
            'I-MITIGATION'      # Inside mitigation (if you have this)
        ]
        
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(self._label_types)}

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "valid.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self._num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            
            # Handle unknown labels by mapping them to 'O'
            processed_labels = []
            for label in labels:
                if label in self._label_map:
                    processed_labels.append(label)
                else:
                    print(f"Warning: Unknown label '{label}' found, mapping to 'O'")
                    processed_labels.append('O')
            
            examples.append(InputExample(
                guid=guid, words=words, labels=processed_labels))
        return examples

def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    
    for i, w in enumerate(example.words):
        # Use BERT tokenizer to split words
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # Subword tokens get 'X' label
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # Truncate if too long
    if len(tokens) > max_seq_length - 1:
        print(f'Example No.{example.guid} is too long, length is {len(tokens)}, truncated to {max_seq_length}!')
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask,
        label_ids=label_ids
    )

    return feat

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.BoolTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

def f1_score(y_true, y_pred):
    """
    Calculate F1 score for NER
    0,1,2,3 are [CLS],[SEP],[X],O - ignore these for evaluation
    """
    ignore_id = 3  # 'O' label index

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1

# CRF helper functions
def log_sum_exp_1vec(vec):
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_mat(log_M, axis=-1):
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))

def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))

# BERT-CRF Model
class BERT_CRF_NER(nn.Module):
    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device

        # Use pre-trained BERT model
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps BERT output to label space
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))

        # Enforce constraints: never transition TO start or FROM stop
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        """Forward algorithm for CRF"""
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0

        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        """Get BERT features"""
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, return_dict=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        """Score a sentence with given labels"""
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t-1]).view(-1, 1)) \
                + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        """Viterbi decoding"""
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)
        
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # Trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        gold_score = self._score_sentence(bert_feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, input_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids

# Initialize processor and load data from all three datasets
processor = CybersecurityNER_DataProcessor()
label_list = processor.get_labels()
label_map = processor.get_label_map()

print("Loading MITRE ICS training data...")
mitre_ics_train_examples = processor.get_train_examples(args.mitre_ics_data_dir)
print(f"Loaded {len(mitre_ics_train_examples)} training examples from MITRE ICS")

print("Loading MITRE Enterprise training data...")
mitre_enterprise_train_examples = processor.get_train_examples(args.mitre_enterprise_data_dir)
print(f"Loaded {len(mitre_enterprise_train_examples)} training examples from MITRE Enterprise")

print("Loading Paper Report training data...")
paper_train_examples = processor.get_train_examples(args.paper_data_dir)
print(f"Loaded {len(paper_train_examples)} training examples from Paper Report")

# Combine all training data
train_examples = mitre_ics_train_examples + mitre_enterprise_train_examples + paper_train_examples
print(f"Combined training examples: {len(train_examples)}")

print("Loading validation data...")
mitre_ics_dev_examples = processor.get_dev_examples(args.mitre_ics_data_dir)
mitre_enterprise_dev_examples = processor.get_dev_examples(args.mitre_enterprise_data_dir)
paper_dev_examples = processor.get_dev_examples(args.paper_data_dir)

# Combine all validation data
dev_examples = mitre_ics_dev_examples + mitre_enterprise_dev_examples + paper_dev_examples
print(f"Combined validation examples: {len(dev_examples)}")

print("Loading test data...")
mitre_ics_test_examples = processor.get_test_examples(args.mitre_ics_data_dir)
mitre_enterprise_test_examples = processor.get_test_examples(args.mitre_enterprise_data_dir)
paper_test_examples = processor.get_test_examples(args.paper_data_dir)

# Combine all test data
test_examples = mitre_ics_test_examples + mitre_enterprise_test_examples + paper_test_examples
print(f"Combined test examples: {len(test_examples)}")

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

print("***** Training Configuration *****")
print(f"  Num examples = {len(train_examples)}")
print(f"  Batch size = {batch_size}")
print(f"  Num steps = {total_train_steps}")
print(f"  Labels = {label_list}")

# Initialize tokenizer and datasets
bert_model_scale = args.bert_model
tokenizer = AutoTokenizer.from_pretrained(bert_model_scale, do_lower_case=True)

train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length)
dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length)
test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=NerDataset.pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=NerDataset.pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=NerDataset.pad)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
start_label_id = processor.get_start_label_id()
stop_label_id = processor.get_stop_label_id()
bert_model = BertModel.from_pretrained(args.bert_model)
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)

start_epoch = 0
valid_acc_prev = 0
valid_f1_prev = 0
model.to(device)

# Optimizer setup
learning_rate0 = args.learning_rate
lr0_crf_fc = 8e-5
weight_decay_crf_fc = 5e-6
weight_decay_finetune = 1e-5

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
        , 'lr': lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
    {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
        , 'lr': lr0_crf_fc, 'weight_decay': 0.0}
]

optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate0)

# Evaluation function
import time

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.time()
    
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    
    print(f'Epoch:{epoch_th}, Acc:{100.*test_acc:.2f}, Precision: {100.*precision:.2f}, Recall: {100.*recall:.2f}, F1: {100.*f1:.2f} on {dataset_name}, Time:{(end-start)/60.0:.3f} min')
    print('--------------------------------------------------------------')
    return test_acc, f1

# Training loop
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
warmup_proportion = 0.1

print("Starting training...")
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

        neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

        neg_log_likelihood.backward()
        tr_loss += neg_log_likelihood.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1

    print('--------------------------------------------------------------')
    print(f"Epoch:{epoch} completed, Total training Loss: {tr_loss:.4f}, Time: {(time.time() - train_start)/60.0:.2f}m")
    
    # Evaluate on validation set
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

    # Save best model
    if valid_f1 > valid_f1_prev:
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                   'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': False},
                   os.path.join(output_dir, 'cybersecurity_ner_bert_crf_checkpoint.pt'))
        valid_f1_prev = valid_f1
        print(f"New best model saved with F1: {valid_f1:.4f}")

# Final evaluation on test set
print("Final evaluation on test set:")
evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')
print("Training completed!")