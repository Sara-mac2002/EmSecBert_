import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils import data
import os
import pandas as pd
import re

class InputExample(object):
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

class DataProcessor(object):
    @classmethod
    def _read_data(cls, input_file, isPredict=False, sentence=''):
        if isPredict == False:
            with open(input_file) as f:
                out_lists = []
                entries = f.read().strip().split("\n\n")
                for entry in entries:
                    words = []
                    ner_labels = []
                    pos_tags = []
                    bio_pos_tags = []
                    for line in entry.splitlines():
                        pieces = line.strip().split()
                        if len(pieces) < 1:
                            continue
                        word = pieces[0]
                        words.append(word)
                        ner_labels.append(pieces[-1])
                    
                    out_lists.append([words, pos_tags, bio_pos_tags, ner_labels])
        else:
            out_lists = []
            words = []
            ner_labels = []
            pos_tags = []
            bio_pos_tags = []
            entries = sentence.strip().split(" ")
            for i in entries:
                if len(i) < 1:
                    continue
                word = i
                words.append(word)
                ner_labels.append('O')
            out_lists.append([words, pos_tags, bio_pos_tags, ner_labels])
        return out_lists

class CybersecurityNER_DataProcessor(DataProcessor):
    """Data processor for cybersecurity NER"""
    def __init__(self):
        self._label_types = [
            'X',           # For padding/subword tokens
            '[CLS]',       # Start token
            '[SEP]',       # End token  
            'O',           # Outside/Other
            'B-TARGET_ASSET',   # Beginning of target asset
            'I-TARGET_ASSET',   # Inside target asset
            'B-PRECON',         # Beginning of precondition
            'I-PRECON',         # Inside precondition
            'B-MITIGATION',     # Beginning of mitigation
            'I-MITIGATION'      # Inside mitigation
        ]
        
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(self._label_types)}

    def get_predict_examples(self, data_dir, predict_string):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "None.txt"), True, predict_string), True)

    def get_labels(self):
        return self._label_types

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists, isPredict=False):
        examples = []
        if isPredict == False:
            for (i, one_lists) in enumerate(all_lists):
                guid = i
                words = one_lists[0]
                labels = one_lists[-1]
                
                processed_labels = []
                for label in labels:
                    if label in self._label_map:
                        processed_labels.append(label)
                    else:
                        processed_labels.append('O')

                examples.append(InputExample(
                    guid=guid, words=words, labels=processed_labels))
        else:
            k = 1
            for i in all_lists:
                guid = k
                k += 1
                words = i[0]
                labels = i[3]
                examples.append(InputExample(
                    guid=guid, words=words, labels=labels))
        return examples

def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    
    for i, w in enumerate(example.words):
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    if len(tokens) > max_seq_length - 1:
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
        label_ids=label_ids)
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
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))
        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + torch.log(torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))

class BERT_CRF_NER(nn.Module):
    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _forward_alg(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, return_dict=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
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
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1, path[:, t+1].view(-1, 1)).squeeze()
        return max_logLL_allz_allx, path

    def forward(self, input_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids

class EMsecBERTExtractor:
    """
    MITRE ATT&CK Entity Extractor using EMsecBERT
    Extracts TARGET_ASSET, PRECON, and MITIGATION entities from attack descriptions
    """
#  Replace "your_project_directory_path" with your actual project directory path   
    def __init__(self, model_checkpoint_path='your_project_directory_path/Automated_Extraction/model/CySecBert_crf_checkpoint.pt'):
        self.model_checkpoint_path = model_checkpoint_path
        self.max_seq_length = 256
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = 'none.txt'  # Dummy file path for processor
        
        # Initialize components
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the BERT-CRF model and tokenizer"""
        print("Initializing EMsecBERT extractor...")
        
        # Initialize tokenizer and processor
        bert_model_scale = 'markusbayer/CySecBERT'
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_scale, do_lower_case=True)
        self.processor = CybersecurityNER_DataProcessor()
        
        # Get label information
        self.start_label_id = self.processor.get_start_label_id()
        self.stop_label_id = self.processor.get_stop_label_id()
        self.label_list = self.processor.get_labels()
        self.label_map = self.processor.get_label_map()
        
        # Initialize model
        bert_model = AutoModel.from_pretrained(bert_model_scale)
        self.model = BERT_CRF_NER(
            bert_model, 
            self.start_label_id, 
            self.stop_label_id, 
            len(self.label_list), 
            self.max_seq_length, 
            16,  # batch_size
            self.device
        )
        
        # Load checkpoint
        print(f"Loading model checkpoint from: {self.model_checkpoint_path}")
        checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')
        
        pretrained_dict = checkpoint['model_state']
        net_state_dict = self.model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        self.model.load_state_dict(net_state_dict)
        
        print(f'Loaded EMsecBERT model - epoch: {checkpoint["epoch"]}, valid acc: {checkpoint["valid_acc"]:.4f}, valid f1: {checkpoint["valid_f1"]:.4f}')
        
        self.model.to(self.device)
        self.model.eval()
        
    def split_text_into_sentences(self, text):
        """
        Split text into sentences using regex patterns
        Handles common sentence endings and abbreviations
        """
        # Clean the text
        text = text.strip()
        if not text:
            return []
        
        # Pattern for sentence splitting
        # Looks for . ! ? followed by whitespace and capital letter or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    
        
        # Split into sentences
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences with at least 3 words to avoid fragments
            if sentence and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        # If no sentences found or text too short, return original as one sentence
        if not cleaned_sentences:
            return [text]
        
        return cleaned_sentences

    def extract_entities_from_sentence(self, sentence):
        """
        Extract entities from a single sentence
        Returns: dict with 'target_assets', 'preconditions', 'mitigations'
        """
        predict_examples = self.processor.get_predict_examples(self.data_dir, predict_string=sentence)
        predict_dataset = NerDataset(predict_examples, self.tokenizer, self.label_map, self.max_seq_length)
        
        with torch.no_grad():
            dataloader = data.DataLoader(
                dataset=predict_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=NerDataset.pad
            )
            
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                _, predicted_label_seq_ids = self.model(input_ids, segment_ids, input_mask)
                
                # Get predictions for valid tokens only
                new_ids = predicted_label_seq_ids[0].cpu().numpy()[predict_mask[0].cpu().numpy() == 1]
                predicted_labels = [self.label_list[i] for i in new_ids]
                
                # Extract words
                words = sentence.strip().split()
                
                # Extract entities using BIO tagging
                entities = self._extract_entities_from_bio(words, predicted_labels)
                return entities

    def extract_entities_from_description(self, description):
        """
        Extract entities from a full description by splitting into sentences
        and processing each sentence separately
        Returns: combined dict with 'target_assets', 'preconditions', 'mitigations'
        """
        print(f"Processing description: {description[:100]}...")
        
        # Split description into sentences
        sentences = self.split_text_into_sentences(description)
        print(f"Split into {len(sentences)} sentences")
        
        # Initialize combined results
        combined_entities = {
            'target_assets': [],
            'preconditions': [],
            'mitigations': []
        }
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            print(f"  Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Extract entities from this sentence
            sentence_entities = self.extract_entities_from_sentence(sentence)
            
            # Combine results (avoid duplicates)
            for entity_type in combined_entities:
                for entity in sentence_entities[entity_type]:
                    if entity not in combined_entities[entity_type]:
                        combined_entities[entity_type].append(entity)
        
        return combined_entities
    
    def _extract_entities_from_bio(self, words, labels):
        """Extract entities from BIO tagged sequence"""
        entities = {
            'target_assets': [],
            'preconditions': [],
            'mitigations': []
        }
        
        current_entity = None
        current_type = None
        
        for word, label in zip(words, labels):
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    self._add_entity_to_dict(entities, current_type, current_entity)
                
                # Start new entity
                current_type = label[2:]  # Remove 'B-' prefix
                current_entity = word
                
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                if label[2:] == current_type:  # Same entity type
                    current_entity += ' ' + word
                else:
                    # Different entity type, save current and start new
                    self._add_entity_to_dict(entities, current_type, current_entity)
                    current_type = label[2:]
                    current_entity = word
                    
            else:
                # End current entity
                if current_entity:
                    self._add_entity_to_dict(entities, current_type, current_entity)
                    current_entity = None
                    current_type = None
        
        # Save last entity if exists
        if current_entity:
            self._add_entity_to_dict(entities, current_type, current_entity)
        
        return entities
    
    def _add_entity_to_dict(self, entities, entity_type, entity_text):
        """Add entity to the appropriate list"""
        if entity_type == 'TARGET_ASSET':
            entities['target_assets'].append(entity_text)
        elif entity_type == 'PRECON':
            entities['preconditions'].append(entity_text)
        elif entity_type == 'MITIGATION':
            entities['mitigations'].append(entity_text)
    
    def process_mitre_data(self, mitre_data, output_file_base="MITRE_Extracted_Entities"):
        """
        Process MITRE data and extract entities from descriptions
        Now splits long descriptions into sentences for better processing
        
        Args:
            mitre_data: List of dictionaries with 'Name' and 'Description' keys
            output_file_base: Base name for output files
        """
        print(f"Processing {len(mitre_data)} MITRE techniques...")
        
        results = []
        
        for i, technique in enumerate(mitre_data):
            print(f"Processing {i+1}/{len(mitre_data)}: {technique['Name']}")
            
            # Extract entities from description using sentence splitting
            entities = self.extract_entities_from_description(technique['Description'])
            
            # Create result record
            result = {
                'Name': technique['Name'],
                'Description': technique['Description'],
                'Extracted_Target_Assets': ', '.join(entities['target_assets']) if entities['target_assets'] else 'N/A',
                'Extracted_Preconditions': ', '.join(entities['preconditions']) if entities['preconditions'] else 'N/A',
                'Extracted_Mitigations': ', '.join(entities['mitigations']) if entities['mitigations'] else 'N/A'
            }
            
            results.append(result)
        
        # Save results
        self._save_results(results, output_file_base)
        
        return results
    
    def _save_results(self, results, output_file_base):
        """Save results to CSV and Excel files"""
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_file = f"{output_file_base}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
        
        # Save to Excel
        excel_file = f"{output_file_base}.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"Results saved to {excel_file}")
def main():
    """Load and process collected MITRE data"""
    # Initialize extractor
    extractor = EMsecBERTExtractor()
    
    print("Loading collected MITRE data...")
    #  Replace "your_project_directory_path" with your actual project directory path   
   
    # Load ICS data
    print("Loading ICS data...")
    ics_data = pd.read_csv('your_project_directory_path/data/mitre/MITRE_ATT&CK_ICS.csv')
    ics_data_list = ics_data.to_dict('records')
    
    # Load Enterprise data
    print("Loading Enterprise data...")
    enterprise_data = pd.read_csv('your_project_directory_path/data/mitre/MITRE_ATT&CK_Enterprise.csv')
    enterprise_data_list = enterprise_data.to_dict('records')
    
    # Process ICS data
    print(f"Processing {len(ics_data_list)} ICS techniques...")
    ics_results = extractor.process_mitre_data(ics_data_list, "ICS_Extracted_Entities")
    
    # Process Enterprise data
    print(f"Processing {len(enterprise_data_list)} Enterprise techniques...")
    enterprise_results = extractor.process_mitre_data(enterprise_data_list, "Enterprise_Extracted_Entities")
    
    print("All processing completed successfully!")
    print(f"ICS results: {len(ics_results)} techniques processed")
    print(f"Enterprise results: {len(enterprise_results)} techniques processed")


if __name__ == "__main__":
    main()
