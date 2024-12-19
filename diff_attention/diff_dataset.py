# In dataset.py (create this if it doesn't exist)
import torch
from torch.utils.data import Dataset
import json


class ArabicQADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, is_training=True, device='cuda'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.device = device

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        combined_text = f"Question: {question} Answer: {answer}"
        encodings = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        # Create the return dictionary
        encoded = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return encoded

    def collate_fn(self, batch):
        # Collect all tensors in batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }