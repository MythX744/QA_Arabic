# dataset.py

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicQADataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_length: int = 128,
            is_training: bool = True,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize the ArabicQA dataset.

        Args:
            data_path: Path to the json data file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            is_training: Whether this is for training or evaluation
            cache_dir: Directory to cache processed features
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer

        # Ensure tokenizer has padding token and proper settings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Set padding to left side

        self.max_length = max_length
        self.is_training = is_training
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Load and process data
        self.examples = self.load_data()
        self.features = self.prepare_features()

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def load_data(self) -> List[Dict]:
        """Load and validate the raw data from json file."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate data structure
            validated_data = []
            for idx, item in enumerate(data):
                if self._validate_example(item, idx):
                    validated_data.append(item)

            return validated_data

        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {self.data_path}")
            raise

    def _validate_example(self, example: Dict, idx: int) -> bool:
        """Validate individual example structure."""
        required_fields = ['question_id', 'answer_id', 'question', 'answer']

        for field in required_fields:
            if field not in example:
                logger.warning(f"Missing {field} in example {idx}")
                return False
            if not isinstance(example[field], str):
                logger.warning(f"Invalid {field} type in example {idx}")
                return False
            if not example[field].strip():
                logger.warning(f"Empty {field} in example {idx}")
                return False

        return True

    def prepare_features(self) -> List[Dict]:
        """Prepare features from examples."""
        features = []

        for example in self.examples:
            # Tokenize question and answer
            question = example['question'].strip()
            answer = example['answer'].strip()

            # Create feature dictionary
            feature = {
                'question_id': example['question_id'],
                'answer_id': example['answer_id'],
                'question': question,
                'answer': answer
            }

            # Add encodings
            encodings = self._encode_qa_pair(question, answer)
            feature.update(encodings)

            features.append(feature)

        return features

    def _encode_qa_pair(self, question: str, answer: str) -> Dict:
        """Encode question-answer pair using tokenizer."""
        # Encode question and answer together
        encoded = self.tokenizer(
            question,
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Remove batch dimension
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        feature = self.features[idx]

        item = {
            'input_ids': feature['input_ids'],
            'attention_mask': feature['attention_mask'],
        }

        if self.is_training:
            # For training, encode answer as labels
            answer_encoding = self.tokenizer(
                feature['answer'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['labels'] = answer_encoding['input_ids'].squeeze(0)

        return item

    def get_example_text(self, idx: int) -> Dict[str, str]:
        """Get original text of an example by index."""
        feature = self.features[idx]
        return {
            'question_id': feature['question_id'],
            'answer_id': feature['answer_id'],
            'question': feature['question'],
            'answer': feature['answer']
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader."""
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch]) if 'labels' in batch[0] else None
        }
