"""Data loading utilities for WikiText-2 and SST-2 datasets."""

from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple

class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Tokenizing {len(texts)} examples...")
        for text in texts:
            if isinstance(text, str) and text.strip():
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                self.examples.append(encoding['input_ids'].squeeze(0))
        
        print(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def get_wikitext2_dataloaders(batch_size=32, max_length=512):
    """Load WikiText-2 dataset."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = TextDataset(dataset['train']['text'], tokenizer, max_length)
    val_dataset = TextDataset(dataset['validation']['text'], tokenizer, max_length)
    test_dataset = TextDataset(dataset['test']['text'], tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == '__main__':
    # Test
    print("Testing data loading...")
    train_loader, val_loader, test_loader, tokenizer = get_wikitext2_dataloaders(batch_size=4, max_length=128)
    batch = next(iter(train_loader))
    print(f"✓ Batch shape: {batch.shape}")
    print(f"✓ Sample text: {tokenizer.decode(batch[0][:50])}")