# text_processor.py
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sentencepiece as spm
from typing import List, Dict
import random

class TextProcessor:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        model_path = self.processed_dir / 'tokenizer.model'
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found at {model_path}")
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(model_path))

class TextDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer: spm.SentencePieceProcessor, max_len: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # Load data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Add special tokens
        tokens = [self.tokenizer.bos_id()] + tokens + [self.tokenizer.eos_id()]
        
        # Truncate or pad
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_id()] * (self.max_len - len(tokens))
        
        # Create attention mask
        attention_mask = [1 if t != self.tokenizer.pad_id() else 0 for t in tokens]
        
        # Create target tokens (shifted right by 1)
        target_tokens = tokens[1:] + [self.tokenizer.pad_id()]
        
        return {
            'src_tokens': torch.tensor(tokens, dtype=torch.long),
            'tgt_tokens': torch.tensor(target_tokens, dtype=torch.long),
            'attn_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

def process_text_files(base_dir: Path, vocab_size: int = 32000, max_len: int = 512):
    base_dir = Path(base_dir)
    raw_dir = base_dir / 'raw'
    processed_dir = base_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Collect all text
    all_text = []
    for text_file in raw_dir.glob('*.txt'):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split into smaller chunks if text is too long
            chunks = [text[i:i+max_len*4] for i in range(0, len(text), max_len*4)]
            all_text.extend(chunks)
    
    # Train tokenizer
    with open(processed_dir / 'train.txt', 'w', encoding='utf-8') as f:
        for text in all_text:
            f.write(text.strip() + '\n')
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=str(processed_dir / 'train.txt'),
        model_prefix=str(processed_dir / 'tokenizer'),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]'
    )
    
    # Process and save data
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(processed_dir / 'tokenizer.model'))
    
    with open(processed_dir / 'dataset.jsonl', 'w', encoding='utf-8') as f:
        for text in all_text:
            if text.strip():  # Skip empty texts
                f.write(json.dumps({'text': text.strip()}) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()
    
    process_text_files(Path(args.base_dir), args.vocab_size, args.max_len)
