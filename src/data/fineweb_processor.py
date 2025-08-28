"""
FineWeb Educational Data Processor for DeepSeek Model
Handles dataset loading, preprocessing, and tokenization for educational web content
"""

import tiktoken
import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from typing import Dict, List, Optional

def load_encoder_decoder():
    """Load the encoder and decoder for text processing"""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc

class FineWebDataProcessor:
    def __init__(self, config=None):
        # Initialize tokenizer with GPT-2 encoding
        self.enc, self.dec = load_encoder_decoder()
        
        # Special tokens for educational content structure
        self.special_tokens = {
            "content_start": "<|content|>",
            "content_end": "</|content|>",
            "metadata_start": "<|metadata|>",
            "metadata_end": "</|metadata|>",
            "url_start": "<|url|>",
            "url_end": "</|url|>",
            "date_start": "<|date|>",
            "date_end": "</|date|>"
        }
        
        # Ensure data directory exists
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory: {self.data_dir}")
        
        # Configuration for processing
        self.max_length = 1024  # DeepSeek context window
        self.min_length = 100   # Minimum content length for educational content
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for educational content"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Remove any inappropriate content markers
        inappropriate_phrases = ['adult content', 'mature', 'explicit', 'nsfw']
        for phrase in inappropriate_phrases:
            if phrase.lower() in text.lower():
                return ""
        
        # Ensure the text is educational and appropriate
        if len(text) < self.min_length:
            return ""
            
        return text
        
    def extract_content_elements(self, example: Dict) -> Dict:
        """Extract content elements for better structure"""
        # Main content
        content = self.preprocess_text(example.get('text', ''))
        
        # Metadata
        url = example.get('url', '')
        date = example.get('date', '')
        language = example.get('language', '')
        score = example.get('score', 0.0)
        
        # Only process if we have valid content
        if not content:
            return {'content': '', 'url': '', 'date': '', 'language': '', 'score': 0.0}
        
        return {
            'content': content,
            'url': url,
            'date': date,
            'language': language,
            'score': score
        }
        
    def process(self, example: Dict) -> Dict:
        """Process a single example for DeepSeek model"""
        # Extract content elements
        elements = self.extract_content_elements(example)
        
        # Skip if no valid content
        if not elements['content']:
            return {'ids': [], 'len': 0}
        
        # Create structured text with special tokens
        full_text = f"{self.special_tokens['content_start']} {elements['content']} {self.special_tokens['content_end']}"
        
        # Add metadata if available
        if elements['url']:
            full_text += f" {self.special_tokens['url_start']} {elements['url']} {self.special_tokens['url_end']}"
        
        if elements['date']:
            full_text += f" {self.special_tokens['date_start']} {elements['date']} {self.special_tokens['date_end']}"
        
        # Add language and score information
        if elements['language'] and elements['score'] > 0.5:  # Only include if language is detected with confidence
            full_text += f" {self.special_tokens['metadata_start']} Language: {elements['language']}, Quality: {elements['score']:.2f} {self.special_tokens['metadata_end']}"
        
        # Tokenize with error handling
        try:
            ids = self.enc.encode_ordinary(full_text)
            
            # Ensure the sequence isn't too long
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # Skip if too short
            if len(ids) < 50:  # Higher minimum for educational content
                return {'ids': [], 'len': 0}
                
            out = {'ids': ids, 'len': len(ids)}
            return out
            
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return {'ids': [], 'len': 0}
        
    def prepare_dataset(self) -> Dict:
        """Prepare the FineWeb Educational dataset for DeepSeek training"""
        # Clear any existing dataset cache first
        print("Clearing existing dataset cache...")
        try:
            from datasets import clear_cache
            clear_cache()
            print("Dataset cache cleared successfully!")
            
            # Also try to clear HuggingFace cache directory
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                print(f"Clearing HuggingFace cache at: {cache_dir}")
                # Note: This is a safety measure - actual clearing happens via clear_cache()
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
        
        # Strategy: Load multiple parquet files for ~10GB download
        print("Loading multiple parquet files from FineWeb Educational dataset (~10GB)...")
        try:
            # Load multiple parquet files to reach ~10GB
            # Each file is ~2.3GB, so we need about 4-5 files
            data_files = [
                "data/CC-MAIN-2024-42/000_00000.parquet",
                "data/CC-MAIN-2024-42/000_00001.parquet", 
                "data/CC-MAIN-2024-42/000_00002.parquet",
                "data/CC-MAIN-2024-42/000_00003.parquet",
                "data/CC-MAIN-2024-42/000_00004.parquet"
            ]
            
            print(f"Loading {len(data_files)} parquet files...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", 
                            data_files=data_files,
                            split='train')
            print(f"Successfully loaded {len(ds)} examples from {len(data_files)} parquet files")
            
        except Exception as e:
            print(f"Failed to load multiple files, trying single file fallback: {e}")
            # Fallback: Single file approach
            ds = load_dataset("HuggingFaceFW/fineweb-edu", 
                            data_files="data/CC-MAIN-2024-42/000_00000.parquet",
                            split='train')
            print(f"Successfully loaded {len(ds)} examples from single parquet file (fallback)")
        
        train_bin_path = os.path.join(self.data_dir, "fineweb_train.bin")
        val_bin_path = os.path.join(self.data_dir, "fineweb_validation.bin")
        
        print(f"Checking for existing processed files...")
        
        # Check if both files exist
        if (os.path.exists(train_bin_path) and 
            os.path.exists(val_bin_path)):
            
            print("Found existing processed files!")
            print(f"Train file: {os.path.getsize(train_bin_path) / (1024*1024):.2f} MB")
            print(f"Validation file: {os.path.getsize(val_bin_path) / (1024*1024):.2f} MB")
            
            return {
                "train": train_bin_path,
                "validation": val_bin_path
            }
        
        print("Processing dataset...")
        
        # Filter out examples that are too short or too long
        def filter_by_length(example):
            text_length = len(example.get('text', ''))
            # More selective filtering for larger dataset
            return (self.min_length <= text_length <= 3000 and  # Reasonable length for educational content
                    example.get('score', 0) > 0.6)  # Higher quality threshold
        
        ds = ds.filter(filter_by_length)
        print(f"After filtering: {len(ds)} examples")
        
        # Split the dataset into train and validation sets (80/20)
        train_val = ds.train_test_split(test_size=0.2, seed=42)
        
        # Create a new dataset dictionary with both splits
        ds = {
            "train": train_val["train"],
            "validation": train_val["test"]
        }
        
        print(f"Dataset split sizes:")
        print(f"Training set: {len(ds['train'])} examples (80%)")
        print(f"Validation set: {len(ds['validation'])} examples (20%)")
        
        # Process each split
        for split_name, split_data in ds.items():
            print(f"\nProcessing {split_name} split...")
            
            # Process the data
            tokenized = split_data.map(
                self.process,
                remove_columns=['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'],
                desc=f"tokenizing {split_name} split",
                num_proc=8,
            )
            
            # Filter out empty sequences
            tokenized = tokenized.filter(lambda x: x['len'] > 0)
            print(f"After processing: {len(tokenized)} valid examples")
            
            # Convert to numpy arrays
            all_ids = []
            for example in tokenized:
                all_ids.extend(example['ids'])
            
            # Convert to numpy array
            arr = np.array(all_ids, dtype=np.uint16)
            
            # Save to binary file
            filename = os.path.join(self.data_dir, f"fineweb_{split_name}.bin")
            arr.tofile(filename)
            
            print(f"Saved {split_name} split to {filename}")
            print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            print(f"Number of tokens: {len(arr):,}")
        
        return {
            "train": train_bin_path,
            "validation": val_bin_path
        }
    
    def load_binary_data(self, filepath: str) -> torch.Tensor:
        """Load binary data file as tensor"""
        try:
            data = np.memmap(filepath, dtype=np.uint16, mode='r')
            return torch.from_numpy(data.copy())
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            raise
    
    def get_batch(self, data: torch.Tensor, batch_size: int, block_size: int) -> tuple:
        """Get a batch of data for training"""
        # Generate random indices
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Get input sequences
        x = torch.stack([data[i:i+block_size].long() for i in ix])
        # Get target sequences (shifted by 1)
        y = torch.stack([data[i+1:i+1+block_size].long() for i in ix])
        
        return x, y
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        try:
            return self.enc.decode(token_ids)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            return ""
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        try:
            return self.enc.encode_ordinary(text)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return []


def main():
    """Main function to process the FineWeb Educational dataset"""
    print("FineWeb Educational Data Processor")
    print("=" * 50)
    
    processor = FineWebDataProcessor()
    processor.prepare_dataset()
    
    print("\nData processing completed successfully!")
    print("Files created:")
    print("- src/data/fineweb_train.bin")
    print("- src/data/fineweb_validation.bin")


if __name__ == "__main__":
    main()
