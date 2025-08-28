# FineWeb Educational Dataset - Construction Guide

This document explains how the FineWeb Educational dataset was constructed, sampled, and processed for training DeepSeek language models.

## Dataset Source

**Original Dataset**: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

**Full Dataset Size**: ~4TB (2,410 parquet files)
**Content Type**: High-quality educational web content from Common Crawl

## Sampling Strategy

### Why Sampling?
The full FineWeb dataset is massive (~4TB) and would take days to download and process. We implemented **Strategy 4: Single Parquet File Loading** for efficient processing.

### Sampling Method
- **Files Selected**: First 5 parquet files from the dataset
- **Download Size**: ~10GB (vs 4TB full dataset)
- **Percentage**: ~0.25% of total dataset
- **Rationale**: Sequential loading is most efficient with HuggingFace datasets

### Files Used
```
data/CC-MAIN-2024-42/
├── 000_00000.parquet (~2.3GB)
├── 000_00001.parquet (~2.3GB)  
├── 000_00002.parquet (~2.3GB)
├── 000_00003.parquet (~2.3GB)
└── 000_00004.parquet (~2.3GB)
```

## Data Processing Pipeline

### Step 1: Raw Data Loading
```python
# Load 5 parquet files (~10GB total)
data_files = [
    "data/CC-MAIN-2024-42/000_00000.parquet",
    "data/CC-MAIN-2024-42/000_00001.parquet", 
    "data/CC-MAIN-2024-42/000_00002.parquet",
    "data/CC-MAIN-2024-42/000_00003.parquet",
    "data/CC-MAIN-2024-42/000_00004.parquet"
]

ds = load_dataset("HuggingFaceFW/fineweb-edu", 
                  data_files=data_files, 
                  split='train')
```

**Result**: 2,294,208 raw examples loaded

### Step 2: Quality Filtering
```python
def filter_by_length(example):
    text_length = len(example.get('text', ''))
    return (100 <= text_length <= 3000 and  # Length filter
            example.get('score', 0) > 0.6)   # Quality filter

ds = ds.filter(filter_by_length)
```

**Filtering Criteria**:
- **Length**: 100-3000 characters (educational content range)
- **Quality Score**: > 0.6 (high-quality content only)
- **Language**: English content (from language detection)

**Result**: 964,864 high-quality examples (42% of raw data)

### Step 3: Dataset Splitting
```python
# 80/20 train/validation split
train_val = ds.train_test_split(test_size=0.2, seed=42)

ds = {
    "train": train_val["train"],      # 80% for training
    "validation": train_val["test"]   # 20% for validation
}
```

**Final Split**:
- **Training**: 771,891 examples (80%)
- **Validation**: 192,973 examples (20%)

### Step 4: Tokenization & Binary Conversion
```python
# Process each split
for split_name, split_data in ds.items():
    # Tokenize with GPT-2 tokenizer
    tokenized = split_data.map(self.process, ...)
    
    # Convert to binary format
    all_ids = []
    for example in tokenized:
        all_ids.extend(example['ids'])
    
    # Save as binary file
    arr = np.array(all_ids, dtype=np.uint16)
    filename = f"fineweb_{split_name}.bin"
    arr.tofile(filename)
```

## Final Dataset Statistics

### File Sizes
- **`fineweb_train.bin`**: 646.95 MB (339,186,828 tokens)
- **`fineweb_validation.bin`**: 161.80 MB (84,832,287 tokens)
- **Total Processed**: 808.75 MB (424,019,115 tokens)

### Content Distribution
- **Training Examples**: 758,265 (after tokenization filtering)
- **Validation Examples**: 189,518 (after tokenization filtering)
- **Total Examples**: 947,783

### Quality Metrics
- **Original Raw Data**: 2,294,208 examples
- **After Quality Filtering**: 964,864 examples (42% retention)
- **After Tokenization**: 947,783 examples (98% of filtered)

## Dataset Structure

### Input Format
Each example contains:
```json
{
  "text": "Main educational content...",
  "url": "https://example.com/article",
  "date": "2024-01-15",
  "language": "en",
  "score": 0.95
}
```

### Processing Output
```python
# Special tokens structure
full_text = (
    f"{self.special_tokens['content_start']} {content} {self.special_tokens['content_end']}"
    f" {self.special_tokens['url_start']} {url} {self.special_tokens['url_end']}"
    f" {self.special_tokens['date_start']} {date} {self.special_tokens['date_end']}"
)
```

## Usage in Training

### Training Script
```bash
python src/run_fineweb_training.py
```

### Data Loading
```python
from src.data.fineweb_processor import FineWebDataProcessor

processor = FineWebDataProcessor()
train_data = processor.load_binary_data('fineweb_train.bin')
val_data = processor.load_binary_data('fineweb_validation.bin')
```

## Reproducibility

### Random Seeds
- **Dataset Split**: `seed=42` (reproducible train/val split)
- **Processing**: Deterministic tokenization and filtering

### File Selection
- **Parquet Files**: First 5 files in chronological order
- **Sampling**: Sequential loading (not random sampling)

## Limitations & Considerations

### Sampling Bias
- **Chronological**: Only includes content from specific time periods
- **Geographic**: May be biased toward certain regions/languages
- **Content Type**: Web content may have different characteristics than curated datasets

### Quality Trade-offs
- **Filtering**: Aggressive filtering removes 58% of raw data
- **Length**: 100-3000 character limit may exclude some valuable content
- **Score Threshold**: 0.6 threshold is somewhat arbitrary

## Future Improvements

### Alternative Sampling Strategies
1. **Random Sampling**: Load random parquet files across time periods
2. **Stratified Sampling**: Ensure representation across different content types
3. **Progressive Loading**: Start small, expand based on training results

### Enhanced Filtering
1. **Content Classification**: Filter by educational topic/domain
2. **Language Detection**: Better multilingual support
3. **Quality Metrics**: More sophisticated quality scoring

## Technical Details

### Tokenizer
- **Type**: GPT-2 tokenizer (50,257 vocabulary)
- **Special Tokens**: Custom tokens for content structure
- **Context Window**: 1024 tokens (DeepSeek architecture)

### Processing Pipeline
- **Parallel Processing**: 8 processes for tokenization
- **Memory Management**: Efficient streaming for large datasets
- **Error Handling**: Graceful fallback for malformed examples

### Storage Format
- **Binary Format**: `.bin` files for efficient loading
- **Data Type**: `uint16` (65,536 token limit, sufficient for GPT-2 vocab)
- **Compression**: No compression (trade-off between size and loading speed)

## Dataset Citation

If you use this processed dataset, please cite:

```bibtex
@dataset{fineweb_edu_processed,
  title={FineWeb Educational Dataset - 10GB Sampled (0.25% of Full Dataset)},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/your-username/fineweb-edu-10gb-5parquet-processed},
  note={Processed subset of HuggingFaceFW/fineweb-edu for DeepSeek training}
}
```

## Contact & Support

For questions about this dataset construction:
- **Repository**: [Tiny-Deepseek](https://github.com/your-username/Tiny-Deepseek)
- **Issues**: GitHub Issues for technical problems
- **Discussions**: GitHub Discussions for general questions

---

*This dataset was constructed as part of the Tiny-Deepseek project for training efficient language models on educational content.*
