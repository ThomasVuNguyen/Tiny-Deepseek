# Training Log

## 2025-08-27
- **16-hour training run**: `python src/run_training.py --max-iters 400000 --eval-interval 2500 --batch-size 18`
- Expected: ~400k iterations, ~42 checkpoints, ~65GB storage
- Training time: ~16 hours (at 0.145s/iter)
- **Added dual checkpoint saving**: Full checkpoints (~1.5GB) + Model-only (~86MB)
- **Updated generate.py**: Now defaults to model-only checkpoints for faster loading

## 2025-08-29
- With all the training done, we have seen that 200k-th iteration is when the model training & evaluation losses are at the lowest. From that point on, model loss does not decrease further.
- All training & model checkpoints are available in: https://huggingface.co/ThomasTheMaker/Tiny-Deepseek-TinyStories
- Next step: Update model hyper-parameter to fit Raspberry Pi Zero 2W & train on the training mix for Olmo2 from Ai2

## 2025-08-30
- **FineWeb Educational Dataset Integration**: Added support for HuggingFaceFW/fineweb-edu dataset
- **Dataset Size Challenge**: FineWeb is ~4TB total, need sampling strategies for manageable training
- **Sampling Strategies Implemented**:
  - **Strategy 1**: Hugging Face built-in sampling (`split='train[:1000]'`)
  - **Strategy 2**: Random sampling with shuffle and select
  - **Strategy 3**: Quality filtering (score > 0.7, length 200-2000 chars)
  - **Strategy 4**: Single parquet file loading (one of 2410 files)
  - **Strategy 5**: Progressive loading (100 → 1000 → 10000 examples)
- **Recommended Approach**: Start with first 5000 examples for testing, scale up as needed
- **Implementation Decision**: Using Strategy 4 (single parquet file) with 1,000 examples 
- **Problem Discovered**: Even `split='train[:10000]'` downloads ALL parquet files (~4TB) first
- **Solution**: Load specific parquet file or use streaming to minimize download
- **Expected Data Size**: ~2.3 GB download (1 file), ~100 MB processed data, ~500 MB total storage
- **Cleanup Issue**: Initial download consumed 77GB before interruption
- **Cleanup Solution**: Created `cleanup_fineweb.sh` script to remove all cached FineWeb data
- **Success**: Fixed processor now works with 1.41GB download, 681 examples, 0.76MB processed data
- **Performance**: Complete processing in under 1 minute (96% reduction in storage)
- **Scale-Up**: Upgraded to 10GB download (5 parquet files) with quality filtering (score > 0.6)
- **Expected**: ~50K-100K examples after filtering, ~50-100MB processed data
- **Split Fix**: Changed from inefficient 80/10/10 to proper 80/20 train/validation split
- **Final Results**: 758K training examples (647MB), 190K validation examples (162MB), 424M tokens total
- **Training Logging**: Added real-time CSV logging for train/val loss tracking during training
- **Files Created**:
  - `src/data/fineweb_processor.py` - FineWeb-specific data processor
  - `src/run_fineweb_training.py` - FineWeb training script
  - `src/process_fineweb_sample.py` - Sample processor for testing

## 2025-08-31
- **FineWeb 150K Iteration Training Run Completed**: `python src/run_fineweb_training.py --batch-size 20 --learning-rate 8e-4 --eval-interval 10000 --eval-iters 100 --max-iters 150000`
- **Training Performance Analysis**:
  - **Major Learning Phase**: 0 → 20,000 iterations (loss dropped from ~11 to ~4.9)
  - **Refinement Phase**: 20,000 → 100,000 iterations (slow improvement from ~4.9 to ~4.8)
  - **Final Optimization**: 100,000 → 140,000 iterations (loss improved from ~4.8 to ~4.76)
  - **Best Performance**: Validation loss 4.762640 at iteration 140,000
  - **Diminishing Returns**: 120,000 iterations for minimal improvement (4.9 → 4.76)
- **Key Insights**:
  - **Dataset Too Small**: 5 parquet files (~10GB) caused overfitting after 20k iterations
  - **Wasted Compute**: 120k iterations for only 0.14 loss improvement
  - **Training Efficiency**: Model plateaued early due to insufficient data variety
- **Optimization Strategy Identified**: Larger dataset (20-30 parquet files) with fewer iterations (40k) should achieve same performance in ~2-3 hours instead of 8+ hours
- **Files Ready for Hugging Face Upload**:
  - `fineweb_final_model_only.pt` (496MB) - Final trained model weights
  - `fineweb_final_model.pt` (1.5GB) - Complete checkpoint with optimizer state
  - `best_model_only.pt` (496MB) - Best performing model (iteration 140,000)
  - `best_model.pt` (1.5GB) - Best complete checkpoint
  - `training_log.csv` (1.4KB) - Complete training history and metrics
  - `deepseek_training_metrics.png` (367KB) - Training visualization
