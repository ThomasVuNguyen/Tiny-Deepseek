# Training Log

## 2025-08-27
- **8-hour training run**: `python src/run_training.py --max-iters 200000 --eval-interval 2000 --batch-size 12`
- Expected: ~200k iterations, ~22 checkpoints, ~33GB storage
- **Added dual checkpoint saving**: Full checkpoints (~1.5GB) + Model-only (~86MB)
- **Updated generate.py**: Now defaults to model-only checkpoints for faster loading
