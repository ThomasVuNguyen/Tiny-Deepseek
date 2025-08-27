# Training Log

## 2025-08-27
- **16-hour training run**: `python src/run_training.py --max-iters 400000 --eval-interval 2500 --batch-size 12`
- Expected: ~400k iterations, ~42 checkpoints, ~65GB storage
- Training time: ~16 hours (at 0.145s/iter)
- **Added dual checkpoint saving**: Full checkpoints (~1.5GB) + Model-only (~86MB)
- **Updated generate.py**: Now defaults to model-only checkpoints for faster loading
