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
