Best current model: PIKA-like v1
Inputs: pill crop image + prescription image
Validation Macro F1: 0.6054
Validation Accuracy: 0.5959
Notes: currently best among all tested variants
- train_pika_v1_1_improved.py: experimental attempt with WeightedRandomSampler + FocalLoss, not better than current best model.
## so sánh vs paper
Paper PIKA best: 0.8571
Baseline image-only: 0.4801
PIKA-like v1: 0.6054
PIKA v2: 0.5957
PIKA v3: 0.5882
