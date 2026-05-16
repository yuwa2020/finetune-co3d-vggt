# Fine-tuning VGGT on Co3D

This directory contains the config and launch script used to fine-tune VGGT on the Co3D dataset as part of an independent study project.

## What we did

- Started from the official `facebook/VGGT-1B` pretrained checkpoint
- Fine-tuned only the **camera head** and **depth head** (~248.8M params); the **aggregator is frozen** (~909M params)
- Trained on Co3D (10K train / 1K val sequences) for 50 epochs on 4 × NVIDIA L40S GPUs (46 GB each)

## Results (fast eval, 10 sequences/category on Co3D test set)

| Model | AUC@30 | AUC@15 | AUC@5 | AUC@3 |
|---|---|---|---|---|
| Original VGGT-1B (paper target) | 89.98 | 83.89 | 67.45 | 56.65 |
| Original VGGT-1B (our eval) | 89.76 | 83.82 | 67.99 | 57.43 |
| **Fine-tuned on Co3D (ours)** | **89.36** | **83.07** | **66.27** | **55.14** |

Our fine-tuned model is within **0.4% AUC@30** of the original despite using only Co3D and a smaller batch size.

## Key hyperparameters

| Hyperparameter | Value |
|---|---|
| Image resolution | 518 × 518 |
| Views per sequence | sampled from [2, 8] |
| Max images per GPU | 4 |
| Optimizer | AdamW |
| Learning rate | 5e-5 (peak) |
| LR schedule | Linear warmup 5% → Cosine decay 95% |
| Weight decay | 0.05 |
| Epochs | 50 |
| Mixed precision | bfloat16 |
| Frozen modules | aggregator |

## How to run

### 1. Download the pretrained checkpoint

```bash
# from HuggingFace
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-1B', 'model.pt', local_dir='./checkpoints')"
```

### 2. Download Co3D and annotations

- Co3D dataset: https://github.com/facebookresearch/co3d
- Annotations (OpenCV convention, required for training): https://huggingface.co/datasets/JianyuanWang/co3d_anno

### 3. Run fine-tuning

```bash
VGGT_REPO_DIR=/path/to/this/repo \
CO3D_DIR=/path/to/co3d \
CO3D_ANNOTATION_DIR=/path/to/co3d_anno \
PRETRAIN_CKPT=/path/to/checkpoints/model.pt \
PYTHON_BIN=$(which python) \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash finetune/run_training.sh
```

### 4. Run evaluation

Follow the instructions in the [`evaluation` branch](https://github.com/facebookresearch/vggt/tree/evaluation) of the original repo.

When loading the fine-tuned checkpoint, use `strict=False` since we disabled `point_head` and `track_head`:

```python
model.load_state_dict(torch.load("path/to/checkpoint.pt"), strict=False)
```
