# Google Colab Training Guide

This guide explains how to train your NCF Movie Recommender model using Google Colab's free GPU.

## Why Use Colab?

| Colab Free Tier | Your WSL2 (RTX 3050 4GB) |
|-----------------|-------------------------|
| T4 GPU with **16GB VRAM** | RTX 3050 with **4GB VRAM** |
| ~12GB RAM | ~8GB RAM (limited by WSL2) |
| No risk to your PC | WSL2 can crash |
| Save to Google Drive | Local files |

## Quick Start (5 minutes)

### Step 1: Open the Notebook

1. Go to https://colab.research.google.com/
2. Click "File" → "Open notebook"
3. Click "GitHub" tab
4. Enter: `albertabayor/NCF-Movie-Recommender`
5. Open: `notebooks/colab_from_github.ipynb`

### Step 2: Update the Username

In the first code cell, change:
```python
GITHUB_USERNAME = "YOUR_USERNAME"  # CHANGE THIS!
```

To:
```python
GITHUB_USERNAME = "albertabayor"
```

### Step 3: Upload Datasets

**Option A: Use the upload widget (easier)**
- Run the upload cell in the notebook
- Upload these files when prompted:
  - `ratings.csv`
  - `movies_metadata.csv`
  - `links.csv`

**Option B: Manual upload**
1. Click the folder icon 📁 on the left sidebar
2. Navigate to `NCF-Movie-Recommender/datasets/`
3. Click the upload button
4. Upload the 3 CSV files

### Step 4: Run All Cells

Click "Runtime" → "Run all" and grab a coffee!

Training will take ~2-3 hours per model on the free GPU.

## Expected Results

After training completes, you'll see:
```
Training complete in 120.5 minutes
Best hr@10: 0.0856
```

## Download Trained Models

1. Run the "Download" cell
2. The `.pt` file will download to your computer
3. Copy it to your local project: `experiments/trained_models/`

## Workflow Summary

```
┌─────────────┐      Clone      ┌──────────────┐
│   GitHub    │ ──────────────> │   Colab      │
│   (code)    │                  │   (GPU)      │
└─────────────┘                  └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Trained      │
                                    │ Models (.pt) │
                                    └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Your PC      │
                                    │ (inference)  │
                                    └──────────────┘
```

## Pro Tips

1. **Save checkpoints**: Models are saved during training, so if Colab disconnects, you don't lose everything
2. **Monitor with TensorBoard**: The notebook includes TensorBoard to watch training live
3. **Copy processed data**: After preprocessing, you can download `data/` folder to use locally
4. **Upgrade to Colab Pro** ($10/mo): Get V100 or A100 GPUs for faster training

## Troubleshooting

### "Runtime disconnected"
- Colab free tier disconnects after ~90 min of inactivity
- Your trained models are saved, so just reconnect and continue

### "Out of memory"
- Reduce `batch_size` in the training cell (try 256 instead of 512)

### "GitHub clone failed"
- Make sure your repo is public, or set up GitHub credentials in Colab

## Next: Local Inference

Once you have trained models:

```bash
# On your local machine
cd ~/projects/NCF-Movie-Recommender

# Run inference
python -m src.inference \
    --model experiments/trained_models/NeuMFPlus_best.pt \
    --user-id 123 \
    --top-k 10
```
