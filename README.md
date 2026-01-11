# NCF Movie Recommender

Neural Collaborative Filtering system with content-based features (genre + synopsis) and gated feature fusion for movie recommendation.

## Project Overview

This project implements:
- **Baseline models**: GMF, MLP, NeuMF
- **Proposed model**: NeuMF + Genre + Synopsis + Gated Fusion
- **Ablation study** to measure each component's contribution
- **Cold-start evaluation** to validate content features

## Dataset

**Required files** (place in `datasets/` folder):

| File | Size | Source |
|------|------|--------|
| `ratings.csv` | ~709MB | [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) |
| `movies_metadata.csv` | ~34MB | [Movies Metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) |
| `links.csv` | ~989KB | MovieLens 25M |
| `keywords.csv` | ~6MB | Movies Metadata |

**Quick setup** (from project root):
```bash
# Create datasets folder
mkdir -p datasets

# Download MovieLens 25M (contains ratings.csv + links.csv)
# Extract and place in datasets/
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
mv ml-25m/ratings.csv datasets/
mv ml-25m/links.csv datasets/

# Download movies_metadata.csv from Kaggle
# Requires kaggle API or manual download
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NCF-Movie-Recommender/
├── datasets/          # Raw data (not in git)
├── data/              # Processed data (not in git)
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Python modules
│   ├── models/        # Model implementations
│   ├── train.py       # Training scripts
│   └── evaluate.py    # Evaluation functions
├── experiments/       # Trained models and logs (not in git)
├── PROJECT_PLAN.md    # Detailed project plan
└── requirements.txt   # Python dependencies
```

## Usage

### 1. Data Preprocessing
```bash
python -m src.preprocessing
```

### 2. Train Baseline Model
```bash
python -m src.train --model gmf --epochs 30
```

### 3. Train Full Model
```bash
python -m src.train --model neumf_plus --epochs 30
```

### 4. Run Ablation Study
```bash
python -m src.train --ablation
```

### 5. Generate Recommendations
```bash
python -m src.inference --user_id 123 --k 10
```

## Results

See `PROJECT_PLAN.md` for detailed ablation study plan and expected outcomes.

## Key Features

- **Dynamic negative sampling** during training (not pre-sampled)
- **Cold-start evaluation** for users/items with few interactions
- **Sentence-BERT** for synopsis encoding (pre-trained, better than TextCNN)
- **Gated fusion** to adaptively balance CF and content signals
- **TensorBoard logging** for training visualization

## References

1. He, X., et al. (2017). "Neural Collaborative Filtering." WWW
2. [Sentence-Transformers](https://www.sbert.net/)
3. [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

## License

MIT
