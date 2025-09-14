# Milk Yield Prediction Backend

## Structure

- `src/` — All source code modules
- `data/` — Dataset(s)
- `requirements.txt` — Python dependencies

## How to Run

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the training pipeline:
   ```sh
   python src/train.py
   ```

## Modules
- `data_loader.py`: Loads the dataset
- `feature_engineering.py`: Feature engineering functions
- `outlier.py`: Outlier capping
- `preprocessing.py`: Preprocessing pipelines
- `models.py`: Model definitions
- `evaluation.py`: Evaluation metrics
- `train.py`: Main training and evaluation script
