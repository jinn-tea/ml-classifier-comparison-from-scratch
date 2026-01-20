# Comparative Study of Classification Algorithms

This project implements three classification algorithms from scratch and compares their performance on the Heart Disease UCI dataset.

## Algorithms Implemented

1. **k-Nearest Neighbors (k-NN)**
   - Euclidean and Manhattan distance metrics
   - Weighted voting based on distance
   - Configurable k values (1, 3, 5, 7, 10)

2. **Random Forest**
   - Decision trees with Gini impurity and Entropy split criteria
   - Bootstrap sampling (bagging)
   - Majority voting aggregation
   - At least 5 trees

3. **Naive Bayes**
   - Gaussian Naive Bayes for continuous features
   - Frequency-based probability estimation for categorical features
   - Laplace smoothing

## Project Structure

```
ml-classifier-comparison-from-scratch/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── REPORT_TEMPLATE.md                 # Report template
│
├── knn_classifier.py                  # k-NN implementation
├── decision_tree.py                   # Decision Tree implementation
├── random_forest.py                   # Random Forest implementation
├── naive_bayes.py                     # Naive Bayes implementation
├── evaluation_utils.py                # Evaluation metrics and visualization utilities
├── main_analysis.py                   # Main analysis script
│
├── 01_EDA_and_Preprocessing.ipynb     # EDA and preprocessing notebook
└── 02_Main_Analysis.ipynb             # Main analysis notebook
```

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Dataset

Download the Heart Disease UCI dataset from Kaggle:
- https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Place the dataset file (`heart.csv`) in the project root directory.

Alternatively, the notebook will attempt to download it automatically from a public URL.

### Step 2: Run EDA and Preprocessing

Open and run `01_EDA_and_Preprocessing.ipynb`:
- Loads and explores the dataset
- Performs exploratory data analysis
- Handles missing values and encodes categorical variables
- Standardizes features
- Splits data into training (70%), validation (15%), and test (15%) sets
- Saves preprocessed data to `preprocessed_data.pkl`

### Step 3: Run Main Analysis

Open and run `02_Main_Analysis.ipynb`:
- Implements and trains all three classifiers
- Performs hyperparameter tuning
- Evaluates models on test set
- Generates visualizations (confusion matrices, ROC curves, learning curves, feature importance)
- Creates comparison plots and summary tables

Alternatively, you can run the analysis as a Python script:
```bash
python main_analysis.py
```

## Output Files

After running the notebooks, the following files will be generated:

- `preprocessed_data.pkl` - Preprocessed data for analysis
- `results_summary.csv` - Summary of model performance metrics
- `knn_learning_curve.png` - k-NN learning curve
- `rf_learning_curve.png` - Random Forest learning curve
- `rf_feature_importance.png` - Random Forest feature importance plot
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves for all models
- `model_comparison.png` - Performance comparison bar charts

## Key Features

### Implementations from Scratch

All algorithms are implemented without using sklearn's core implementations:
- Only used sklearn for utilities like `train_test_split`, `StandardScaler`, and evaluation metrics
- Full understanding of algorithm mechanics
- Educational value

### Comprehensive Evaluation

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Learning curves
- Feature importance (Random Forest)

### Hyperparameter Tuning

- k-NN: k values, distance metrics, voting schemes
- Random Forest: max_depth, split criteria (Gini/Entropy)
- Naive Bayes: Automatic feature type detection

## Report

A detailed report template is provided in `REPORT_TEMPLATE.md`. Fill it in with your results and analysis.

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

## Notes

- All implementations are educational and may not be optimized for production use
- The code is well-documented for learning purposes
- Feel free to extend or modify the implementations

## License

[Add your license here]

## Author

[Your Name]

---

For questions or issues, please refer to the assignment instructions or contact your instructor.
