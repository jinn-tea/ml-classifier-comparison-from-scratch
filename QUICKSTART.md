# Quick Start Guide

Follow these steps to run the project:

## Step 1: Install Python Dependencies

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

Or if you're using conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Step 2: Get the Dataset

You have two options:

### Option A: Download from Kaggle (Recommended)
1. Go to: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Download the `heart.csv` file
3. Place it in the project root directory (same folder as the notebooks)

### Option B: Automatic Download (Fallback)
The notebook will try to automatically download the dataset from a public URL if the file is not found.

## Step 3: Run the Notebooks

### Option 1: Using Jupyter Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Run the notebooks in order:**
   
   **First:** Open and run `01_EDA_and_Preprocessing.ipynb`
   - This will load the data, perform EDA, preprocess it, and save to `preprocessed_data.pkl`
   - Run all cells (Cell → Run All)
   
   **Then:** Open and run `02_Main_Analysis.ipynb`
   - This will train all models, tune hyperparameters, and generate visualizations
   - Run all cells (Cell → Run All)

### Option 2: Using JupyterLab

1. **Start JupyterLab:**
   ```bash
   jupyter lab
   ```

2. Follow the same steps as above - open and run the notebooks in order.

### Option 3: Using VS Code

1. Open the project folder in VS Code
2. Open the `.ipynb` files
3. Run cells sequentially or "Run All"

## Step 4: Check the Results

After running the notebooks, you'll find:

### Generated Files:
- `preprocessed_data.pkl` - Preprocessed data (from notebook 01)
- `results_summary.csv` - Summary of model performance
- `knn_learning_curve.png` - k-NN learning curve
- `rf_learning_curve.png` - Random Forest learning curve
- `rf_feature_importance.png` - Feature importance plot
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves for all models
- `model_comparison.png` - Performance comparison charts

### What You'll See:
- Model training progress
- Hyperparameter tuning results
- Evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Visualizations inline in the notebooks
- Summary tables

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError for heart.csv
**Solution:** 
- Download the dataset from Kaggle and place it in the project root
- Or let the notebook download it automatically

### Issue: FileNotFoundError for preprocessed_data.pkl
**Solution:** Run `01_EDA_and_Preprocessing.ipynb` first before running `02_Main_Analysis.ipynb`

### Issue: Import errors
**Solution:** Make sure you're running from the project root directory where all the `.py` files are located.

## Running as a Python Script (Alternative)

If you prefer running as a script instead of notebooks:

1. First run the preprocessing notebook (or create a preprocessing script)
2. Then run:
   ```bash
   python main_analysis.py
   ```

## Expected Runtime

- **Notebook 01 (EDA & Preprocessing):** ~1-2 minutes
- **Notebook 02 (Main Analysis):** ~5-10 minutes (depending on dataset size and hyperparameter tuning)

## Next Steps

After running the notebooks:
1. Review the generated visualizations
2. Check the `results_summary.csv` for performance metrics
3. Fill in the `REPORT_TEMPLATE.md` with your findings
4. Use the generated plots in your report

---

**Need Help?** Check the main `README.md` for more detailed information.
