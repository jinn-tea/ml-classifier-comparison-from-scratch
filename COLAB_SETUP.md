# Running on Google Colab

This guide explains how to run the project on Google Colab.

## Method 1: Upload Project Files to Colab (Recommended)

### Step 1: Prepare Your Files

1. Create a zip file of your project (excluding the dataset if you want to download it separately):
   - Include all `.py` files
   - Include both `.ipynb` notebook files
   - Exclude `preprocessed_data.pkl` (will be generated)
   - Exclude output images (will be regenerated)

### Step 2: Upload to Colab

1. **Go to Google Colab:** https://colab.research.google.com/

2. **Upload the notebooks:**
   - Click `File` → `Upload notebook`
   - Upload `01_EDA_and_Preprocessing.ipynb`
   - Upload `02_Main_Analysis.ipynb`

3. **Upload Python files:**
   - Create a new code cell in the first notebook
   - Use the file upload feature, OR
   - Create the `.py` files directly in Colab (see Method 2 below)

### Step 3: Upload Python Modules

In the first notebook, add a code cell at the beginning to upload your Python files:

```python
# Upload required Python modules
from google.colab import files
import os

# Create upload cell (run this once)
uploaded = files.upload()

# Or create the files directly using %%writefile magic command
```

### Step 4: Install Dependencies

Add a cell at the beginning of `01_EDA_and_Preprocessing.ipynb`:

```python
!pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Step 5: Run the Notebooks

1. Run all cells in `01_EDA_and_Preprocessing.ipynb`
2. Download the generated `preprocessed_data.pkl` file
3. Upload it to `02_Main_Analysis.ipynb` before running it

---

## Method 2: Create Files Directly in Colab (Easier)

### Step 1: Create Python Module Files

In a new Colab notebook, create each Python file using `%%writefile`:

#### Create `knn_classifier.py`:
```python
%%writefile knn_classifier.py
[Paste the entire content of knn_classifier.py here]
```

#### Create `decision_tree.py`:
```python
%%writefile decision_tree.py
[Paste the entire content of decision_tree.py here]
```

#### Create `random_forest.py`:
```python
%%writefile random_forest.py
[Paste the entire content of random_forest.py here]
```

#### Create `naive_bayes.py`:
```python
%%writefile naive_bayes.py
[Paste the entire content of naive_bayes.py here]
```

#### Create `evaluation_utils.py`:
```python
%%writefile evaluation_utils.py
[Paste the entire content of evaluation_utils.py here]
```

### Step 2: Install Dependencies

```python
!pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Step 3: Upload/Create Notebooks

Either:
- Upload your existing notebooks, OR
- Create new notebooks and copy-paste the code cells

### Step 4: Handle Dataset

The dataset will be automatically downloaded by the notebook, or you can upload it:

```python
from google.colab import files
uploaded = files.upload()
# Then use: df = pd.read_csv('heart.csv')
```

---

## Method 3: Using Google Drive (Best for Large Projects)

### Step 1: Upload Project to Google Drive

1. Upload your entire project folder to Google Drive
2. Note the folder path

### Step 2: Mount Google Drive in Colab

Add this cell at the beginning of your first notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Navigate to Your Project Folder

```python
import os
# Change to your project folder path
project_path = '/content/drive/MyDrive/path/to/your/project'
os.chdir(project_path)
!pwd  # Verify you're in the right directory
!ls   # List files to verify everything is there
```

### Step 4: Install Dependencies

```python
!pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Step 5: Run Notebooks

Now you can run the notebooks normally - they'll access files from Google Drive.

---

## Quick Setup Template for Colab

Copy this entire cell into a new Colab notebook to set everything up:

```python
# ========================================
# Quick Setup for Colab
# ========================================

# Step 1: Install dependencies
!pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Step 2: Download Python modules from GitHub (if hosted)
# OR upload them manually (see Method 2)

# Step 3: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Step 4: Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("Setup complete!")
```

---

## Important Colab-Specific Notes

### 1. File Persistence
- Files uploaded to Colab are **temporary** and will be deleted when the runtime is disconnected
- Use Google Drive mount (Method 3) for permanent storage

### 2. Session Timeout
- Colab sessions timeout after inactivity (~90 minutes)
- Save your work frequently, especially `preprocessed_data.pkl`

### 3. Downloading Results

After running the analysis, download your results:

```python
# Download preprocessed data
files.download('preprocessed_data.pkl')

# Download results CSV
files.download('results_summary.csv')

# Download all images
import zipfile
with zipfile.ZipFile('results.zip', 'w') as zipf:
    zipf.write('knn_learning_curve.png')
    zipf.write('rf_learning_curve.png')
    zipf.write('rf_feature_importance.png')
    zipf.write('confusion_matrices.png')
    zipf.write('roc_curves.png')
    zipf.write('model_comparison.png')
files.download('results.zip')
```

### 4. Display Images

Colab will display images inline automatically when you use `plt.show()`

### 5. GPU/TPU (Optional)

You generally don't need GPU for this project, but if you want to enable it:
- Runtime → Change runtime type → GPU

---

## Complete Step-by-Step Process

### First Notebook (01_EDA_and_Preprocessing.ipynb):

1. **Install dependencies:**
   ```python
   !pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```

2. **Import libraries:**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   import warnings
   warnings.filterwarnings('ignore')
   ```

3. **Upload Python modules** (if using Method 2, create them first)

4. **Run all remaining cells** - the dataset will download automatically

5. **Download `preprocessed_data.pkl`** after completion:
   ```python
   from google.colab import files
   files.download('preprocessed_data.pkl')
   ```

### Second Notebook (02_Main_Analysis.ipynb):

1. **Install dependencies** (same as above)

2. **Upload `preprocessed_data.pkl`** from previous step:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. **Run all cells**

4. **Download results** after completion (images and CSV files)

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure all `.py` files are uploaded/created in Colab

### Issue: File not found errors
**Solution:** Check your current directory with `!pwd` and `!ls`

### Issue: Runtime disconnects
**Solution:** Keep the browser tab active, or use Google Drive mount for persistence

### Issue: Can't find uploaded files
**Solution:** Files uploaded with `files.upload()` are in the current directory. Check with `!ls`

---

## Pro Tips

1. **Use Drive Mount:** Most reliable for keeping files across sessions
2. **Save frequently:** Download important files before closing
3. **Use git:** If you have the project on GitHub, clone it in Colab:
   ```python
   !git clone https://github.com/yourusername/your-repo.git
   %cd your-repo
   ```
4. **Check RAM:** Click on RAM/Disk indicator to see usage (Colab has limits)

---

**Need the Python file contents?** They're all in your project directory - just copy-paste them into Colab cells using `%%writefile filename.py` at the top of each cell.
