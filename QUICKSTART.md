# 🚀 Quick Start Guide
## Aadhaar Societal Intelligence Project

**Get up and running in 5 minutes!**

---

## Step 1: Verify Setup

Open PowerShell in your project directory and check if you're in the right folder:

```powershell
cd "D:\UIDAI Hackathon"
Get-ChildItem
```

You should see:
- `config/`
- `data/`
- `notebooks/`
- `src/`
- `README.md`
- `requirements.txt`

---

## Step 2: Install Dependencies

### Option A: Using Conda (Recommended)

```powershell
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate aadhaar-analysis

# Verify installation
python -c "import pandas; import numpy; import matplotlib; print('✓ All libraries installed')"
```

### Option B: Using pip

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import numpy; import matplotlib; print('✓ All libraries installed')"
```

---

## Step 3: Test the Data Loader

```powershell
# Navigate to src directory
cd src

# Run data loader test
python data_loader.py
```

**Expected output**:
```
✓ Data loader initialized
Loading enrolment data...
Creating sample enrolment data for testing
...
Sample data created successfully!
```

This creates **sample data** automatically since you don't have real UIDAI files yet.

---

## Step 4: Test Feature Engineering

```powershell
# Still in src directory
python feature_engineering.py
```

**Expected output**:
```
Feature engineering complete. Total columns: 40+
✓ Layer 2 complete: 5 features added
✓ Layer 3 complete: 6 CORE features added
...
```

---

## Step 5: Launch Jupyter Notebooks

```powershell
# Go back to project root
cd ..

# Launch Jupyter
jupyter notebook
```

This opens your browser. Navigate to:
1. `notebooks/01_data_loading.ipynb`
2. Run all cells (Cell → Run All)

---

## Step 6: Quick Data Analysis (Manual Test)

Create a test script to see everything working:

```powershell
# Create test file
@"
import sys
sys.path.append('src')

from data_loader import quick_load
from feature_engineering import quick_feature_engineering
from visualization import AadhaarVisualizer

# Load data
print("Loading data...")
df = quick_load(sample_data=True)
print(f"✓ Loaded {len(df)} records")

# Engineer features
print("\nEngineering features...")
df_features = quick_feature_engineering(df)
print(f"✓ Created {len(df_features.columns)} total columns")

# Show key features
print("\nKey Societal Indicators:")
indicators = [
    'mobility_indicator',
    'digital_instability_index', 
    'identity_stability_score',
    'update_burden_index'
]
print(df_features[indicators].describe())

print("\n✅ SUCCESS! All systems operational.")
"@ | Out-File -FilePath test_quick_run.py -Encoding UTF8

# Run the test
python test_quick_run.py
```

---

## Step 7: Generate Your First Visualization

```powershell
@"
import sys
sys.path.append('src')

from data_loader import quick_load
from feature_engineering import quick_feature_engineering
from visualization import AadhaarVisualizer
import matplotlib.pyplot as plt

# Load and prepare data
df = quick_load(sample_data=True)
df = quick_feature_engineering(df)

# Create visualizer
viz = AadhaarVisualizer()

# Plot Identity Stability Dashboard
print("Creating Identity Stability Dashboard...")
fig = viz.plot_identity_stability_dashboard(df, save_path='outputs/figures/identity_stability_dashboard.png')
plt.show()

print("✓ Dashboard saved to outputs/figures/identity_stability_dashboard.png")
"@ | Out-File -FilePath test_visualization.py -Encoding UTF8

python test_visualization.py
```

---

## What You've Just Done ✅

1. ✅ Set up Python environment
2. ✅ Tested data loading (with sample data)
3. ✅ Tested feature engineering (25+ features)
4. ✅ Generated your first visualization
5. ✅ Verified entire pipeline works

---

## Next Steps

### Option A: Exploratory Analysis (Recommended First)

1. Open `notebooks/01_data_loading.ipynb`
2. Run all cells
3. Examine the outputs
4. Proceed to notebooks 02-11

### Option B: Add Real UIDAI Data

1. Download UIDAI datasets from official source
2. Place in appropriate folders:
   - Enrolment → `data/raw/enrolment/`
   - Demographic → `data/raw/demographic_update/`
   - Biometric → `data/raw/biometric_update/`
3. Rerun notebooks (will automatically use real data)

### Option C: Jump to ML Models

1. Review `PROJECT_STATUS.md` for pending tasks
2. Implement forecasting models next
3. Follow the plan in TODO list

---

## Common Issues & Solutions

### Issue 1: "Module not found"
**Solution**: Make sure you're in the project root directory and Python path includes `src/`

```powershell
cd "D:\UIDAI Hackathon"
$env:PYTHONPATH="D:\UIDAI Hackathon\src"
```

### Issue 2: "Permission denied" on PowerShell scripts
**Solution**: 
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Issue 3: Plots not showing
**Solution**: Use `plt.show()` or save to file instead

```python
fig.savefig('output.png')
```

### Issue 4: Out of memory
**Solution**: Work with sampled data

```python
df_sample = df.sample(frac=0.1)  # Use 10% of data
```

---

## Project Structure Reminder

```
📁 Your current working directory
│
├── 📁 config/          # Configuration files
├── 📁 data/            # Data files (raw & processed)
├── 📁 notebooks/       # Jupyter notebooks (YOUR WORKSPACE)
├── 📁 src/             # Python modules (pre-built for you)
├── 📁 outputs/         # Generated visualizations
└── 📁 models/          # Saved ML models
```

---

## Quick Commands Reference

```powershell
# Activate environment
conda activate aadhaar-analysis

# Run data loader
python src/data_loader.py

# Run feature engineering
python src/feature_engineering.py

# Launch Jupyter
jupyter notebook

# Run a specific notebook
jupyter nbconvert --execute --to notebook notebooks/01_data_loading.ipynb

# Check project status
cat PROJECT_STATUS.md
```

---

## Get Help

1. **Check logs**: All modules have logging enabled
2. **Read docstrings**: `help(UidaiDataLoader)`
3. **Review examples**: Each module has `if __name__ == "__main__"` test code
4. **Check status**: `PROJECT_STATUS.md` has implementation checklist

---

## Ready to Start! 🎯

You now have:
- ✅ Complete project infrastructure
- ✅ 25+ engineered features
- ✅ Data loading pipeline
- ✅ Visualization toolkit
- ✅ Sample data for testing

**Time to analyze and win the hackathon!** 🏆

---

**Pro Tip**: Start with the notebooks in sequence (01 → 02 → 03...) to build your understanding systematically.
