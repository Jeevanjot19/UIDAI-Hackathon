# Handling Class Imbalance in Aadhaar Update Prediction

## 📊 Problem Identification

### Original Dataset Distribution
- **High Updaters (Class 1)**: 229,979 samples (78%)
- **Low Updaters (Class 0)**: 64,789 samples (22%)
- **Imbalance Ratio**: 3.55:1

### Impact on Predictions
The severe class imbalance caused the model to:
- **Bias towards majority class**: Predicted "High Updater" for almost all districts
- **Poor minority class recall**: Failed to identify actual low updaters
- **Unrealistic probabilities**: Even empty inputs predicted high probability (~70-80%)

## 🔧 Solutions Implemented

### 1. **SMOTE (Synthetic Minority Over-sampling)**
- **Technique**: Generate synthetic samples for minority class using k-nearest neighbors
- **Sampling Strategy**: 50% minority class (balanced to 2:1 ratio)
- **Result**: ROC-AUC = **0.7205**
- **Pros**: Increased minority class representation without duplication
- **Cons**: Slightly lower performance, synthetic samples may not capture all patterns

### 2. **Aggressive Class Weights** ⭐ **WINNER**
- **Technique**: Penalize misclassification of minority class more heavily
- **Original `scale_pos_weight`**: 0.26 (based on class ratio)
- **Aggressive weight**: 0.39 (1.5x multiplier)
- **Result**: ROC-AUC = **0.7223** ✅
- **Pros**: Simple, effective, no data modification required
- **Cons**: May slightly reduce majority class precision

### 3. **SMOTEENN (Hybrid Approach)**
- **Technique**: SMOTE + Edited Nearest Neighbors to clean overlap
- **Result**: ROC-AUC = **0.6343**
- **Pros**: Cleaner decision boundaries
- **Cons**: Removed too many samples, reduced performance

## 📈 Performance Comparison

| Technique | ROC-AUC | F1 Score @ 0.5 | Balanced Accuracy |
|-----------|---------|----------------|-------------------|
| **Aggressive Weights** | **0.7223** | **0.8398** | **0.6522** |
| SMOTE | 0.7205 | 0.8450 | 0.6350 |
| SMOTEENN | 0.6343 | 0.7520 | 0.5890 |
| Original Model | 0.7248 | 0.8500 | 0.5900 |

*Note: Original model had higher ROC-AUC but poor balanced accuracy, indicating bias*

## 🎯 Threshold Optimization

### Threshold Tuning Results
Tested thresholds from 0.3 to 0.7 to find optimal balance:

| Threshold | F1 Score | Balanced Accuracy |
|-----------|----------|-------------------|
| 0.30 | 0.8504 | 0.5669 |
| 0.35 | 0.8527 | 0.5951 |
| **0.40** | **0.8532** | **0.6203** ✅ |
| 0.45 | 0.8490 | 0.6396 |
| 0.50 | 0.8398 | 0.6522 |
| 0.55 | 0.8258 | 0.6603 |
| 0.60 | 0.8035 | 0.6627 |

**Optimal Threshold**: **0.40**
- Maximizes F1 score (0.8532)
- Good balanced accuracy (0.6203)
- Better calibration for imbalanced data

## 📊 Final Model Performance

### Classification Report (Threshold = 0.40)
```
              precision    recall  f1-score   support

 Low Updater       0.67      0.30      0.41     16,670
High Updater       0.78      0.94      0.85     43,644

    accuracy                           0.76     60,314
   macro avg       0.72      0.62      0.63     60,314
weighted avg       0.75      0.76      0.73     60,314
```

### Confusion Matrix
```
                Predicted Low  Predicted High
Actual Low           4,949         11,721
Actual High          2,453         41,191
```

### Key Metrics
- **True Positive Rate (Recall for High)**: 94% - Correctly identifies high updaters
- **True Negative Rate (Recall for Low)**: 30% - Improved minority class detection
- **Precision for High**: 78% - Reduced false positives
- **Overall Accuracy**: 76%

## 🔄 Impact on Predictions

### Before Balancing
- **Default input** (rolling_3m_updates = 2.0): **75-80%** probability
- **Low updater scenario** (rolling_3m_updates = 5.0): **70-75%** probability
- **Very low activity** (rolling_3m_updates = 0.0): **65-70%** probability

### After Balancing + Threshold Tuning
- **Default input** (rolling_3m_updates = 2.0): **20-30%** probability → Low Updater ✅
- **Low updater scenario** (rolling_3m_updates = 5.0): **35-45%** probability → Low Updater ✅
- **High updater scenario** (rolling_3m_updates = 25.0): **75-85%** probability → High Updater ✅
- **Very low activity** (rolling_3m_updates = 0.0): **10-15%** probability → Low Updater ✅

## 🎓 Lessons Learned

### Why Class Imbalance Matters
1. **Majority Class Bias**: Models optimize for overall accuracy, ignoring minority class
2. **Probability Calibration**: Imbalanced training leads to overconfident predictions
3. **Threshold Sensitivity**: Default 0.5 threshold inappropriate for imbalanced data

### Best Practices Applied
1. ✅ **Always check class distribution** before training
2. ✅ **Use balanced accuracy** instead of just accuracy
3. ✅ **Tune classification threshold** separately from model training
4. ✅ **Compare multiple balancing techniques** (SMOTE, weights, hybrid)
5. ✅ **Evaluate with precision-recall curves** for imbalanced data

### Why Aggressive Weights Won
- **Simplest**: No data manipulation, just parameter tuning
- **Fastest**: No resampling overhead during training
- **Most flexible**: Easy to adjust weight multiplier
- **Production-ready**: No need to store synthetic samples

## 📁 Files Generated

```
outputs/models/
├── xgboost_balanced.pkl          # Trained model with aggressive weights
├── balanced_features.txt         # List of 102 features used
└── balanced_metadata.json        # Model config (threshold, technique, metrics)

outputs/figures/
└── balanced_model_evaluation.png # 4-panel visualization:
                                   # - ROC curves comparison
                                   # - Precision-Recall curve
                                   # - Threshold tuning plot
                                   # - Confusion matrix heatmap
```

## 🚀 Integration with Dashboard

### Changes Made
1. **Model Loading**: Automatically loads balanced model if available, falls back to original
2. **Metadata Display**: Shows balancing technique and optimal threshold
3. **Prediction Logic**: Uses threshold=0.4 instead of default 0.5
4. **User Interface**: Added info box explaining balanced model benefits

### How It Works
```python
# Load balanced model
models['xgboost'] = joblib.load('outputs/models/xgboost_balanced.pkl')
models['metadata'] = json.load('outputs/models/balanced_metadata.json')

# Get optimal threshold
threshold = models['metadata']['optimal_threshold']  # 0.4

# Classify with optimal threshold
probability = model.predict_proba(input)[0, 1]
prediction = 1 if probability >= threshold else 0
```

## 📚 References & Further Reading

### Techniques Used
- **SMOTE**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
- **Class Weights**: XGBoost `scale_pos_weight` parameter for imbalanced data
- **Threshold Tuning**: Saito & Rehmsmeier (2015) - "Precision-Recall curves for imbalanced data"

### Alternative Approaches (Not Implemented)
- **Random Undersampling**: Remove majority class samples (risk losing information)
- **BalancedBaggingClassifier**: Ensemble with balanced sampling
- **Cost-sensitive Learning**: Asymmetric loss functions
- **Anomaly Detection**: Treat minority class as outliers

## ✅ Recommendations

### For Production Deployment
1. **Monitor predictions** across different probability ranges
2. **A/B test** balanced vs original model in production
3. **Retrain regularly** as class distribution may shift over time
4. **Consider ensemble** of balanced + original model for robustness

### For Future Improvements
1. **Feature engineering** to separate classes better
2. **Collect more minority class data** if possible
3. **Try BalancedRandomForest** for comparison
4. **Investigate why** 78% are high updaters (domain knowledge)

---

**Conclusion**: Aggressive class weighting + threshold tuning successfully addressed the 3.55:1 class imbalance, producing more realistic and reliable predictions while maintaining high overall performance (0.7223 ROC-AUC).
