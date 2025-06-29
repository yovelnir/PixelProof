# Imbalanced Data Evaluation Metrics

## 🎯 **Training Context**

Your models were trained with **class weights: {0: 4.1, 1: 1}**, indicating:
- **Fake images (class 0)**: Weight = 4.1 (higher penalty for misclassification)
- **Real images (class 1)**: Weight = 1.0 (standard penalty)

This suggests significant **class imbalance** in the training data, with many more fake samples than real samples.

## 📊 **Updated Evaluation Metrics**

The evaluation system now includes **imbalance-aware metrics** that provide more accurate performance assessment:

### **🌟 Primary Metrics (Recommended for Imbalanced Data)**

1. **Balanced Accuracy** ⭐
   - Formula: `(Sensitivity + Specificity) / 2`
   - **Why better**: Accounts for performance on both classes equally
   - **Range**: 0.0 to 1.0 (higher is better)
   - **Interpretation**: 0.5 = random performance, 1.0 = perfect

2. **Weighted F1-Score** ⭐  
   - Accounts for class frequency in the test set
   - **Why better**: Balances precision and recall across both classes
   - **Use**: Primary metric for model comparison

3. **AUC-ROC** 
   - **Why good**: Threshold-independent, robust to class imbalance
   - **Already included**: Shows discriminative ability

### **🔍 Class-Specific Metrics**

- **Fake Detection F1**: Performance specifically on fake images
- **Real Detection F1**: Performance specifically on real images  
- **Fake Precision/Recall**: Detailed fake detection performance
- **Real Precision/Recall**: Detailed real detection performance

## 📈 **Confusion Matrix Interpretation**

```
                    Predicted
                 Fake    Real
Actual   Fake  │  120  │   15  │  ← Fake images
         Real  │   8   │  157  │  ← Real images
```

### **In Your Context:**

- **True Negatives (TN)**: Correctly identified fake images ✅
- **False Positives (FP)**: Real images incorrectly called fake ⚠️
- **False Negatives (FN)**: Fake images incorrectly called real ⚠️⚠️ *(More critical!)*
- **True Positives (TP)**: Correctly identified real images ✅

**⚠️ False Negatives are particularly concerning** because:
- Missing fake content can have serious consequences
- Your models were trained to be more sensitive to fake detection (weight 4.1)

## 🏆 **Model Ranking System**

Models are now ranked using:
- **60% Balanced Accuracy** (primary metric)
- **40% Weighted F1-Score** (secondary metric)

This combination provides robust evaluation for imbalanced scenarios.

## 📋 **How to Interpret Results**

### **✅ Good Performance Indicators:**
- `Balanced Accuracy > Standard Accuracy` - Model handles imbalance well
- `High Fake F1-Score` - Good at detecting fake content
- `Low False Negative Rate` - Doesn't miss fake images
- `High AUC (>0.8)` - Strong discriminative ability

### **⚠️ Warning Signs:**
- `Balanced Accuracy << Standard Accuracy` - Model biased toward majority class
- `Low Fake F1-Score` - Poor fake detection
- `High False Negative Rate` - Missing too many fake images
- `Big difference between Fake F1 and Real F1` - Imbalanced performance

## 🔧 **Evaluation Output**

You'll now see:

1. **Individual Detailed Matrices** for each model:
   ```
   CONFUSION MATRIX - Model Name
   ======================================================================
   Balanced Accuracy: 0.923 ⭐ (Better for imbalanced data)
   F1-Weighted: 0.932 ⭐ (Accounts for class imbalance)
   
   CLASS-SPECIFIC PERFORMANCE:
   Fake Detection: 0.889 (120/135 correct)
   Real Detection: 0.951 (157/165 correct)
   
   IMBALANCE-AWARE ASSESSMENT:
   ✅ Balanced accuracy (0.923) > Standard accuracy (0.910)
   Model performs well considering class imbalance
   ```

2. **Summary Comparison Table**:
   ```
   Model Name                TN  FP  FN  TP  Acc   Bal_Acc  F1   F1_W   Fake_F1  Real_F1
   SRM-CNN-L256              120 15  8   157 0.923 0.920    0.932 0.928  0.889    0.975
   DCT-AE-New                118 17  12  153 0.903 0.900    0.913 0.910  0.874    0.953
   ENSEMBLE                  125 10  5   160 0.950 0.948    0.955 0.952  0.926    0.985
   ```

## 🎯 **Key Takeaways**

- **Focus on Balanced Accuracy and Weighted F1** for overall performance
- **Monitor Fake F1-Score** for fake detection capability  
- **Watch False Negative Rate** (fake images called real)
- **Use ensemble if it shows better balanced performance**

The updated metrics provide a more accurate and fair evaluation of your models' performance on imbalanced deepfake detection data! 🚀 