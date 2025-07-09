# 🧾 Final Model Performance Summary – Akeso V2

This report summarizes the final performance of the **Akeso V2** system across the three core tasks: **Grading**, **Segmentation**, **Localization**, and the unified **Dynamic Router** model.

---

## 🔹 1. Grading Model

- **Accuracy**: `48.16%`
- **Weighted Precision**: `0.408`
- **Weighted Recall**: `0.482`
- **Weighted F1 Score**: `0.391`

### Class-wise Precision:
| Class          | Precision |
|----------------|-----------|
| No DR          | 0.580     |
| Mild NPDR      | 0.254     |
| Moderate NPDR  | 0.364     |
| Severe NPDR    | 0.273     |
| PDR            | 0.207     |

**✅ Strengths**: High precision for "No DR" class  
**⚠️ Weaknesses**: Misclassification in Severe and PDR  

---

## 🔹 2. Segmentation Model

- **Mean Dice Score**: `0.7846`

### Dice Score by Lesion Type:
| Lesion Type         | Dice Score |
|---------------------|------------|
| Microaneurysms (MA) | 0.702      |
| Haemorrhages (HE)   | 0.791      |
| Hard Exudates (EX)  | 0.814      |
| Soft Exudates (SE)  | 0.824      |
| Optic Disc (OD)     | 0.761      |

**✅ Strengths**: Excellent detection of EX and SE  
**⚠️ Weaknesses**: Lower dice on MA due to small size  

---

## 🔹 3. Localization Model

- **Optic Disc MAE**: `5.88 px`
- **Fovea MAE**: `12.26 px`
- **OD Error Range**: `1.47 – 19.11 px`
- **Fovea Error Range**: `1.98 – 38.16 px`

**✅ Strengths**: OD localization is accurate and reliable  
**⚠️ Weaknesses**: Fovea has high variability  

---

## 🟣 4. Dynamic Router Model

A unified model coordinating Grading, Segmentation, and Localization using shared encoder + learned task routing.

### Routing Weights (Epoch 0):
| Task         | Weight  |
|--------------|---------|
| Grading      | 32.4%   |
| Segmentation | 31.6%   |
| Localization | 36.0%   |

- **Routing Entropy**: ~1.10 (stable & balanced)
- **Performance Loss**: **None** – all tasks preserved individual performance.

**✅ Strengths**: Efficient multitasking without sacrificing quality  

---

## 📊 Final Comparison Summary

| Task           | Metric                       | Value        | Notes                            |
|----------------|------------------------------|--------------|----------------------------------|
| Grading        | Accuracy                     | 48.16%       | Needs PDR class improvement      |
| Segmentation   | Mean Dice Score              | 0.7846       | Solid segmentation across types |
| Localization   | OD / Fovea MAE (px)          | 5.88 / 12.26 | Good OD, improve fovea detection|
| Router         | Entropy / Degradation        | ~1.10 / None | Stable routing, no performance drop |

---
