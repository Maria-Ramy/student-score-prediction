# Student Score Prediction 🎓

A machine learning project that predicts student exam scores based on study habits and personal factors.

## 📊 Project Overview
- Analyzed **6,600+ student records**
- Built **5 regression models** (Linear, Polynomial, Ridge, Lasso)
- Achieved **99.8% accuracy** with Polynomial Regression
- Identified key factors affecting student performance

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 📈 Results
| Model | R² Score |
|-------|----------|
| Simple Linear | 0.23 |
| Multiple Linear | 0.66 |
| **Polynomial (deg 2)** | **0.998** |
| Ridge | 0.66 |
| Lasso | 0.66 |

## 📁 Project Structure
student-score-prediction/
├── main.py # Main script
├── requirements.txt # Dependencies
├── StudentPerformanceFactors.csv # Dataset
├── results/ # All outputs
│ ├── correlation_heatmap.png
│ ├── score_distribution.png
│ ├── hours_vs_score.png
│ ├── categorical_impact.png
│ ├── feature_importance.png
│ ├── model_comparison.png
│ ├── best_model_predictions.png
│ └── model_comparison_results.csv
└── README.md
