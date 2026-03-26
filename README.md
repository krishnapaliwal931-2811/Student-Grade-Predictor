# 🎓 Student Grade Predictor

A beginner-level Machine Learning project that predicts a student's final exam score and letter grade based on academic habits and past performance, using **Linear Regression** and a **Decision Tree Classifier**.

---

## 📌 Problem Statement

Students often wonder: *"How will I perform at the end of this semester?"*  
This project answers that question by analyzing key factors — attendance, study hours, previous scores, assignment completion, and sleep — to predict both a **numeric score** and a **letter grade (A–F)**.

---

## 🚀 Features

- 🔢 **Score Prediction** — Predicts final score (0–100) using Linear Regression
- 🔤 **Grade Classification** — Predicts letter grade (A/B/C/D/F) using a Decision Tree
- 📊 **EDA Visualizations** — Histograms, correlation heatmap, actual vs predicted plots
- 🌳 **Tree Visualization** — Visual diagram of the decision tree
- 🖥️ **Sample Predictor** — Plug in a student's data and get instant output

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models (LinearRegression, DecisionTreeClassifier) |
| matplotlib | Plotting |
| seaborn | Statistical visualizations |

---

## 📁 Project Structure

```
student_grade_predictor/
│
├── grade_predictor.py       # Main script (data + models + plots)
├── plots/                   # Auto-generated output plots
│   ├── 01_eda_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_lr_actual_vs_predicted.png
│   ├── 04_dt_confusion_matrix.png
│   └── 05_decision_tree.png
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/student-grade-predictor.git
cd student-grade-predictor
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Project
```bash
python grade_predictor.py
```

That's it! The script will:
- Generate a synthetic dataset of 300 students
- Train and evaluate both ML models
- Print results to the terminal
- Save all plots to the `plots/` folder

---

## 📊 Input Features

| Feature | Description | Example |
|---------|-------------|---------|
| `attendance_pct` | % of classes attended | 85 |
| `study_hours_day` | Average daily study hours | 6.0 |
| `prev_exam_score` | Score in previous exam | 72 |
| `assignments_done` | Assignments submitted (out of 10) | 8 |
| `sleep_hours` | Average nightly sleep | 7.0 |

---

## 🎯 Sample Output

```
Input  → Attendance: 85%, Study: 6h, Prev Score: 72,
          Assignments: 8/10, Sleep: 7h
Output → Predicted Score : 88.0 / 100
         Predicted Grade : B
```

---

## 📈 Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Linear Regression | R² Score | 0.87 |
| Linear Regression | MAE | ~3.9 |
| Decision Tree | Accuracy | ~53% |

> The linear regression performs well. The decision tree has moderate accuracy due to grade boundaries being close together — a normal challenge in grade classification.

---

## 💡 Key Insights

- **Study hours per day** has the highest influence on final score (coefficient ≈ 4.17)
- Students attending **>80%** of classes almost always score above 70
- Sleep and assignment completion both contribute positively to performance

---

## 🧠 Concepts Used (from Course)

- Supervised Learning
- Linear Regression
- Decision Tree Classification
- Train/Test Split
- Model Evaluation (MAE, RMSE, R², Accuracy, Confusion Matrix)
- Feature Engineering

---

## 👤 Author

**KRISHNA PALIWAL** 
Registration Number: 25BAI11317
Course: Fundamentals of AI and ML  
Platform: VITyarthi

---

## 📄 License

This project is created for academic purposes as part of the BYOP (Bring Your Own Project) assignment.
