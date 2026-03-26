#  Student Grade Predictor
#  Author : KRISHNA PALIWAL
#  Course : Fundamentals of AI and ML 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# 1.  GENERATE SYNTHETIC DATASET

np.random.seed(42)
N = 300

attendance       = np.random.randint(50, 100, N)          # % attendance
study_hours      = np.random.uniform(1, 10, N)            # hrs/day
prev_score       = np.random.randint(40, 100, N)          # previous exam score
assignments_done = np.random.randint(0, 10, N)            # out of 10
sleep_hours      = np.random.uniform(4, 9, N)             # hrs/night

# Realistic score formula with noise

final_score = (
    0.30 * attendance
    + 4.5 * study_hours
    + 0.25 * prev_score
    + 1.5 * assignments_done
    + 1.0 * sleep_hours
    + np.random.normal(0, 5, N)
).clip(0, 100)

# Grade labels
def to_grade(s):
    if s >= 90: return "A"
    elif s >= 75: return "B"
    elif s >= 60: return "C"
    elif s >= 45: return "D"
    else: return "F"

grade_labels = [to_grade(s) for s in final_score]

df = pd.DataFrame({
    "attendance_pct":    attendance,
    "study_hours_day":   study_hours.round(2),
    "prev_exam_score":   prev_score,
    "assignments_done":  assignments_done,
    "sleep_hours":       sleep_hours.round(2),
    "final_score":       final_score.round(2),
    "grade":             grade_labels
})

print("=" * 55)
print("  STUDENT GRADE PREDICTOR")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print("\nFirst 5 rows :")
print(df.head().to_string(index=False))
print("\nGrade distribution :")
print(df["grade"].value_counts().sort_index().to_string())


# 2.  EXPLORATORY DATA ANALYSIS  (save plots)

import os
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Student Grade Predictor – EDA", fontsize=16, fontweight="bold")

cols   = ["attendance_pct", "study_hours_day", "prev_exam_score",
          "assignments_done", "sleep_hours", "final_score"]
colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]

for ax, col, color in zip(axes.flatten(), cols, colors):
    ax.hist(df[col], bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("plots/01_eda_distributions.png", dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop("grade", axis=1).corr(), annot=True, fmt=".2f",
            cmap="Blues", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_correlation_heatmap.png", dpi=150)
plt.close()


# 3.  MODEL A – LINEAR REGRESSION (predict score)

features = ["attendance_pct","study_hours_day","prev_exam_score",
            "assignments_done","sleep_hours"]
X = df[features]
y_reg = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2   = r2_score(y_test, y_pred_lr)

print("\n─────────────────────────────────────────")
print("  LINEAR REGRESSION (Score Prediction)")
print("─────────────────────────────────────────")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print("\n  Feature Coefficients:")
for feat, coef in zip(features, lr.coef_):
    print(f"    {feat:<25} {coef:+.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color="#4C72B0", edgecolors="white", s=60)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Linear Regression: Actual vs Predicted Score", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("plots/03_lr_actual_vs_predicted.png", dpi=150)
plt.close()


# 4.  MODEL B – DECISION TREE (predict grade)

le = LabelEncoder()
y_clf = le.fit_transform(df["grade"])          # A=0 B=1 C=2 D=3 F=4

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train2, y_train2)
y_pred_dt = dt.predict(X_test2)

acc = accuracy_score(y_test2, y_pred_dt)

print("\n─────────────────────────────────────────")
print("  DECISION TREE CLASSIFIER (Grade)")
print("─────────────────────────────────────────")
print(f"  Accuracy : {acc*100:.2f}%")
print("\n  Classification Report:")
present_labels = np.unique(np.concatenate([y_test2, y_pred_dt]))
present_names  = le.inverse_transform(present_labels)
print(classification_report(y_test2, y_pred_dt,
                             labels=present_labels,
                             target_names=present_names))

# Confusion matrix
cm = confusion_matrix(y_test2, y_pred_dt)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5)
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.title("Decision Tree – Confusion Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_dt_confusion_matrix.png", dpi=150)
plt.close()

# Decision tree plot
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=features, class_names=le.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree Visualization (max_depth=5)", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/05_decision_tree.png", dpi=150)
plt.close()


# 5.  INTERACTIVE PREDICTOR  (simple CLI)

print("\n─────────────────────────────────────────")
print("  SAMPLE PREDICTION (new student data)")
print("─────────────────────────────────────────")

sample = pd.DataFrame([{
    "attendance_pct":   85,
    "study_hours_day":  6.0,
    "prev_exam_score":  72,
    "assignments_done": 8,
    "sleep_hours":      7.0
}])

pred_score = lr.predict(sample)[0]
pred_grade_enc = dt.predict(sample)[0]
pred_grade = le.inverse_transform([pred_grade_enc])[0]

print(f"  Input  → Attendance: 85%, Study: 6h, Prev Score: 72,")
print(f"            Assignments: 8/10, Sleep: 7h")
print(f"  Output → Predicted Score : {pred_score:.1f} / 100")
print(f"           Predicted Grade : {pred_grade}")
print("\n✅  All plots saved to  ./plots/")
print("=" * 55)
