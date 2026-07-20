import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Učitavanje dataseta
data, _ = arff.loadarff("data/phishing_dataset.arff")
df = pd.DataFrame(data)

# 2. Konverzija i obrada podataka
for col in df.columns:
    df[col] = df[col].astype(float).astype(int)

X = df.drop(columns=["Result"])
y = (df["Result"] == -1).astype(int)  # -1 = phishing → 1, 1 = legitimno → 0

# 3. Podjela na trening i test skup (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Definisanje modela
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

# 5. Treniranje i evaluacija
results = {}
print(f"\n{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results[name] = [acc, prec, rec, f1]
    print(f"{name:<22} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

print("-" * 65)

# 6. Vizualizacija – usporedba metrika modela
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_names  = list(results.keys())
values       = np.array(list(results.values()))

x     = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(model_names):
    ax.bar(x + i * width, values[i], width, label=name)

ax.set_xlabel("Metrika")
ax.set_ylabel("Vrijednost")
ax.set_title("Usporedba performansi modela")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0.85, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/metrics_comparison.png", dpi=150)
plt.show()
