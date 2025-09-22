# iris_pipeline.py
from typing import Tuple
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    return X, y, target_names

def quick_explore(X: np.ndarray, y: np.ndarray, target_names: list) -> None:
    print("Shape:", X.shape)
    print("Targets:", target_names)
    print("First row:", X[0].tolist())
    print("Class distribution:", {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

def build_baseline_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

def train_and_eval(test_size: float = 0.2, random_state: int = 42) -> float:
    X, y, target_names = load_data()
    quick_explore(X, y, target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_baseline_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    train_and_eval()
