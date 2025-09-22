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


# iris_pipeline.py
from typing import Tuple, Dict, Any
import os
import json
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # make sure requirements.txt includes joblib

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    return X, y, target_names

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

def train_eval_save(
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: str = "models",
    save_artifacts: bool = True
) -> Dict[str, Any]:
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()

    # cross validation on train
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    # fit on full train
    pipe.fit(X_train, y_train)

    # evaluate on test
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clsrep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "cv_mean_accuracy": float(np.mean(cv_scores)),
        "cv_std_accuracy": float(np.std(cv_scores)),
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": clsrep
    }

    print(json.dumps({"test_accuracy": metrics["test_accuracy"]}, indent=2))
    print("CV mean accuracy:", metrics["cv_mean_accuracy"])

    if save_artifacts:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipe, os.path.join(model_dir, "iris_logreg.pkl"))
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

def predict(sample: list, model_path: str = "models/iris_logreg.pkl") -> int:
    """Predict class index for a single sample [sepal_len, sepal_wid, petal_len, petal_wid]."""
    model = joblib.load(model_path)
    arr = np.array(sample).reshape(1, -1)
    pred = model.predict(arr)[0]
    return int(pred)

if __name__ == "__main__":
    train_eval_save()
# iris_pipeline.py
from typing import Tuple, Dict, Any
import os
import json
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # make sure requirements.txt includes joblib

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    return X, y, target_names

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

def train_eval_save(
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: str = "models",
    save_artifacts: bool = True
) -> Dict[str, Any]:
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()

    # cross validation on train
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    # fit on full train
    pipe.fit(X_train, y_train)

    # evaluate on test
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clsrep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "cv_mean_accuracy": float(np.mean(cv_scores)),
        "cv_std_accuracy": float(np.std(cv_scores)),
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": clsrep
    }

    print(json.dumps({"test_accuracy": metrics["test_accuracy"]}, indent=2))
    print("CV mean accuracy:", metrics["cv_mean_accuracy"])

    if save_artifacts:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(pipe, os.path.join(model_dir, "iris_logreg.pkl"))
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

def predict(sample: list, model_path: str = "models/iris_logreg.pkl") -> int:
    """Predict class index for a single sample [sepal_len, sepal_wid, petal_len, petal_wid]."""
    model = joblib.load(model_path)
    arr = np.array(sample).reshape(1, -1)
    pred = model.predict(arr)[0]
    return int(pred)

if __name__ == "__main__":
    train_eval_save()
