# tests/test_iris_pipeline.py
import os
import json
import numpy as np

# Import from the repo root
import iris_pipeline


def test_load_data_shapes():
    X, y, target_names = iris_pipeline.load_data()
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert len(target_names) == 3  # Iris has 3 classes


def test_build_pipeline_type():
    # Your extended script likely uses build_pipeline()
    pipe = iris_pipeline.build_pipeline()
    # Fallback for the baseline name if you used build_baseline_pipeline()
    # pipe = iris_pipeline.build_baseline_pipeline()
    from sklearn.pipeline import Pipeline
    assert isinstance(pipe, Pipeline)


def test_train_eval_save_creates_metrics(tmp_path):
    # Train and save into a temporary models directory so tests do not touch your real repo artifacts
    metrics_dir = tmp_path / "models"
    result = iris_pipeline.train_eval_save(
        test_size=0.2,
        random_state=42,
        model_dir=str(metrics_dir),
        save_artifacts=True
    )

    # Basic checks on returned metrics
    assert isinstance(result, dict)
    for key in ["cv_mean_accuracy", "cv_std_accuracy", "test_accuracy", "confusion_matrix", "classification_report"]:
        assert key in result

    # Files created
    model_path = metrics_dir / "iris_logreg.pkl"
    metrics_path = metrics_dir / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()

    # Metrics.json is valid JSON and has expected keys
    with open(metrics_path) as f:
        saved = json.load(f)
    assert "test_accuracy" in saved
    assert isinstance(saved["confusion_matrix"], list)


def test_predict_runs_with_saved_model(tmp_path):
    # First train and persist a temp model
    model_dir = tmp_path / "models"
    iris_pipeline.train_eval_save(model_dir=str(model_dir), save_artifacts=True)

    # Use the saved model for a one off prediction
    demo = [5.1, 3.5, 1.4, 0.2]
    pred = iris_pipeline.predict(demo, model_path=str(model_dir / "iris_logreg.pkl"))
    assert pred in (0, 1, 2)

    # Input shape check
    bad = np.array([5.1, 3.5, 1.4, 0.2, 9.9])  # wrong length
    try:
        iris_pipeline.predict(bad.tolist(), model_path=str(model_dir / "iris_logreg.pkl"))
        raised = False
    except Exception:
        raised = True
    assert raised  # should fail on wrong shape


