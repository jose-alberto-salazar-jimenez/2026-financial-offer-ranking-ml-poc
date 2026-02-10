"""Train propensity model (Logistic Regression + Gradient Boosting)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.pipelines.features import build_preprocessor, save_preprocessor, transform
from src.pipelines.ingest import load_raw
from src.utils.config import get_model_config
from src.utils.logging import get_logger
from src.utils.paths import get_model_dir, get_processed_data_dir

logger = get_logger(__name__)


def main() -> None:
    cfg = get_model_config()
    target = cfg["target"]
    ratio = cfg.get("train_split_ratio", 0.8)
    rs = cfg.get("random_state", 42)
    primary = cfg.get("primary_model", "gradient_boosting")

    df = load_raw()
    # Binary target
    y = (df[target].astype(str).str.lower() == "yes").astype(int).values
    preprocessor, feature_names = build_preprocessor(df)
    X = transform(preprocessor, df)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=ratio, random_state=rs, stratify=y
    )
    get_processed_data_dir().mkdir(parents=True, exist_ok=True)
    # Save split indices or processed data for evaluate step
    np.save(get_processed_data_dir() / "X_train.npy", X_train)
    np.save(get_processed_data_dir() / "X_test.npy", X_test)
    np.save(get_processed_data_dir() / "y_train.npy", y_train)
    np.save(get_processed_data_dir() / "y_test.npy", y_test)
    # Save full df for baseline stats (package_model)
    df.to_parquet(get_processed_data_dir() / "train_raw.parquet", index=False)

    models_cfg = cfg.get("models", {})
    lr_cfg = models_cfg.get("logistic_regression", {})
    gb_cfg = models_cfg.get("gradient_boosting", {})

    lr = LogisticRegression(**lr_cfg)
    lr.fit(X_train, y_train)
    logger.info("Logistic Regression train score: %.4f", lr.score(X_train, y_train))

    gb = GradientBoostingClassifier(**gb_cfg)
    gb.fit(X_train, y_train)
    logger.info("Gradient Boosting train score: %.4f", gb.score(X_train, y_train))

    model = gb if primary == "gradient_boosting" else lr
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    save_preprocessor(preprocessor, feature_names, model_dir)
    logger.info("Saved primary model (%s) and preprocessor to %s", primary, model_dir)


if __name__ == "__main__":
    main()
