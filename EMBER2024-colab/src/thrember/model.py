import os
import json
import multiprocessing
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import tqdm
import optuna
import xgboost as xgb
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from .features import PEFeatureExtractor

ORDERED_COLUMNS = [
    "sha256",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "label",
    "file_type",
    "family",
    "family_confidence",
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
]


def raw_feature_iterator(file_paths: list[Path]) -> Iterator[str]:
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with path.open("r") as fin:
            for line in fin:
                yield line


def gather_feature_paths(data_dir: Path | str, subset: str, filetype: str = None, week: str = None) -> list[Path]:
    """
    Gather paths to raw metadata .jsonl files in the given data_dir
    Supports filtering by train/test/challenge subset, file type, and/or data collection week
    """
    data_path = Path(data_dir)
    feature_paths = []
    for path in sorted(data_path.rglob("*.jsonl")):
        file_name = path.name
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(path)

    if not len(feature_paths):
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths


def read_label(raw_features_string: str, label_type: str) -> str:
    """
    Read the label or tag from raw features and return it
    """
    raw_features = json.loads(raw_features_string)
    label = raw_features[label_type]
    return label


def read_label_unpack(args):
    """
    Pass through function for unpacking read_label arguments
    """
    return read_label(*args)


def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> dict:
    """
    Read the unique labels/tags in the subset.
    Prints progress every 20,000 completed rows.
    """
    pool = multiprocessing.Pool()
    argument_iterator = (
        (raw_features_string, label_type)
        for _, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )

    label_counts = {}
    try:
        for completed, labels in enumerate(
            pool.imap_unordered(read_label_unpack, argument_iterator),
            start=1,
        ):
            if not isinstance(labels, list):
                labels = [labels]

            for label in labels:
                if label_counts.get(label) is None:
                    label_counts[label] = 0
                label_counts[label] += 1

            if completed % 20000 == 0 or completed == nrows:
                print(f"Processed {completed}/{nrows} rows")
    finally:
        pool.close()
        pool.join()

    return label_counts
# def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> dict:
#     """
#     Read the unique labels/tags in the subset.
#     """
#     argument_iterator = (
#         (raw_features_string, label_type)
#         for raw_features_string in raw_feature_iterator(raw_feature_paths)
#     )

#     label_counts = {}
#     for args in tqdm.tqdm(argument_iterator, total=nrows):
#         labels = read_label_unpack(args)
#         if not isinstance(labels, list):
#             labels = [labels]
#         for label in labels:
#             label_counts[label] = label_counts.get(label, 0) + 1

#     return label_counts


def vectorize(irow: int, raw_features_string: str, X_path: str, y_path: str, extractor: PEFeatureExtractor, nrows: int, label_type: str = "label", label_map: dict = {}) -> None:
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError("Invalid label_type!")
    label = raw_features[label_type]

    # Figure out what 'label' is
    if label is None and (label_type == "label" or label_type == "family"):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = -1
    elif isinstance(label, int): # Benign/Malicious labels (binary)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = label
    elif isinstance(label, str): # Family labels (multiclass)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        if label_map.get(label) is not None:
            y[irow] = label_map[label]
        else:
            y[irow] = -1
    elif isinstance(label, list): # Tags (multiclass, multilabel)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows, len(label_map.keys())))
        for l in label:
            if label_map.get(l) is not None:
                y[irow,label_map[l]] = 1
    else:
        raise ValueError("Unable to parse label format")

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(
    X_path: Path,
    y_path: Path,
    raw_feature_paths: list[Path],
    extractor: PEFeatureExtractor,
    nrows: int,
    label_type: str = "label",
    label_map: dict = {},
) -> None:
    """
    Vectorize a subset of data and write it to disk.
    Prints progress every 20,000 completed rows.
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    if label_type == "label" or label_type == "family":
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    else:
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(nrows, len(label_map.keys())))
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = (
        (irow, raw_features_string, X_path, y_path, extractor, nrows, label_type, label_map)
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )

    try:
        for completed, _ in enumerate(
            pool.imap_unordered(vectorize_unpack, argument_iterator),
            start=1,
        ):
            if completed % 20000 == 0 or completed == nrows:
                print(f"Processed {completed}/{nrows} rows")
    finally:
        pool.close()
        pool.join()
# def vectorize_subset(
#     X_path: Path,
#     y_path: Path,
#     raw_feature_paths: list[Path],
#     extractor: PEFeatureExtractor,
#     nrows: int,
#     label_type: str = "label",
#     label_map: dict = {},
# ) -> None:
#     """
#     Vectorize a subset of data and write it to disk.
#     Prints progress every 20,000 completed rows.
#     """
#     # Create space on disk to write features to
#     X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
#     if label_type == "label" or label_type == "family":
#         y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
#     else:
#         y = np.memmap(
#             y_path,
#             dtype=np.float32,
#             mode="w+",
#             shape=(nrows, len(label_map.keys())),
#         )
#     del X, y

#     # Process rows one at a time in the current process
#     for completed, (irow, raw_features_string) in enumerate(
#         enumerate(raw_feature_iterator(raw_feature_paths)),
#         start=1,
#     ):
#         vectorize_unpack(
#             (irow, raw_features_string, X_path, y_path, extractor, nrows, label_type, label_map)
#         )

#         if completed % 20000 == 0 or completed == nrows:
#             print(f"Processed {completed}/{nrows} rows")



def create_vectorized_features(data_dir: Path | str, label_type: str = "label", class_min: int = 10) -> None:
    """
    Create feature vectors from raw features and write them to disk

    Arguments:
    data_dir - Path to the directory containing the dataset.
    label_type - The type of classification problem.
    class_min - The minimum number of instances of a class in the dataset. Data
                points belonging to a class with fewer than class_min instances
                are ignored.

    Valid label_types:
    label - malicious/benign (binary)
    family - malware family classification (multiclass)
    behavior - malware behavior prediction (multiclass, multi-label)
    file_property - malware file property prediction (multiclass, multi-label)
    packer - malware packer prediction (multiclass, multi-label)
    exploit - malware exploit prediction (multiclass, multi-label)
    group - malware threat group prediction (multiclass, multi-label)
    """
    # Ignore empty tags and self-describing file format tags
    ignore_tags = set(["", "win32", "win64", "elf", "linux", "pdf", "apk", "android"])

    extractor = PEFeatureExtractor()
    data_path: Path = Path(data_dir)

    prep = tqdm.tqdm(total=3, desc="Preparing to vectorize raw features")
    X_train_path = data_path / "X_train.dat"
    y_train_path = data_path / "y_train.dat"
    train_feature_paths = gather_feature_paths(data_path, "train")
    train_nrows = sum([1 for fp in train_feature_paths for _ in fp.open()])
    prep.update(1)

    X_test_path = data_path / "X_test.dat"
    y_test_path = data_path / "y_test.dat"
    test_feature_paths = gather_feature_paths(data_path, "test")
    test_nrows = sum([1 for fp in test_feature_paths for _ in fp.open()])
    prep.update(1)

    # Map string labels/tags to numeric labels
    label_map = {}
    i = 0
    if label_type != "label": # No work needed for the default malicious/benign labels
        train_label_counts = read_label_subset(train_feature_paths, train_nrows, label_type)

        # Remove labels/tags that appear fewer than class_min time
        for l, count in train_label_counts.items():
            if l in ignore_tags:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing training set")
    vectorize_subset(X_train_path, y_train_path, train_feature_paths, extractor, train_nrows, label_type, label_map)

    if label_type != "label": # No work needed for the default malicious/benign labels
        test_label_counts = read_label_subset(test_feature_paths, test_nrows, label_type)

        # Remove labels/tags that appear fewer than class_min time
        for l, count in test_label_counts.items():
            if l in ignore_tags:
                continue
            if label_map.get(l) is not None:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing test set")
    vectorize_subset(X_test_path, y_test_path, test_feature_paths, extractor, test_nrows, label_type, label_map)

    print("Vectorizing challenge set")
    X_test_path = data_path / "X_challenge.dat"
    y_test_path = data_path / "y_challenge.dat"
    raw_feature_paths = gather_feature_paths(data_path, "challenge")
    nrows = sum([1 for fp in raw_feature_paths for _ in fp.open()])
    prep.update(1)
    prep.close()
    vectorize_subset(X_test_path, y_test_path, raw_feature_paths, extractor, nrows)


def read_vectorized_features(data_dir: Path | str, subset: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Read vectorized features into memory mapped numpy arrays
    """
    data_path: Path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    extractor = PEFeatureExtractor()
    ndim: int = extractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="r")
    X = np.array(X).reshape(-1, ndim)
    N: int = X.shape[0]
    y = np.memmap(y_path, dtype=np.int32, mode="r")
    y = np.array(y)
    if y.shape[0] > N:
        y = y.reshape(N, -1)

    return X, y


def read_metadata_record(raw_features_string: str) -> dict:
    """
    Decode a raw features string and return the metadata fields
    """
    all_data = json.loads(raw_features_string)
    metadata_keys = set(ORDERED_COLUMNS)
    return {k: all_data[k] for k in all_data.keys() & metadata_keys}


def read_metadata(data_dir: Path | str) -> pl.DataFrame:
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()
    data_path: Path = Path(data_dir)

    train_feature_paths = gather_feature_paths(data_path, "train")
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
    train_metadf = pl.DataFrame(train_records).with_columns(subset=pl.lit("train")).select(ORDERED_COLUMNS)

    test_feature_paths = gather_feature_paths(data_path, "test")
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
    test_metadf = pl.DataFrame(test_records).with_columns(subset=pl.lit("test")).select(ORDERED_COLUMNS)

    challenge_feature_paths = gather_feature_paths(data_path, "challenge")
    challenge_records = list(pool.imap(read_metadata_record, raw_feature_iterator(challenge_feature_paths)))
    challenge_metadf = pl.DataFrame(challenge_records).with_columns(subset=pl.lit("challenge")).select(ORDERED_COLUMNS)

    return train_metadf, test_metadf, challenge_metadf


def optimize_model(data_dir: Path | str, output_dir: Path | str) -> dict:
    """
    Run an Optuna search to find the best XGBoost parameters
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train")
    train_rows = y_train != -1
    X_train_labeled = X_train[train_rows]
    y_train_labeled = y_train[train_rows]

    # Score by ROC AUC
    # We're interested in low FPR rates, so we'll consider only the AUC for FPRs in [0,5e-3]
    def objective(trial: optuna.Trial) -> float:
        #V1
        # params = {
        #     "n_estimators": trial.suggest_int("n_estimators", 250, 1000, step=250),
        #     "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        #     "max_depth": trial.suggest_int("max_depth", 0, 16, step=2),
        #     "max_leaves": trial.suggest_int("max_leaves", 32, 256, step=32),
        #     "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        #     "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        #     "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0, step=0.1),
        #     "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
        #     "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        #     "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 10.0, log=True),
        #     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        #     "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0, log=True),
        # }
        #V2
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 750, 2500, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 32, step=2),
            "max_leaves": trial.suggest_int("max_leaves", 160, 512, step=32),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise"]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.05),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.7, 1.0, step=0.05),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.00001, 0.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0, log=True),
        }
        #V3 for future work
        # params = {
        #     "n_estimators": trial.suggest_int("n_estimators", 2000, 4000, step=250),
        #     "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.25, log=True),
        #     "max_depth": trial.suggest_int("max_depth", 8, 32, step=2),
        #     "max_leaves": trial.suggest_int("max_leaves", 448, 736, step=32),
        #     "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise"]),
        #     "subsample": trial.suggest_float("subsample", 0.65, 1.0, step=0.05),
        #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0, step=0.05),
        #     "colsample_bynode": trial.suggest_float("colsample_bynode", 0.65, 1.0, step=0.05),
        #     "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 5.0, log=True),
        #     "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        #     "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2, log=True),
        #     "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        #     "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
        # }
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
            random_state=42,
            **params,
        )

        # Each row in X_train appears in chronological order of "first_seen_date" so this works for
        # progressive time series splitting
        progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train_labeled)
        scores = []
        for step, (train_idx, val_idx) in enumerate(progressive_cv):
            X_tr, X_val = X_train_labeled[train_idx], X_train_labeled[val_idx]
            y_tr, y_val = y_train_labeled[train_idx], y_train_labeled[val_idx]

            model.fit(X_tr, y_tr, verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_pred, max_fpr=5e-3))

            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def log_trial(study: optuna.Study, trial: optuna.Trial) -> None:
        print(f"Trial {trial.number} finished with value={trial.value}")

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    try:
        study.optimize(objective, n_trials=30, show_progress_bar=True, callbacks=[log_trial])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    except Exception as exc:
        print(f"Optimization failed with error: {exc}")
    finally:
        optuna_dir = Path(output_dir) / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(by="value", ascending=False)
    trials_df.to_csv(optuna_dir / "trials.csv", index=False)

    with (optuna_dir / "study.pkl").open("wb") as fout:
        pickle.dump(study, fout)

    if len(study.trials) == 0:
        print("No completed trials to summarize.")
        return {}

    with (optuna_dir / "best_params.json").open("w") as fout:
        json.dump(study.best_params, fout, indent=2, sort_keys=True)

    with (optuna_dir / "best_value.txt").open("w") as fout:
        fout.write(f"{study.best_value}\n")

    study_summary = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
    with (optuna_dir / "study_summary.json").open("w") as fout:
        json.dump(study_summary, fout, indent=2, sort_keys=True)

    print(f"Best value={study.best_value}, best_trial={study.best_trial.number}")
    print(f"Best params={study.best_params}")

    return study.best_params


# def train_model(data_dir: Path | str, params: dict = {}) -> xgb.XGBClassifier:
#     """
#     Train XGBoost model on the vectorized features.
#     """
#     # Read data
#     X, y = read_vectorized_features(data_dir, "train")
#     print(f"Training model on {X.shape[0]} samples with {X.shape[1]} features")

#     # Verify that y_train is not formatted for multi-label classification
#     if len(y.shape) != 1:
#         raise ValueError("Encounted y_train with invalid shape. Use train_ovr_model() instead.")

#     # Ignore files without a label/tag
#     num_classes = np.max(y) + 1
#     X = X[y != -1, :]
#     y = y[y != -1]

#     # Use a stratified split to make a validation set
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
#     print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
#     print(f"{num_classes} classes detected in training set")
#     if num_classes == 2:
#         base_params = {
#             "objective": "binary:logistic",
#             "eval_metric": "logloss",
#             "n_jobs": -1,
#             "tree_method": "hist",
#             "device": "cuda",
#             "verbosity": 0,
#         }
#         base_params.update(params)
#         print(base_params)
#         model = xgb.XGBClassifier(**base_params)
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#         return model

#     # Multiclass classification
#     base_params = {
#         "objective": "multi:softprob",
#         "eval_metric": "mlogloss",
#         "num_class": int(num_classes),
#         "n_jobs": -1,
#         "tree_method": "hist",
#         "seed": 42,
#         "device": "cuda",
#         "verbosity": 0,
#     }
#     base_params.update(params)
#     print(base_params)
#     model = xgb.XGBClassifier(**base_params)
#     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#     return model


def train_final_model(data_dir: Path | str, params: dict = {}) -> xgb.XGBClassifier:
    """
    Train a final XGBoost model on the full training set.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")
    print(f"Training final model on {X.shape[0]} samples with {X.shape[1]} features")

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 1:
        raise ValueError("Encounted y_train with invalid shape. Use train_ovr_model() instead.")

    # Ignore files without a label/tag
    num_classes = np.max(y) + 1
    X = X[y != -1, :]
    y = y[y != -1]

    print(f"{num_classes} classes detected in training set")
    if num_classes == 2:
        base_params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eval_metric": ["auc", "logloss", "error"],
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 1,
            "seed": 42,
        }
        base_params.update(params)
        print(base_params)
        model = xgb.XGBClassifier(**base_params)
        model.fit(X, y, verbose=False)
        return model

    # Multiclass classification
    base_params = {
        "objective": "multi:softprob",
        "eval_metric": ["auc", "mlogloss", "error"],
        "num_class": int(num_classes),
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 1,
        "seed": 42,
    }
    base_params.update(params)
    print(base_params)
    model = xgb.XGBClassifier(**base_params)
    model.fit(X, y, verbose=False)
    return model


def train_ovr_model(data_dir: Path | str, params: dict = {}) -> list[xgb.XGBClassifier]:
    """
    Returns a list of One-vs-Rest (OvR) XGBoost classifiers trained on the vectorized features.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 2:
        raise ValueError("Encounted y_train with invalid shape. Use train_ovr_model() instead.")

    # OvR Multilabel classification
    xgb_models = []
    for i in range(y.shape[1]):
        base_params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eval_metric": ["auc", "logloss", "error"],
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 1,
            "seed": 42,
        }
        base_params.update(params)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_i, test_size=0.1, stratify=y_i)
        model = xgb.XGBClassifier(**base_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_models.append(model)
    return xgb_models


def predict_sample(xgb_model: xgb.XGBClassifier, file_data: bytes) -> float:
    """
    Predict a PE file with an XGBoost model
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    predict_proba: np.ndarray = xgb_model.predict_proba([features])
    if predict_proba.ndim == 2 and predict_proba.shape[1] >= 2:
        return float(predict_proba[0, 1])
    return float(predict_proba[0])