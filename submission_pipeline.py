from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC


REQUIRED_COLUMNS = [
    "UserID",
    "Menu",
    "Review",
    "Total",
    "Taste",
    "Quantity",
    "Delivery",
    "Date",
    "HasPicture",
    "BiasFree",
    "RestaurantID",
]

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
WS_RE = re.compile(r"\s+")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    csv_name: str = "Reviews Translated into English.csv"
    project_dir: str = ROOT_DIR
    output_dir: str = os.path.join(ROOT_DIR, "outputs")
    data_dir: str = os.path.join(ROOT_DIR, "data")
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.20
    split_max_tries: int = 40
    ngram_range: Tuple[int, int] = (1, 2)
    hash_n_features: int = 2**18
    srp_dim: int = 128
    text_stop_words: str = "english"
    positive_class_name: str = "BiasFree=1 (positive class)"
    export_plots: bool = True
    zip_outputs: bool = True
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42
    picture_context_folds: int = 5


def safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def latex_escape(text: Any) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def df_to_booktabs(df: pd.DataFrame, caption: str, label: str, floatfmt: str = "%.4f") -> str:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda x: floatfmt % float(x) if np.isfinite(x) else "")
        else:
            table[col] = table[col].astype(str).map(latex_escape)

    headers = [latex_escape(col) for col in table.columns]
    align = "l" * max(1, len(headers))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{latex_escape(label)}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    return WS_RE.sub(" ", text).strip().lower()


def stable_text_cast(series: pd.Series, missing_token: str = "") -> pd.Series:
    return series.where(series.notna(), missing_token).astype(str)


def canonical_text(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return WS_RE.sub(" ", text).strip()


def is_valid_canonical_review(text: Any) -> bool:
    text = str(text).strip().lower()
    return text not in {"", "nan", "none", "null", "na", "n/a"}


def upper_ratio(value: Any) -> float:
    text = str(value) if value is not None else ""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return float(sum(1 for ch in letters if ch.isupper()) / len(letters))


def type_token_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


def iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def resolve_csv_path(cfg: Config) -> str:
    candidates = [
        cfg.csv_name,
        os.path.join(cfg.project_dir, cfg.csv_name),
        os.path.join(cfg.data_dir, cfg.csv_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    patterns = [
        os.path.join(cfg.data_dir, "**", cfg.csv_name),
        os.path.join(cfg.project_dir, "**", cfg.csv_name),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.abspath(matches[0])

    raise FileNotFoundError(f"Could not find {cfg.csv_name}. Checked {candidates}.")


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    validate_schema(df)
    return df


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce", utc=True)
    work = work.dropna(subset=["Date"]).copy()

    work["BiasFree"] = pd.to_numeric(work["BiasFree"], errors="coerce")
    work = work[work["BiasFree"].isin([0, 1])].copy()
    work["BiasFree"] = work["BiasFree"].astype(int)

    work["HasPicture"] = pd.to_numeric(work["HasPicture"], errors="coerce").fillna(0).astype(int)
    work["Review_raw"] = stable_text_cast(work["Review"], missing_token="")
    work["Review_clean"] = work["Review_raw"].map(clean_text)
    work["Review_can"] = work["Review_raw"].map(canonical_text)
    work["Menu_clean"] = stable_text_cast(work["Menu"], missing_token="").map(clean_text)
    work["TextCombined"] = (work["Review_clean"] + " " + work["Menu_clean"]).str.strip()
    work = work[work["Review_can"].map(is_valid_canonical_review)].copy()

    for col in ["UserID", "RestaurantID"]:
        work[col] = stable_text_cast(work[col], missing_token="unknown")

    return work.reset_index(drop=True)


class ContextFeatureBuilder:
    def __init__(self) -> None:
        self.dup_mass_map: Dict[str, float] = {}
        self.dup_breadth_map: Dict[str, float] = {}
        self.user_degree_map: Dict[str, float] = {}
        self.restaurant_degree_map: Dict[str, float] = {}
        self.user_med_iri_map: Dict[str, float] = {}
        self.user_iqr_iri_map: Dict[str, float] = {}
        self.restaurant_day_count_map: Dict[Tuple[str, str], float] = {}
        self.restaurant_day_mean_map: Dict[str, float] = {}
        self.restaurant_day_std_map: Dict[str, float] = {}
        self.multi_poster_users: set[str] = set()
        self.min_date: pd.Timestamp | None = None

    @staticmethod
    def _compute_multi_poster_users(train_df: pd.DataFrame) -> set[str]:
        flagged: set[str] = set()
        ordered = train_df.sort_values(["UserID", "Date"], kind="mergesort")
        for uid, group in ordered.groupby("UserID", sort=False):
            if len(group) < 3:
                continue
            ts = group["Date"].values.astype("datetime64[ns]")
            rests = group["RestaurantID"].astype(str).values
            left = 0
            counts: Dict[str, int] = {}
            distinct = 0

            for right in range(len(group)):
                rest = rests[right]
                counts[rest] = counts.get(rest, 0) + 1
                if counts[rest] == 1:
                    distinct += 1

                while ts[right] - ts[left] > np.timedelta64(24, "h"):
                    old = rests[left]
                    counts[old] -= 1
                    if counts[old] == 0:
                        distinct -= 1
                    left += 1

                if (right - left + 1) >= 3 and distinct >= 2:
                    flagged.add(str(uid))
                    break

        return flagged

    def fit(self, train_df: pd.DataFrame) -> None:
        train_df = train_df.copy()
        self.min_date = train_df["Date"].min()

        grp = train_df.groupby("Review_can", sort=False)
        self.dup_mass_map = grp.size().astype(float).to_dict()
        self.dup_breadth_map = grp["RestaurantID"].nunique().astype(float).to_dict()
        self.user_degree_map = train_df.groupby("UserID").size().astype(float).to_dict()
        self.restaurant_degree_map = train_df.groupby("RestaurantID").size().astype(float).to_dict()

        ordered = train_df.sort_values(["UserID", "Date"], kind="mergesort").copy()
        ordered["prev_date"] = ordered.groupby("UserID")["Date"].shift(1)
        ordered["iri_hours"] = (ordered["Date"] - ordered["prev_date"]).dt.total_seconds() / 3600.0

        self.user_med_iri_map = (
            ordered.groupby("UserID")["iri_hours"].median().fillna(0.0).astype(float).to_dict()
        )
        self.user_iqr_iri_map = (
            ordered.groupby("UserID")["iri_hours"]
            .apply(
                lambda s: float(
                    np.nanpercentile(s.values.astype(float), 75)
                    - np.nanpercentile(s.values.astype(float), 25)
                )
                if np.isfinite(s.values.astype(float)).any()
                else 0.0
            )
            .fillna(0.0)
            .astype(float)
            .to_dict()
        )

        ordered["day_key"] = ordered["Date"].dt.floor("D").astype(str)
        daily = ordered.groupby(["RestaurantID", "day_key"]).size().rename("restaurant_day_cnt").reset_index()
        for _, row in daily.iterrows():
            key = (str(row["RestaurantID"]), str(row["day_key"]))
            self.restaurant_day_count_map[key] = float(row["restaurant_day_cnt"])

        stats = daily.groupby("RestaurantID")["restaurant_day_cnt"].agg(["mean", "std"])
        self.restaurant_day_mean_map = stats["mean"].fillna(0.0).astype(float).to_dict()
        self.restaurant_day_std_map = stats["std"].fillna(0.0).astype(float).to_dict()
        self.multi_poster_users = self._compute_multi_poster_users(ordered.dropna(subset=["Date"]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        tokens = work["Review_clean"].str.split()

        work["wc"] = tokens.map(lambda seq: len(seq) if isinstance(seq, list) else 0).astype(float)
        work["ttr"] = tokens.map(lambda seq: type_token_ratio(seq if isinstance(seq, list) else [])).astype(float)
        work["char_len"] = work["Review_can"].map(lambda text: float(len(text)))
        work["exc"] = work["Review_raw"].map(lambda text: float(str(text).count("!")))
        work["qst"] = work["Review_raw"].map(lambda text: float(str(text).count("?")))
        work["upper_ratio"] = work["Review_raw"].map(upper_ratio).astype(float)
        work["day_of_week"] = work["Date"].dt.dayofweek.astype(float)
        work["hour"] = work["Date"].dt.hour.astype(float)
        work["month"] = work["Date"].dt.month.astype(float)

        base_date = self.min_date
        if base_date is None or pd.isna(base_date):
            base_date = work["Date"].min()
        work["days_from_min"] = ((work["Date"] - base_date).dt.total_seconds() / 86400.0).astype(float)

        work["dup_mass"] = work["Review_can"].map(self.dup_mass_map).fillna(1.0).astype(float)
        work["dup_breadth"] = work["Review_can"].map(self.dup_breadth_map).fillna(1.0).astype(float)
        work["user_degree"] = work["UserID"].map(self.user_degree_map).fillna(0.0).astype(float)
        work["restaurant_degree"] = work["RestaurantID"].map(self.restaurant_degree_map).fillna(0.0).astype(float)
        work["user_med_iri_hours"] = work["UserID"].map(self.user_med_iri_map).fillna(0.0).astype(float)
        work["user_iqr_iri_hours"] = work["UserID"].map(self.user_iqr_iri_map).fillna(0.0).astype(float)
        work["multi_poster"] = work["UserID"].astype(str).isin(self.multi_poster_users).astype(float)
        work["day_key"] = work["Date"].dt.floor("D").astype(str)

        def _burst(rest_id: str, day_key: str) -> float:
            cnt = self.restaurant_day_count_map.get((str(rest_id), str(day_key)), 0.0)
            mu = self.restaurant_day_mean_map.get(str(rest_id), 0.0)
            sd = self.restaurant_day_std_map.get(str(rest_id), 0.0)
            if not np.isfinite(sd) or sd <= 0.0:
                return 0.0
            return float((cnt - mu) / sd)

        work["burst_z"] = [
            _burst(rest_id, day_key)
            for rest_id, day_key in zip(work["RestaurantID"].astype(str), work["day_key"].astype(str))
        ]
        work["burst_z"] = pd.to_numeric(work["burst_z"], errors="coerce").fillna(0.0).astype(float)
        return work


def build_text_matrix(text: pd.Series, train_idx: np.ndarray, cfg: Config) -> np.ndarray:
    hv = HashingVectorizer(
        n_features=cfg.hash_n_features,
        ngram_range=cfg.ngram_range,
        stop_words=cfg.text_stop_words,
        alternate_sign=False,
        norm=None,
    )
    hashed = hv.transform(text.fillna("").astype(str).tolist())
    srp = SparseRandomProjection(n_components=cfg.srp_dim, dense_output=True, random_state=cfg.seed)
    srp.fit(hashed[train_idx])
    return srp.transform(hashed)


def primary_model(cfg: Config) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def linear_svm_model(cfg: Config) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    dual=False,
                    max_iter=5000,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def picture_context_model(cfg: Config) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, mode: str) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if mode == "roc":
        return float(roc_auc_score(y_true, y_score))
    return float(average_precision_score(y_true, y_score))


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    if threshold.size == 0:
        return 0.5
    f1s = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(threshold[int(np.nanargmax(f1s))])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "roc_auc": safe_auc(y_true, y_score, "roc"),
        "pr_auc": safe_auc(y_true, y_score, "pr"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    metrics.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    return metrics


def bootstrap_metric_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_resamples: int,
    seed: int,
    cluster_ids: np.ndarray | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr_auc_vals: List[float] = []
    precision_vals: List[float] = []
    recall_vals: List[float] = []
    f1_vals: List[float] = []

    unique_clusters = None
    cluster_to_indices = None
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids).astype(str)
        if len(cluster_ids) != n:
            raise ValueError("cluster_ids must have the same length as y_true.")
        unique_clusters = np.unique(cluster_ids)
        cluster_to_indices = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique_clusters}

    for _ in range(int(n_resamples)):
        if unique_clusters is None:
            sample_idx = rng.integers(0, n, n)
        else:
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            sample_idx = np.concatenate([cluster_to_indices[cluster] for cluster in sampled_clusters])

        y_boot = y_true[sample_idx]
        if len(np.unique(y_boot)) < 2:
            continue

        score_boot = y_score[sample_idx]
        boot_metrics = compute_metrics(y_boot, score_boot, threshold)
        pr_auc_vals.append(safe_auc(y_boot, score_boot, "pr"))
        precision_vals.append(boot_metrics["precision"])
        recall_vals.append(boot_metrics["recall"])
        f1_vals.append(boot_metrics["f1"])

    def _bounds(values: List[float], alpha: float = 0.025) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))

    pr_low, pr_high = _bounds(pr_auc_vals)
    p_low, p_high = _bounds(precision_vals)
    r_low, r_high = _bounds(recall_vals)
    f1_low, f1_high = _bounds(f1_vals)

    return {
        "bootstrap_n": float(len(pr_auc_vals)),
        "bootstrap_n_clusters": float(len(unique_clusters)) if unique_clusters is not None else float("nan"),
        "pr_auc_ci_low": pr_low,
        "pr_auc_ci_high": pr_high,
        "precision_ci_low": p_low,
        "precision_ci_high": p_high,
        "recall_ci_low": r_low,
        "recall_ci_high": r_high,
        "f1_ci_low": f1_low,
        "f1_ci_high": f1_high,
    }


def get_model_scores(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(matrix), dtype=float).reshape(-1)
    raise AttributeError("Model must expose either predict_proba or decision_function.")


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_dir: str, tag: str) -> None:
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {tag}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_{tag}.png"), dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {tag}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{tag}.png"), dpi=160)
    plt.close()


def split_baseline(y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    for offset in range(cfg.split_max_tries):
        seed = cfg.seed + offset
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=cfg.test_size, random_state=seed, stratify=y
        )
        y_train_val = y[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=cfg.val_size, random_state=seed, stratify=y_train_val
        )
        if len(np.unique(y[train_idx])) == len(np.unique(y[val_idx])) == len(np.unique(y[test_idx])) == 2:
            return train_idx, val_idx, test_idx
    raise RuntimeError("Could not create a valid baseline split.")


def split_group(groups: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    for offset in range(cfg.split_max_tries):
        seed = cfg.seed + offset
        gss_test = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(all_idx, y, groups))

        groups_train_val = groups[train_val_idx]
        y_train_val = y[train_val_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=seed)
        train_rel, val_rel = next(gss_val.split(np.arange(len(train_val_idx)), y_train_val, groups_train_val))

        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]

        if len(np.unique(y[train_idx])) == len(np.unique(y[val_idx])) == len(np.unique(y[test_idx])) == 2:
            return train_idx, val_idx, test_idx

    raise RuntimeError("Could not create a valid group split.")


def leakage_check(review_can: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, int]:
    tr = set(review_can.iloc[train_idx].astype(str))
    va = set(review_can.iloc[val_idx].astype(str))
    te = set(review_can.iloc[test_idx].astype(str))
    return {
        "overlap_train_val": len(tr & va),
        "overlap_train_test": len(tr & te),
        "overlap_val_test": len(va & te),
    }


def split_distribution(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name, idx in {"train": train_idx, "validation": val_idx, "test": test_idx}.items():
        yy = y[idx]
        report[name] = {
            "n": int(len(idx)),
            "positive_rate": float(np.mean(yy)),
            "n_positive": int(np.sum(yy == 1)),
            "n_negative": int(np.sum(yy == 0)),
        }
    return report


def learn_picture_context(
    work: pd.DataFrame,
    train_idx: np.ndarray,
    out_dir: str,
    protocol_name: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = work.copy()
    pic_features = [
        "HasPicture",
        "dup_mass",
        "dup_breadth",
        "burst_z",
        "user_degree",
        "restaurant_degree",
        "user_med_iri_hours",
        "user_iqr_iri_hours",
        "multi_poster",
        "wc",
        "ttr",
        "char_len",
    ]

    X_all = work[pic_features].copy()
    X_train = X_all.iloc[train_idx].copy()
    y_train = work.iloc[train_idx]["BiasFree"].values.astype(int)

    oof_scores = np.full(len(train_idx), np.nan, dtype=float)
    class_counts = pd.Series(y_train).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    n_splits = min(cfg.picture_context_folds, min_class_count)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        for inner_train_rel, inner_val_rel in skf.split(X_train, y_train):
            fold_model = picture_context_model(cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_model.fit(X_train.iloc[inner_train_rel], y_train[inner_train_rel])
            oof_scores[inner_val_rel] = fold_model.predict_proba(X_train.iloc[inner_val_rel])[:, 1]
    else:
        oof_scores[:] = float(np.mean(y_train)) if len(y_train) else 0.5

    model = picture_context_model(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    work["pic_context_score"] = model.predict_proba(X_all)[:, 1]
    missing_oof = ~np.isfinite(oof_scores)
    if missing_oof.any():
        oof_scores[missing_oof] = model.predict_proba(X_train.iloc[missing_oof])[:, 1]
    work.loc[work.index[train_idx], "pic_context_score"] = oof_scores

    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    imputer = model.named_steps["imputer"]

    coef_df = pd.DataFrame(
        {
            "feature": pic_features,
            "imputer_median": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "logit_coef": clf.coef_[0],
            "odds_ratio": np.exp(clf.coef_[0]),
        }
    ).sort_values("logit_coef", ascending=False)

    coef_df.to_csv(os.path.join(out_dir, "picture_context_coefficients.csv"), index=False)
    write_text(
        os.path.join(out_dir, "picture_context_coefficients.tex"),
        df_to_booktabs(
            coef_df.reset_index(drop=True),
            caption=(
                "Training-only coefficients for the logistic submodel that learns the Picture-Context "
                "score from the raw picture indicator and coordination features."
            ),
            label=f"tab:pic_context_coef_{protocol_name}",
        ),
    )

    plt.figure(figsize=(8, max(4, 0.30 * len(coef_df))))
    plot_df = coef_df.sort_values("logit_coef")
    plt.barh(plot_df["feature"], plot_df["logit_coef"])
    plt.xlabel("Logit coefficient")
    plt.title(f"Picture-Context coefficients - {protocol_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "picture_context_coefficients.png"), dpi=160)
    plt.close()

    report = {
        "train_only": True,
        "train_scores": "out_of_sample",
        "n_oof_splits": int(n_splits),
        "intercept": float(clf.intercept_[0]),
        "features": pic_features,
        "n_features": len(pic_features),
    }
    return work, report


def prepare_feature_sets(work: pd.DataFrame) -> Dict[str, np.ndarray]:
    behavior_cols = [
        "char_len",
        "wc",
        "ttr",
        "exc",
        "qst",
        "upper_ratio",
        "day_of_week",
        "hour",
        "month",
        "days_from_min",
        "dup_mass",
        "dup_breadth",
        "user_degree",
        "restaurant_degree",
        "user_med_iri_hours",
        "user_iqr_iri_hours",
        "burst_z",
        "multi_poster",
    ]

    behavior = work[behavior_cols].astype(float).values
    text = work[[col for col in work.columns if col.startswith("text_srp_")]].astype(float).values
    has_picture = work[["HasPicture"]].astype(float).values
    pic_context = work[["pic_context_score"]].astype(float).values

    return {
        "text_only": text,
        "behavior_only": behavior,
        "combined_no_haspicture": np.hstack([text, behavior]),
        "combined_raw_haspicture": np.hstack([text, behavior, has_picture]),
        "combined_pic_context": np.hstack([text, behavior, pic_context]),
        "picture_only_raw": has_picture,
        "picture_only_context": pic_context,
    }


def evaluate_matrix(
    matrix: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    model: Pipeline | None = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    model = clone(model if model is not None else primary_model(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(matrix[train_idx], y[train_idx])

    val_score = get_model_scores(model, matrix[val_idx])
    threshold = find_best_f1_threshold(y[val_idx], val_score)

    test_score = get_model_scores(model, matrix[test_idx])
    metrics = compute_metrics(y[test_idx], test_score, threshold)
    metrics["threshold"] = float(threshold)
    return metrics, test_score


def run_protocol(work0: pd.DataFrame, cfg: Config, protocol_name: str) -> pd.DataFrame:
    out_dir = os.path.join(cfg.output_dir, protocol_name)
    safe_mkdir(out_dir)

    y = work0["BiasFree"].values.astype(int)
    if protocol_name == "baseline":
        train_idx, val_idx, test_idx = split_baseline(y, cfg)
    else:
        train_idx, val_idx, test_idx = split_group(work0["Review_can"].values, y, cfg)

    leaks = leakage_check(work0["Review_can"], train_idx, val_idx, test_idx)
    write_json(os.path.join(out_dir, "leakage_check.json"), leaks)
    write_json(os.path.join(out_dir, "split_distribution.json"), split_distribution(y, train_idx, val_idx, test_idx))

    builder = ContextFeatureBuilder()
    builder.fit(work0.iloc[train_idx].copy())
    work = builder.transform(work0.copy())

    text_matrix = build_text_matrix(work["TextCombined"], train_idx, cfg)
    text_cols = [f"text_srp_{dim}" for dim in range(text_matrix.shape[1])]
    text_df = pd.DataFrame(text_matrix, columns=text_cols, index=work.index)
    work = pd.concat([work, text_df], axis=1)

    work, pic_report = learn_picture_context(work, train_idx, out_dir, protocol_name, cfg)
    feature_sets = prepare_feature_sets(work)

    rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    scored_export_sets = {"combined_no_haspicture", "combined_raw_haspicture", "combined_pic_context"}

    for name, matrix in feature_sets.items():
        metrics, test_score = evaluate_matrix(matrix, y, train_idx, val_idx, test_idx, cfg)
        rows.append({"protocol": protocol_name, "feature_set": name, "model": "logreg", **metrics})

        if name in scored_export_sets:
            scored = work.iloc[test_idx].copy()
            scored["protocol"] = protocol_name
            scored["feature_set"] = name
            scored["score_biasfree"] = test_score
            scored["thr_val"] = float(metrics["threshold"])
            scored["pred_biasfree"] = (scored["score_biasfree"] >= scored["thr_val"]).astype(int)
            scored.to_csv(os.path.join(out_dir, f"test_scored_{name}.csv"), index=False)

            cluster_ids = work.iloc[test_idx]["Review_can"].astype(str).values if protocol_name == "group" else None
            ci = bootstrap_metric_interval(
                y_true=y[test_idx],
                y_score=test_score,
                threshold=float(metrics["threshold"]),
                n_resamples=cfg.bootstrap_resamples,
                seed=cfg.bootstrap_seed,
                cluster_ids=cluster_ids,
            )
            bootstrap_rows.append(
                {
                    "protocol": protocol_name,
                    "feature_set": name,
                    "bootstrap_unit": "review_text_cluster" if cluster_ids is not None else "row",
                    "pr_auc": float(metrics["pr_auc"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    **ci,
                }
            )

        if cfg.export_plots and name in scored_export_sets:
            plot_roc_pr(y[test_idx], test_score, out_dir, f"{protocol_name}_{name}")

    results = pd.DataFrame(rows).sort_values(["protocol", "pr_auc"], ascending=[True, False]).reset_index(drop=True)
    results.to_csv(os.path.join(out_dir, "experiment_results.csv"), index=False)
    write_text(
        os.path.join(out_dir, "experiment_results.tex"),
        df_to_booktabs(
            results,
            caption=(
                f"Same-model experiment matrix for the {protocol_name} protocol. "
                f"The positive class is {cfg.positive_class_name}."
            ),
            label=f"tab:{protocol_name}_experiment_matrix",
        ),
    )

    pic_compare = results[
        results["feature_set"].isin(
            ["picture_only_raw", "picture_only_context", "combined_raw_haspicture", "combined_pic_context"]
        )
    ].copy()
    pic_compare.to_csv(os.path.join(out_dir, "haspicture_vs_piccontext.csv"), index=False)
    write_text(
        os.path.join(out_dir, "haspicture_vs_piccontext.tex"),
        df_to_booktabs(
            pic_compare,
            caption=(
                f"Direct comparison of raw HasPicture and the learned Picture-Context "
                f"signal under the {protocol_name} protocol."
            ),
            label=f"tab:{protocol_name}_pic_signal_compare",
        ),
    )

    ablation = results[
        results["feature_set"].isin(["combined_no_haspicture", "combined_raw_haspicture", "combined_pic_context"])
    ].copy()
    ablation.to_csv(os.path.join(out_dir, "haspicture_ablation.csv"), index=False)
    write_text(
        os.path.join(out_dir, "haspicture_ablation.tex"),
        df_to_booktabs(
            ablation,
            caption=(
                f"HasPicture ablation for the {protocol_name} protocol, comparing no picture feature, "
                f"raw HasPicture, and the learned Picture-Context score."
            ),
            label=f"tab:{protocol_name}_haspicture_ablation",
        ),
    )

    text_behavior = results[
        results["feature_set"].isin(["text_only", "behavior_only", "combined_no_haspicture"])
    ].copy()
    text_behavior.to_csv(os.path.join(out_dir, "text_behavior_combined.csv"), index=False)
    write_text(
        os.path.join(out_dir, "text_behavior_combined.tex"),
        df_to_booktabs(
            text_behavior,
            caption=(
                f"Controlled comparison between text-only, behavior-only, and combined feature families "
                f"under the {protocol_name} protocol."
            ),
            label=f"tab:{protocol_name}_text_behavior_combined",
        ),
    )

    robustness_feature_sets = ["combined_no_haspicture", "combined_raw_haspicture", "combined_pic_context"]
    robustness_models = {
        "logreg": primary_model(cfg),
        "linear_svm": linear_svm_model(cfg),
    }
    robustness_rows: List[Dict[str, Any]] = []

    for model_name, model_obj in robustness_models.items():
        for feature_name in robustness_feature_sets:
            metrics, _ = evaluate_matrix(
                feature_sets[feature_name],
                y,
                train_idx,
                val_idx,
                test_idx,
                cfg,
                model=model_obj,
            )
            robustness_rows.append(
                {
                    "protocol": protocol_name,
                    "classifier": model_name,
                    "feature_set": feature_name,
                    **metrics,
                }
            )

    robustness = (
        pd.DataFrame(robustness_rows)
        .sort_values(["protocol", "classifier", "feature_set"], kind="mergesort")
        .reset_index(drop=True)
    )
    robustness.to_csv(os.path.join(out_dir, "classifier_robustness.csv"), index=False)
    write_text(
        os.path.join(out_dir, "classifier_robustness.tex"),
        df_to_booktabs(
            robustness,
            caption=(
                f"Classifier-robustness check for the {protocol_name} protocol. "
                f"The same qualitative conclusions are tested under logistic regression and a linear SVM."
            ),
            label=f"tab:{protocol_name}_classifier_robustness",
        ),
    )

    bootstrap_df = (
        pd.DataFrame(bootstrap_rows)
        .sort_values(["protocol", "feature_set"], kind="mergesort")
        .reset_index(drop=True)
    )
    bootstrap_df.to_csv(os.path.join(out_dir, "bootstrap_ci.csv"), index=False)
    write_text(
        os.path.join(out_dir, "bootstrap_ci.tex"),
        df_to_booktabs(
            bootstrap_df,
            caption=(
                f"Percentile-bootstrap 95\\% confidence intervals for key held-out test metrics "
                f"under the {protocol_name} protocol."
            ),
            label=f"tab:{protocol_name}_bootstrap_ci",
        ),
    )

    summary = {
        "protocol": protocol_name,
        "n_rows_used": int(len(work)),
        "positive_rate": float(y.mean()),
        "leakage_check": leaks,
        "split_distribution": split_distribution(y, train_idx, val_idx, test_idx),
        "picture_context_report": pic_report,
    }
    write_json(os.path.join(out_dir, "run_summary.json"), summary)
    return results


def write_project_notes(cfg: Config, csv_path: str, df: pd.DataFrame) -> None:
    data_notes = f"""# Data Notes

Project CSV: `{os.path.basename(csv_path)}`

## Source article
- Hyunmin Lee, SeungYoung Oh, JinHyun Han, Hyunggu Jung.
- *Creating a bias-free dataset with food delivery app reviews under data poisoning attack*.
- Data in Brief, Volume 55, August 2024, 110598.
- DOI: https://doi.org/10.1016/j.dib.2024.110598

## Source dataset
- Mendeley Data: *Bias-Free Dataset of Food Delivery App Reviews with Data Poisoning Attacks*
- DOI: https://doi.org/10.17632/rnyrpzyw3h.2

## Dataset facts used in this project
- Rows in current CSV after cleaning: {len(df):,}
- Distinct restaurants: {df["RestaurantID"].nunique():,}
- Distinct users: {df["UserID"].nunique():,}
- Columns: {", ".join(df.columns.tolist())}
- Positive class convention in code: `{cfg.positive_class_name}`

## Submission-facing implication
- `BiasFree=1` is treated as the positive class.
- `HasPicture` is evaluated both as a raw indicator and as a learned Picture-Context score.
- Duplicate-aware group splitting is defined on canonicalized review text to control template leakage.
"""
    write_text(os.path.join(cfg.project_dir, "data_notes.md"), data_notes)

    method_notes = """# Paper Snippets

## Positive class convention
We use a single positive-class convention throughout all experiments: BiasFree=1 (legitimate). Accordingly, precision, recall, F1, ROC-AUC, and PR-AUC are always computed with respect to the same positive class.

## Learning the Picture-Context Score
The Picture-Context Score is not hand-tuned. For each outer split, we fit a separate logistic submodel using the raw picture indicator and coordination features such as duplication mass, duplication breadth, burstiness, user cadence, and multi-poster behavior. Training rows receive out-of-sample Picture-Context predictions, while validation and test rows receive predictions from the final submodel fitted only on the training partition.

## Same-model split comparison
To isolate evaluation leakage from model choice, every feature family and every split protocol is evaluated with the same downstream classifier. This ensures that any baseline-vs-group performance gap is attributable to the split regime rather than to selecting different models per protocol.
"""
    write_text(os.path.join(cfg.project_dir, "paper_snippets.md"), method_notes)


def zip_outputs(cfg: Config) -> str:
    archive_path = os.path.join(cfg.output_dir, "submission_outputs.zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(cfg.output_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if full_path == archive_path:
                    continue
                zf.write(full_path, os.path.relpath(full_path, cfg.output_dir))
    return archive_path


def run_all(config_updates: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = Config(**(config_updates or {}))
    safe_mkdir(cfg.project_dir)
    safe_mkdir(cfg.output_dir)

    csv_path = resolve_csv_path(cfg)
    df = load_dataset(csv_path)
    work0 = build_base_frame(df)
    write_project_notes(cfg, csv_path, work0)

    schema_report = {
        "csv_path": csv_path,
        "required_columns": REQUIRED_COLUMNS,
        "n_rows_after_cleaning": int(len(work0)),
        "positive_rate": float(work0["BiasFree"].mean()),
        "n_users": int(work0["UserID"].nunique()),
        "n_restaurants": int(work0["RestaurantID"].nunique()),
        "has_picture_rate": float(work0["HasPicture"].mean()),
    }
    write_json(os.path.join(cfg.project_dir, "schema_report.json"), schema_report)
    write_json(os.path.join(cfg.project_dir, "config_used.json"), asdict(cfg))

    baseline = run_protocol(work0, cfg, "baseline")
    group = run_protocol(work0, cfg, "group")
    all_results = pd.concat([baseline, group], ignore_index=True)
    all_results.to_csv(os.path.join(cfg.output_dir, "all_experiment_results.csv"), index=False)

    tables_dir = os.path.join(cfg.output_dir, "tables")
    safe_mkdir(tables_dir)

    counts = pd.crosstab(work0["HasPicture"].astype(int), work0["BiasFree"].astype(int))
    counts = counts.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    row_pct = counts.div(row_totals, axis=0).fillna(0.0) * 100.0

    picture_label_table = pd.DataFrame(
        [
            {
                "HasPicture": "No picture",
                "BiasFree=0 n": int(counts.loc[0, 0]),
                "BiasFree=0 row %": float(row_pct.loc[0, 0]),
                "BiasFree=1 n": int(counts.loc[0, 1]),
                "BiasFree=1 row %": float(row_pct.loc[0, 1]),
            },
            {
                "HasPicture": "Picture present",
                "BiasFree=0 n": int(counts.loc[1, 0]),
                "BiasFree=0 row %": float(row_pct.loc[1, 0]),
                "BiasFree=1 n": int(counts.loc[1, 1]),
                "BiasFree=1 row %": float(row_pct.loc[1, 1]),
            },
        ]
    )
    picture_label_table.to_csv(os.path.join(tables_dir, "biasfree_haspicture_crosstab.csv"), index=False)
    write_text(
        os.path.join(tables_dir, "biasfree_haspicture_crosstab.tex"),
        df_to_booktabs(
            picture_label_table,
            caption=(
                "Cross-tabulation of HasPicture and BiasFree in the cleaned analytical sample. "
                "Percentages are row percentages within each picture-status group."
            ),
            label="tab:biasfree_haspicture",
            floatfmt="%.2f",
        ),
    )

    split_effect = all_results[all_results["feature_set"] == "combined_pic_context"].copy()
    split_effect.to_csv(os.path.join(tables_dir, "same_model_split_effect.csv"), index=False)
    write_text(
        os.path.join(tables_dir, "same_model_split_effect.tex"),
        df_to_booktabs(
            split_effect,
            caption=(
                "Same-model split comparison using the combined Picture-Context feature set. "
                "This isolates the effect of duplicate-aware evaluation from model choice."
            ),
            label="tab:same_model_split_effect",
        ),
    )

    classifier_robustness = pd.concat(
        [
            pd.read_csv(os.path.join(cfg.output_dir, "baseline", "classifier_robustness.csv")),
            pd.read_csv(os.path.join(cfg.output_dir, "group", "classifier_robustness.csv")),
        ],
        ignore_index=True,
    )
    classifier_robustness.to_csv(os.path.join(tables_dir, "classifier_robustness.csv"), index=False)
    write_text(
        os.path.join(tables_dir, "classifier_robustness.tex"),
        df_to_booktabs(
            classifier_robustness,
            caption=(
                "Classifier-robustness check across the two split protocols. "
                "The same qualitative conclusions are tested under logistic regression and a linear SVM "
                "for the no-picture, raw-picture, and picture-context combined feature sets."
            ),
            label="tab:classifier_robustness",
        ),
    )

    bootstrap_ci = pd.concat(
        [
            pd.read_csv(os.path.join(cfg.output_dir, "baseline", "bootstrap_ci.csv")),
            pd.read_csv(os.path.join(cfg.output_dir, "group", "bootstrap_ci.csv")),
        ],
        ignore_index=True,
    )
    bootstrap_ci.to_csv(os.path.join(tables_dir, "bootstrap_ci.csv"), index=False)
    write_text(
        os.path.join(tables_dir, "bootstrap_ci.tex"),
        df_to_booktabs(
            bootstrap_ci,
            caption="Percentile-bootstrap 95\\% confidence intervals for the key held-out comparisons.",
            label="tab:bootstrap_ci",
        ),
    )

    archive_path = zip_outputs(cfg) if cfg.zip_outputs else ""
    return {
        "csv_path": csv_path,
        "output_dir": cfg.output_dir,
        "archive_path": archive_path,
        "schema_report": schema_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-aware Yogiyo review-event benchmark audit.")
    parser.add_argument("--project-dir", default=ROOT_DIR, help="Directory for notes and config exports.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "outputs"),
        help="Directory for tables, figures, and metrics.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(ROOT_DIR, "data"),
        help="Directory that contains the input CSV.",
    )
    parser.add_argument("--csv-name", default="Reviews Translated into English.csv", help="Input CSV filename.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--picture-context-folds",
        type=int,
        default=5,
        help="Number of folds used to produce training Picture-Context scores.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Skip creation of submission_outputs.zip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_all(
        {
            "project_dir": args.project_dir,
            "output_dir": args.output_dir,
            "data_dir": args.data_dir,
            "csv_name": args.csv_name,
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "picture_context_folds": args.picture_context_folds,
            "zip_outputs": not args.no_zip,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
