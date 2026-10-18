"""Lightweight experiment tracker: JSONL run log, CSV results table, TensorBoard.

Every experiment row written to ``reports/experiment_results.csv`` has a run
ID and timestamp that also appear in ``reports/experiment_log.jsonl`` and in
the TensorBoard event files under ``logs/tensorboard/<run_id>``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .utils import PROJECT_ROOT


class ExperimentTracker:
    def __init__(self, reports_dir: str | Path = PROJECT_ROOT / "reports",
                 log_dir: str | Path = PROJECT_ROOT / "logs", reset: bool = False):
        self.reports_dir = Path(reports_dir)
        self.log_dir = Path(log_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.reports_dir / "experiment_log.jsonl"
        self.csv_path = self.reports_dir / "experiment_results.csv"
        if reset:
            for p in (self.jsonl_path, self.csv_path):
                if p.exists():
                    p.unlink()
        self.records: list[dict] = []

    @staticmethod
    def new_run_id(experiment_id: str) -> str:
        return f"{experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def log_run(self, experiment_id: str, model: str, paradigm: str, config: dict, metrics: dict,
                train_seconds: float, observation: str = "", run_id: str | None = None,
                split: str = "train 70% / val 15% / test 15% (stratified, seed 42)") -> dict:
        run_id = run_id or self.new_run_id(experiment_id)
        record = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "paradigm": paradigm,
            "split": split,
            "train_seconds": round(float(train_seconds), 2),
            "config": config,
            "val_metrics": {k: round(float(v), 5) for k, v in metrics.items()},
            "observation": observation,
        }
        self.records.append(record)
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self._write_tensorboard(record)
        self._write_csv()
        return record

    def annotate(self, observations: dict[str, str]) -> None:
        """Attach post-run observations (written after reading the results) to logged runs.

        Rewrites the JSONL log and CSV table for this session's runs and adds the
        text to each run's TensorBoard directory.
        """
        unknown = set(observations) - {r["experiment_id"] for r in self.records}
        if unknown:
            raise KeyError(f"No logged run for: {sorted(unknown)}")
        for record in self.records:
            if record["experiment_id"] in observations:
                record["observation"] = observations[record["experiment_id"]]
                self._write_observation_tb(record)
        with open(self.jsonl_path, "w") as fh:
            for record in self.records:
                fh.write(json.dumps(record, default=str) + "\n")
        self._write_csv()

    def _write_observation_tb(self, record: dict) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            return
        writer = tf.summary.create_file_writer(str(self.log_dir / "tensorboard" / record["run_id"] / "summary"))
        with writer.as_default():
            tf.summary.text("observation", record["observation"], step=0)
        writer.flush()

    def _write_tensorboard(self, record: dict) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            return
        writer = tf.summary.create_file_writer(str(self.log_dir / "tensorboard" / record["run_id"] / "summary"))
        with writer.as_default():
            for k, v in record["val_metrics"].items():
                tf.summary.scalar(f"val/{k}", v, step=0)
            tf.summary.scalar("train_seconds", record["train_seconds"], step=0)
            tf.summary.text("config", json.dumps(record["config"], default=str), step=0)
        writer.flush()

    def results_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            row = {k: r[k] for k in ["experiment_id", "run_id", "timestamp_utc", "model", "paradigm",
                                      "split", "train_seconds"]}
            row["preprocessing"] = r["config"].get("preprocessing", "")
            row["features"] = r["config"].get("features", "")
            row["imbalance_strategy"] = r["config"].get("imbalance_strategy", "")
            row["hyperparameters"] = json.dumps(r["config"].get("hyperparameters", {}), default=str)
            row.update(r["val_metrics"])
            row["observation"] = r["observation"]
            rows.append(row)
        return pd.DataFrame(rows)

    def _write_csv(self) -> None:
        self.results_frame().to_csv(self.csv_path, index=False)
