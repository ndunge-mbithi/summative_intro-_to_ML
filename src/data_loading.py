"""Download, load and verify the CDC BRFSS 2015 public-use data file.

Official source: https://www.cdc.gov/brfss/annual_data/annual_2015.html
The SAS transport file (LLCP2015.XPT) is read in chunks, reduced to the
variables this project needs, verified against the published 2015 codebook,
and cached locally as Parquet. Raw data are never committed to Git.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

CDC_LANDING_PAGE = "https://www.cdc.gov/brfss/annual_data/annual_2015.html"
CDC_XPT_ZIP_URL = "https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip"
CDC_CODEBOOK_URL = "https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
CACHE_PATH = INTERIM_DIR / "brfss2015_subset.parquet"

# Published values from the 2015 BRFSS codebook (LLCP, released 23 Aug 2016).
EXPECTED_N_RECORDS = 441_456
EXPECTED_N_VARIABLES = 330
EXPECTED_DIABETE3 = {1.0: 57_256, 2.0: 3_608, 3.0: 372_104, 4.0: 7_690,
                     7.0: 598, 9.0: 193, "missing": 7}
# SHA-256 of the LLCP2015.XPT copy used for the reported experiments.
REFERENCE_SHA256 = "bfe9e62977cfc5183e51c3e8bdb5193510995cc3c21b225e568f537ad300b1b9"

# Raw variables retained: identifiers / design, target, candidate predictors,
# and attributes used only for subgroup evaluation (never as model inputs).
RAW_COLUMNS = [
    "SEQNO", "_STATE", "_PSU", "_STSTR", "_LLCPWT", "DISPCODE",
    "DIABETE3",
    "_RFHYPE5", "TOLDHI2", "BLOODCHO", "_CHOLCHK", "_BMI5", "_SMOKER3",
    "CVDSTRK3", "_MICHD", "_TOTINDA", "_FRTLT1", "_VEGLT1", "_RFDRHV5",
    "HLTHPLN1", "MEDCOST", "GENHLTH", "MENTHLTH", "PHYSHLTH", "DIFFWALK",
    "SEX", "_AGEG5YR", "_AGE80", "EDUCA", "INCOME2", "CHCKIDNY", "HAVARTH3",
    "ADDEPEV2", "CHECKUP1", "_RACEGR3",
]


@dataclass
class VerificationReport:
    """Result of comparing the loaded file with the published codebook."""

    n_records: int
    n_variables: int
    diabete3_counts: dict
    records_match: bool
    variables_match: bool
    target_counts_match: bool
    sha256: str | None = None
    sha256_matches_reference: bool | None = None
    source_file: str | None = None
    notes: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.records_match and self.variables_match and self.target_counts_match

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 22) -> str:
    """Return the SHA-256 digest of a (potentially very large) file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download_brfss_2015(dest_dir: str | os.PathLike = RAW_DIR, retries: int = 3,
                        timeout: int = 120) -> Path:
    """Download and extract the official CDC 2015 BRFSS SAS transport file.

    Returns the path to the extracted ``.XPT`` file. The CDC archive stores the
    file with a trailing space in its name ("LLCP2015.XPT "), so the extracted
    file is normalised to ``LLCP2015.XPT``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "LLCP2015XPT.zip"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d)", CDC_XPT_ZIP_URL, attempt)
            req = urllib.request.Request(CDC_XPT_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(zip_path, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            break
        except Exception as exc:  # network errors, HTTP errors, timeouts
            last_error = exc
            logger.warning("Download failed: %s", exc)
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(
            f"Could not download BRFSS 2015 from {CDC_XPT_ZIP_URL}: {last_error}. "
            f"Download it manually from {CDC_LANDING_PAGE} and place LLCP2015.XPT in {dest_dir}."
        )
    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = next(m for m in zf.namelist() if m.strip().upper().endswith(".XPT"))
            zf.extract(member, dest_dir)
    except (zipfile.BadZipFile, StopIteration) as exc:
        raise RuntimeError(f"Downloaded archive {zip_path} is not a valid BRFSS XPT zip") from exc
    extracted = dest_dir / member
    target = dest_dir / "LLCP2015.XPT"
    if extracted != target:
        extracted.replace(target)
    return target


def find_raw_file(search_dirs: list[str | os.PathLike] | None = None) -> Path | None:
    """Locate LLCP2015.XPT in the usual places (repo data dir, Colab Drive mount)."""
    search_dirs = search_dirs or [RAW_DIR, Path.cwd(), "/content", "/content/drive/MyDrive"]
    for d in search_dirs:
        for candidate in glob.glob(os.path.join(str(d), "**", "LLCP2015.XPT*"), recursive=True):
            if candidate.upper().rstrip().endswith(".XPT"):
                return Path(candidate)
    return None


def read_and_verify_xpt(xpt_path: str | os.PathLike, columns: list[str] = RAW_COLUMNS,
                        chunksize: int = 100_000, compute_hash: bool = True
                        ) -> tuple[pd.DataFrame, VerificationReport]:
    """Read the XPT file in chunks, keep ``columns`` and verify it against the codebook."""
    xpt_path = Path(xpt_path)
    if not xpt_path.exists():
        raise FileNotFoundError(xpt_path)
    parts, n_rows, n_vars, counts = [], 0, None, None
    try:
        reader = pd.read_sas(xpt_path, format="xport", chunksize=chunksize)
        for chunk in reader:
            n_vars = chunk.shape[1]
            n_rows += len(chunk)
            missing_cols = set(columns) - set(chunk.columns)
            if missing_cols:
                raise KeyError(f"Expected BRFSS variables not found: {sorted(missing_cols)}")
            vc = chunk["DIABETE3"].value_counts(dropna=False)
            counts = vc if counts is None else counts.add(vc, fill_value=0)
            parts.append(chunk[columns].copy())
    except ValueError as exc:
        raise RuntimeError(f"{xpt_path} could not be parsed as a SAS XPORT file") from exc
    df = pd.concat(parts, ignore_index=True)

    observed = {}
    for key, val in counts.items():
        observed["missing" if pd.isna(key) else float(key)] = int(val)
    report = VerificationReport(
        n_records=n_rows,
        n_variables=int(n_vars),
        diabete3_counts=observed,
        records_match=n_rows == EXPECTED_N_RECORDS,
        variables_match=int(n_vars) == EXPECTED_N_VARIABLES,
        target_counts_match=observed == EXPECTED_DIABETE3,
        source_file=str(xpt_path),
    )
    if compute_hash:
        report.sha256 = sha256_file(xpt_path)
        report.sha256_matches_reference = report.sha256 == REFERENCE_SHA256
        if not report.sha256_matches_reference:
            report.notes.append("SHA-256 differs from the reference copy; content checks still apply "
                                "(a re-export of the same data can differ byte-wise).")
    return df, report


def load_brfss_2015(xpt_path: str | os.PathLike | None = None, cache_path: Path = CACHE_PATH,
                    download_if_missing: bool = True, use_cache: bool = True
                    ) -> tuple[pd.DataFrame, VerificationReport | None]:
    """Load the project subset of BRFSS 2015, using a Parquet cache when available.

    Order of precedence: cached Parquet -> local XPT -> official CDC download.
    The verification report is returned whenever the XPT is read; when the
    cache is used, the report saved next to the cache is returned instead.
    """
    import json

    report_path = Path(cache_path).with_suffix(".verification.json")
    if use_cache and Path(cache_path).exists():
        df = pd.read_parquet(cache_path)
        report = None
        if report_path.exists():
            report = VerificationReport(**{k: v for k, v in json.loads(report_path.read_text()).items()
                                           if k != "passed"})
            report.diabete3_counts = {("missing" if k == "missing" else float(k)): v
                                      for k, v in report.diabete3_counts.items()}
        return df, report

    xpt_path = Path(xpt_path) if xpt_path else find_raw_file()
    if xpt_path is None:
        if not download_if_missing:
            raise FileNotFoundError(f"LLCP2015.XPT not found. Download it from {CDC_LANDING_PAGE}.")
        xpt_path = download_brfss_2015()
    df, report = read_and_verify_xpt(xpt_path)
    if not report.passed:
        raise ValueError(f"Loaded file does not match the published codebook: {report.to_dict()}")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return df, report
