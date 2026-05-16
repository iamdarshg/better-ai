import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DatasetQACheckResult:
    name: str
    passed: int = 0
    failed: int = 0
    offenders: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetQAReport:
    dataset_path: str
    total_records: int
    checks: Dict[str, DatasetQACheckResult]
    critical_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "total_records": self.total_records,
            "critical_failed": self.critical_failed,
            "checks": {k: asdict(v) for k, v in self.checks.items()},
        }


@dataclass
class DatasetQAThresholds:
    max_duplicate_ratio: float = 0.05
    min_token_length: int = 2
    max_token_length: int = 4096
    outlier_length_multiplier: float = 10.0
    max_malformed_ratio: float = 0.01
    max_schema_error_ratio: float = 0.01
    max_language_domain_mismatch_ratio: float = 0.2
    offender_sample_size: int = 5


DEFAULT_ALLOWED_DOMAINS = {"general", "code", "math", "dialogue"}


def _token_length(text: str) -> int:
    return len(text.split())


def _load_records(dataset_path: str) -> List[Any]:
    ext = os.path.splitext(dataset_path)[1].lower()
    records: List[Any] = []
    if ext == ".jsonl":
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append({"__malformed__": line})
    elif ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
    elif ext == ".csv":
        with open(dataset_path, "r", encoding="utf-8") as f:
            records.extend(csv.DictReader(f))
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append({"text": line})
    return records


def run_dataset_qa(
    dataset_path: str,
    thresholds: Optional[DatasetQAThresholds] = None,
) -> DatasetQAReport:
    thresholds = thresholds or DatasetQAThresholds()
    records = _load_records(dataset_path)
    checks = {
        "schema_validity": DatasetQACheckResult("schema_validity"),
        "empty_or_duplicate": DatasetQACheckResult("empty_or_duplicate"),
        "length_outliers": DatasetQACheckResult("length_outliers"),
        "language_domain_mismatch": DatasetQACheckResult("language_domain_mismatch"),
        "malformed_records": DatasetQACheckResult("malformed_records"),
    }

    seen = set()
    lengths: List[int] = []

    for idx, record in enumerate(records):
        malformed = not isinstance(record, dict) or "__malformed__" in record
        if malformed:
            checks["malformed_records"].failed += 1
            if len(checks["malformed_records"].offenders) < thresholds.offender_sample_size:
                checks["malformed_records"].offenders.append({"index": idx, "record": record})
            continue
        checks["malformed_records"].passed += 1

        text = record.get("text")
        domain = record.get("domain", "general")
        language = record.get("language", "unknown")
        valid_schema = isinstance(text, str) and text.strip() != ""
        if valid_schema:
            checks["schema_validity"].passed += 1
        else:
            checks["schema_validity"].failed += 1
            if len(checks["schema_validity"].offenders) < thresholds.offender_sample_size:
                checks["schema_validity"].offenders.append({"index": idx, "record": record})
            continue

        norm_text = text.strip().lower()
        if not norm_text or norm_text in seen:
            checks["empty_or_duplicate"].failed += 1
            if len(checks["empty_or_duplicate"].offenders) < thresholds.offender_sample_size:
                checks["empty_or_duplicate"].offenders.append({"index": idx, "record": record})
        else:
            seen.add(norm_text)
            checks["empty_or_duplicate"].passed += 1

        tok_len = _token_length(text)
        lengths.append(tok_len)
        if tok_len < thresholds.min_token_length or tok_len > thresholds.max_token_length:
            checks["length_outliers"].failed += 1
            if len(checks["length_outliers"].offenders) < thresholds.offender_sample_size:
                checks["length_outliers"].offenders.append(
                    {"index": idx, "token_length": tok_len, "record": record}
                )
        else:
            checks["length_outliers"].passed += 1

        mismatch = (domain not in DEFAULT_ALLOWED_DOMAINS) or (
            language == "en" and domain == "code" and "def " not in text and "class " not in text
        )
        if mismatch:
            checks["language_domain_mismatch"].failed += 1
            if len(checks["language_domain_mismatch"].offenders) < thresholds.offender_sample_size:
                checks["language_domain_mismatch"].offenders.append(
                    {"index": idx, "language": language, "domain": domain, "record": record}
                )
        else:
            checks["language_domain_mismatch"].passed += 1

    # percentile-ish outlier check by dynamic threshold
    if lengths:
        avg = sum(lengths) / len(lengths)
        dynamic_threshold = max(thresholds.max_token_length, int(avg * thresholds.outlier_length_multiplier))
        for idx, record in enumerate(records):
            if isinstance(record, dict) and isinstance(record.get("text"), str):
                tok_len = _token_length(record["text"])
                if tok_len > dynamic_threshold:
                    checks["length_outliers"].failed += 1
                    if len(checks["length_outliers"].offenders) < thresholds.offender_sample_size:
                        checks["length_outliers"].offenders.append(
                            {"index": idx, "token_length": tok_len, "dynamic_threshold": dynamic_threshold}
                        )

    total = max(1, len(records))
    duplicate_ratio = checks["empty_or_duplicate"].failed / total
    malformed_ratio = checks["malformed_records"].failed / total
    schema_ratio = checks["schema_validity"].failed / total
    mismatch_ratio = checks["language_domain_mismatch"].failed / total

    critical_failed = (
        duplicate_ratio > thresholds.max_duplicate_ratio
        or malformed_ratio > thresholds.max_malformed_ratio
        or schema_ratio > thresholds.max_schema_error_ratio
        or mismatch_ratio > thresholds.max_language_domain_mismatch_ratio
    )

    return DatasetQAReport(dataset_path=dataset_path, total_records=len(records), checks=checks, critical_failed=critical_failed)


def persist_report(report: DatasetQAReport, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def run_audit_or_raise(dataset_path: str, thresholds: Optional[DatasetQAThresholds] = None, output_path: Optional[str] = None) -> DatasetQAReport:
    report = run_dataset_qa(dataset_path, thresholds)
    if output_path:
        persist_report(report, output_path)
    if report.critical_failed:
        raise ValueError(f"Dataset QA failed for {dataset_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset QA audit")
    parser.add_argument("dataset_path")
    parser.add_argument("--output", default="./logs/dataset_qa_report.json")
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.05)
    parser.add_argument("--min-token-length", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=4096)
    args = parser.parse_args()

    thresholds = DatasetQAThresholds(
        max_duplicate_ratio=args.max_duplicate_ratio,
        min_token_length=args.min_token_length,
        max_token_length=args.max_token_length,
    )
    report = run_audit_or_raise(args.dataset_path, thresholds=thresholds, output_path=args.output)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
