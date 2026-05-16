import json
import os
import tempfile
import unittest

from better_ai.data.qa import DatasetQAThresholds, run_dataset_qa, run_audit_or_raise


class TestDatasetQA(unittest.TestCase):
    def _write_jsonl(self, records):
        fd, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(fd)
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                if isinstance(r, str):
                    f.write(r + '\n')
                else:
                    f.write(json.dumps(r) + '\n')
        return path

    def test_schema_validity_check(self):
        p = self._write_jsonl([{"text": "ok", "domain": "general"}, {"text": ""}])
        report = run_dataset_qa(p)
        self.assertGreater(report.checks["schema_validity"].failed, 0)

    def test_empty_duplicate_check(self):
        p = self._write_jsonl([{"text": "dup"}, {"text": "dup"}])
        report = run_dataset_qa(p)
        self.assertGreater(report.checks["empty_or_duplicate"].failed, 0)

    def test_length_outlier_check(self):
        p = self._write_jsonl([{"text": "short"}, {"text": "x " * 5000}])
        report = run_dataset_qa(p)
        self.assertGreater(report.checks["length_outliers"].failed, 0)

    def test_language_domain_mismatch_check(self):
        p = self._write_jsonl([{"text": "hello world", "language": "en", "domain": "code"}])
        report = run_dataset_qa(p)
        self.assertGreater(report.checks["language_domain_mismatch"].failed, 0)

    def test_malformed_record_check(self):
        p = self._write_jsonl(['{"text": "ok"}', '{bad json'])
        report = run_dataset_qa(p)
        self.assertGreater(report.checks["malformed_records"].failed, 0)

    def test_audit_gating_behavior(self):
        p = self._write_jsonl([{"text": "dup"}, {"text": "dup"}])
        with self.assertRaises(ValueError):
            run_audit_or_raise(
                p,
                thresholds=DatasetQAThresholds(max_duplicate_ratio=0.1),
                output_path=tempfile.mktemp(suffix='.json'),
            )


if __name__ == '__main__':
    unittest.main()
