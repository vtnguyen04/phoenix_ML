"""Tests for DataValidator — data quality checks."""

import pandas as pd
import pytest

from phoenix_ml.domain.training.services.data_validator import DataValidator, Severity


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(max_null_percent=30, outlier_threshold=3.0, max_duplicate_percent=10)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame({
        "income": [50000, 60000, 70000, 80000, 55000],
        "age": [25, 30, 35, 40, 28],
        "debt": [1000, 2000, 1500, 3000, 500],
        "target": [0, 1, 1, 0, 1],
    })


class TestDataValidator:
    def test_clean_data_passes(self, validator: DataValidator, clean_df: pd.DataFrame) -> None:
        report = validator.validate(clean_df, target_column="target")
        assert report.passed is True
        assert len(report.errors) == 0
        assert report.total_rows == 5
        assert report.total_columns == 4

    def test_empty_dataset_fails(self, validator: DataValidator) -> None:
        df = pd.DataFrame({"a": []})
        report = validator.validate(df)
        assert report.passed is False
        assert any(i.issue_type == "empty_dataset" for i in report.issues)

    def test_high_null_rate_error(self, validator: DataValidator) -> None:
        df = pd.DataFrame({"a": [1.0, None, None, None, None], "b": [1, 2, 3, 4, 5]})
        report = validator.validate(df)
        null_errors = [i for i in report.issues if i.issue_type == "high_null_rate"]
        assert len(null_errors) == 1
        assert null_errors[0].column == "a"
        assert null_errors[0].severity == Severity.ERROR

    def test_low_null_rate_warning(self, validator: DataValidator) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0], "b": [1, 2, 3, 4, 5]})
        report = validator.validate(df)
        null_warns = [i for i in report.issues if i.issue_type == "has_nulls"]
        assert len(null_warns) == 1
        assert null_warns[0].severity == Severity.WARNING

    def test_duplicates_detected(self) -> None:
        v = DataValidator(max_duplicate_percent=5)
        df = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": [10, 10, 10, 20, 30]})
        report = v.validate(df)
        dup_issues = [i for i in report.issues if i.issue_type == "high_duplicates"]
        assert len(dup_issues) == 1

    def test_outliers_detected(self, validator: DataValidator) -> None:
        data = list(range(100)) + [99999]
        df = pd.DataFrame({"val": data})
        report = validator.validate(df)
        outlier_issues = [i for i in report.issues if i.issue_type == "outliers_detected"]
        assert len(outlier_issues) >= 1

    def test_range_violation(self, validator: DataValidator) -> None:
        df = pd.DataFrame({"age": [25, 30, 150, 35], "income": [50000, 60000, 70000, 80000]})
        report = validator.validate(df, feature_ranges={"age": (0, 120)})
        range_issues = [i for i in report.issues if i.issue_type == "out_of_range"]
        assert len(range_issues) == 1

    def test_constant_target_error(self, validator: DataValidator) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 0, 0]})
        report = validator.validate(df, target_column="target")
        target_issues = [i for i in report.issues if i.issue_type == "constant_target"]
        assert len(target_issues) == 1

    def test_column_stats_computed(self, validator: DataValidator, clean_df: pd.DataFrame) -> None:
        report = validator.validate(clean_df)
        assert len(report.column_stats) == 4
        income_stat = next(s for s in report.column_stats if s.name == "income")
        assert income_stat.mean is not None
        assert income_stat.null_count == 0

    def test_to_dict(self, validator: DataValidator, clean_df: pd.DataFrame) -> None:
        report = validator.validate(clean_df)
        d = report.to_dict()
        assert "total_rows" in d
        assert "passed" in d
        assert isinstance(d["issues"], list)
