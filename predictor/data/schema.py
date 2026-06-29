import datetime
import warnings
import polars as pl

from predictor.config.pipeline_config import PipelineConfig
from ..constants.constants import EXPECTED_COLUMNS

def validate_schema(
    df: pl.DataFrame,
    expected: dict[str, type[pl.DataType]] | None = None,
    start_date: datetime.date | None = None,
    stale_days: int = 7,
    pipeline_config: PipelineConfig | None = None
) -> None:
    """Assert that the DataFrame matches the expected schema.

    Performs three checks:
    1. All expected column names are present (raises on diff).
    2. Each column's dtype matches the expected dtype (raises on mismatch).
    3. The most recent ``date`` value is not more than ``stale_days`` old
       (emits a warning if stale, does not raise).

    Args:
        df: The raw ingested DataFrame to validate.
        expected: Schema mapping.  Defaults to ``EXPECTED_COLUMNS``.
        start_date: If provided, asserts ``df['date'].min() <=
            start_date`` (i.e. data reaches back far enough).
        stale_days: Warn if the most recent date is older than this many
            days relative to today.

    Raises:
        ValueError: If any expected columns are missing or have wrong dtypes.
    """
    assert pipeline_config is not None, "pipeline_config must be provided for date checks"
    
    if expected is None:
        expected = EXPECTED_COLUMNS

    actual_cols = set(df.columns)
    expected_cols = set(expected.keys())

    # 1. Column presence
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(
            f"validate_schema: missing columns {sorted(missing)}. "
            f"Present columns: {sorted(actual_cols)}"
        )

    # 2. Dtype check
    mismatches: list[str] = []
    schema_map = {col: df.schema[col] for col in df.columns}
    for col, exp_dtype in expected.items():
        actual_dtype = schema_map[col]
        # Compare base dtype (ignoring nullability wrappers)
        if type(actual_dtype) is not exp_dtype:
            mismatches.append(
                f"  {col}: expected {exp_dtype.__name__}, got {actual_dtype}"
            )
    if mismatches:
        raise ValueError(
            "validate_schema: dtype mismatches:\n" + "\n".join(mismatches)
        )

    # 3. Staleness warning
    max_date: datetime.date = df["date"].max()
    today = pipeline_config.prod_ref_date
    delta_days = (today - max_date).days
    if delta_days > stale_days:
        warnings.warn(
            f"validate_schema: most recent match date is {max_date} "
            f"({delta_days} days ago). Dataset may be stale.",
            stacklevel=2,
        )

    # 4. Optional start-date coverage check
    if start_date is not None:
        min_date: datetime.date = df["date"].min()
        assert min_date <= start_date, (
            f"validate_schema: earliest date {min_date} is after "
            f"start_date {start_date}"
        )
