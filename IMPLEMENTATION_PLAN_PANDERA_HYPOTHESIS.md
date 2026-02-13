# Implementation Plan: Migrate Test Suite to Pandera + Hypothesis

> **Epic:** [#45 — Change testing to use pandera + hypothesis](https://github.com/esalient/pysalient/issues/45)
> **Target Milestone:** v0.1.0
> **Audience:** Python programmer implementing step-by-step
> **Branch:** `plan/pandera-hypothesis-migration`

---

## Context and Goal

pySALIENT currently tests against a static 1.1 MB parquet file (`tests/test_data/anonymised_sample.parquet`) and hand-built synthetic fixtures with hard-coded expected values. This approach:

- Couples tests to a specific dataset rather than proving the library works on **any** valid input
- Uses a simple 8-point synthetic fixture that is not complex enough to catch subtle bugs
- Stores a binary file (1.1 MB) in the repository that cannot be diffed or reviewed
- Makes generalisation hard to verify

The goal is to replace all static test data with **Pandera schemas** (defining what valid data looks like) and **Hypothesis strategies** (generating random valid data to test against), then **delete the static parquet file** from `tests/`.

The `examples/data/anonymised_sample.parquet` file stays — it is for user-facing notebooks, not tests.

---

## Ground Rules (Read Before Every PR)

1. **Run `pixi run lint` (Ruff) on every change before committing.** No exceptions. CI will reject unlinted code.
2. **Run `pixi run format` after lint.** Ruff fix + format, then commit.
3. **Run `pixi run test` locally before pushing.** All 245+ existing tests must keep passing until explicitly replaced.
4. **Coverage gate is 75%.** Do not drop below this — CI enforces it (`--cov-fail-under=75`).
5. **One PR per phase** (or per issue if preferred). Do not bundle everything into a single PR.
6. **Follow TDD:** write the new test first, see it pass, then remove the old test it replaces.
7. **Never remove an old test until its replacement is merged and green.**

---

## Current State: What Exists Today

### Test Files With Active Tests (245 tests total)

| Area | File | Tests | Static Data Used |
|------|------|-------|------------------|
| **Evaluation core** | `tests/evaluation/test_evaluation.py` | 39 | `synth_data_basic` (8-pt), `synth_data_larger` (100-pt), `anonymised_sample.parquet` |
| **Threshold utils** | `tests/evaluation/test_evaluation_utils.py` | 46 | None (pure logic) |
| **Bootstrap CI** | `tests/evaluation/test_bootstrap_utils.py` | 17 | `sample_data` (100-pt, seed=123) |
| **Eval process** | `tests/evaluation/test_evaluation_process.py` | 19 | `basic_test_data` (4-pt), `larger_test_data` (100-pt) |
| **Time-to-event** | `tests/evaluation/test_evaluation_time_to_event.py` | 20 | `time_to_event_data` (3 encounters) |
| **Pydantic models** | `tests/evaluation/test_models.py` | 13 | `sample_table` fixture |
| **Analytical CI** | `tests/evaluation/test_analytical_ci_utils.py` | 4 | None (pure math) |
| **Config API** | `tests/evaluation/test_evaluation_config_api.py` | 6 | Inline `_build_table_with_metadata()` |
| **Parallel bootstrap** | `tests/evaluation/test_bootstrap_utils_parallel.py` | 3 | Inline fixture |
| **Parallel process** | `tests/evaluation/test_evaluation_process_parallel.py` | 2 | Inline fixture |
| **I/O** | `tests/io/test_io.py` | 31 | `sample_data_dict` (6-row), `anonymised_sample.parquet` |
| **I/O core** | `tests/io/test_io_core.py` | 9 | PyArrow tables inline |
| **I/O config** | `tests/io/test_io_config_api.py` | 2 | `sample_data_dict()` |
| **I/O utils** | `tests/io/test_io_utils.py` | 0 | Empty stub |
| **Visualisation** | `tests/visualisation/test_visualisation.py` | 34 | `sample_eval_table` (inline fixture) |

### Static Data Files to Remove

| File | Size | Used By |
|------|------|---------|
| `tests/test_data/anonymised_sample.parquet` | 1.1 MB | `test_evaluation.py` (integration tests), `test_io.py` (realistic data fixture) |

### Files to Keep (NOT part of this migration)

| File | Reason |
|------|--------|
| `examples/data/anonymised_sample.parquet` | User-facing example notebook — not test infrastructure |

### Pixi Environments Already Configured

- `pandera-testing` — includes `pandera` + `pandera-strategies` (Hypothesis integration)
- `pandera-full` — adds `pandera-io`
- `all` — everything

You will work in the `default` environment (which already includes `pandera`) for most work. Use `pandera-testing` when writing Hypothesis-powered tests.

---

## Phase 1: Foundation (Issues #47, #48, #49, #55)

**Goal:** Build the shared infrastructure that all later phases depend on. No existing tests are changed or deleted in this phase.

### Step 1.1 — Define Pandera Schemas (`#47`)

**File to create:** `tests/schemas.py`

Define one Pandera `DataFrameSchema` for each data shape used across the test suite. Each schema encodes the **contract** for what valid data looks like — column names, types, value ranges, nullability.

Schemas to define:

| Schema Name | Purpose | Key Constraints |
|-------------|---------|-----------------|
| `evaluation_input_schema` | Input to `evaluation()` | `y_proba` float [0, 1], `y_label` int {0, 1}, `encounter_id` str not null, `event_timestamp` float >= 0 |
| `evaluation_results_schema` | Output of `evaluation()` | All 46+ columns, `threshold` float [0, 1], `TP/TN/FP/FN` int >= 0, `AUROC/AUPRC` float [0, 1] or None |
| `io_csv_input_schema` | CSV files loaded by I/O | `prediction_probability` float [0, 1], `true_label` int {0, 1}, `encounter_id` not null |
| `io_parquet_input_schema` | Parquet files loaded by I/O | Same as CSV but with `event_timestamp` as datetime |
| `time_to_event_input_schema` | Time-to-event evaluation input | Adds `culture_event`, `suspected_infection` as float {0, 1} |
| `visualisation_input_schema` | Input to `format_evaluation_table()` | `threshold` float, metric columns float, `Sample Size` int |

**How to write a schema (example):**

```python
import pandera as pa

evaluation_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(
            float, pa.Check.greater_than_or_equal_to(0.0)
        ),
        "y_proba": pa.Column(
            float, pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)
        ),
        "y_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    # Allow additional columns (model_id, task_id, etc.)
    strict=False,
)
```

**Validation:**
- Write tests in `tests/test_schemas.py` that confirm each schema accepts known-good data and rejects known-bad data (wrong types, out-of-range values, nulls where not allowed).
- Run `pixi run lint && pixi run test`.

---

### Step 1.2 — Create Hypothesis Strategies (`#48`)

**File to create:** `tests/strategies.py`

Hypothesis strategies are **generators** that produce random valid data matching your schemas. They replace hard-coded fixtures.

Strategies to define:

| Strategy | What It Generates | Used By |
|----------|-------------------|---------|
| `evaluation_data_strategy(n_rows, n_encounters)` | DataFrame matching `evaluation_input_schema` with correlated labels/probas | Evaluation tests |
| `temporal_data_strategy(n_rows, n_encounters)` | Adds datetime `event_timestamp` | Time-to-event tests |
| `encounter_grouped_strategy(n_encounters, events_per_encounter)` | Multiple events per encounter, realistic grouping | Aggregation tests |
| `io_sample_strategy(n_rows)` | DataFrame matching `io_csv_input_schema` | I/O tests |
| `evaluation_results_strategy(n_thresholds)` | DataFrame matching `evaluation_results_schema` | Visualisation tests |

**How to write a strategy (example):**

```python
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings


def evaluation_data_strategy(
    n_rows=st.integers(min_value=2, max_value=200),
):
    """Generate a random valid evaluation DataFrame."""

    @st.composite
    def _build(draw):
        n = draw(n_rows)
        probas = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=n,
                max_size=n,
            )
        )
        labels = draw(
            st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n)
        )
        encounter_ids = draw(
            st.lists(
                st.text(min_size=1, max_size=5, alphabet="ABCDE"),
                min_size=n,
                max_size=n,
            )
        )
        return pd.DataFrame(
            {
                "y_proba": probas,
                "y_label": labels,
                "encounter_id": encounter_ids,
                "event_timestamp": list(range(n)),
            }
        )

    return _build()
```

**Validation:**
- Write tests in `tests/test_strategies.py` that generate 10+ examples and validate each against the corresponding Pandera schema.
- Run `pixi run lint && pixi run test`.

---

### Step 1.3 — Centralise Shared Fixtures in conftest.py (`#49`)

**File to modify:** `tests/conftest.py`

Currently this file only adds the project root to `sys.path`. Add shared fixtures that use your new strategies and schemas. These will be available to all test files automatically.

Fixtures to add:

```python
import pytest
from tests.schemas import evaluation_input_schema
from tests.strategies import evaluation_data_strategy


@pytest.fixture
def valid_evaluation_df():
    """A single random valid evaluation DataFrame."""
    from hypothesis import given, settings, HealthCheck

    # Generate one example for use as a fixture
    df = evaluation_data_strategy().example()
    return evaluation_input_schema.validate(df)
```

Also add the Hypothesis profile configuration here (for `#55`):

```python
from hypothesis import settings, HealthCheck

# Local development: fast, fewer examples
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=500,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI: thorough, more examples
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load profile from environment variable (default: dev)
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

**Validation:**
- Run `pixi run lint && pixi run test` — all existing tests must still pass (you added fixtures, you did not change existing tests).

---

### Step 1.4 — Update CI for Hypothesis (`#55`)

**File to modify:** `.github/workflows/ci.yml`

Add the `HYPOTHESIS_PROFILE` environment variable so CI runs more examples:

```yaml
    - name: Run tests with coverage gate
      env:
        HYPOTHESIS_PROFILE: ci
      run: pixi run pytest --cov=pysalient --cov-report=term-missing --cov-fail-under=75 tests/
```

**Validation:**
- Push the branch. CI must pass on all 4 Python versions (3.11, 3.12, 3.13, 3.14).

---

### Phase 1 Checklist

- [ ] `tests/schemas.py` exists with 6+ schemas
- [ ] `tests/test_schemas.py` passes
- [ ] `tests/strategies.py` exists with 5+ strategies
- [ ] `tests/test_strategies.py` passes
- [ ] `tests/conftest.py` has shared fixtures + Hypothesis profiles
- [ ] CI updated with `HYPOTHESIS_PROFILE: ci`
- [ ] `pixi run lint` clean
- [ ] `pixi run test` passes (all 245 existing tests still green)
- [ ] Coverage >= 75%

---

## Phase 2: Property-Based Tests (Issue #53)

**Goal:** Add new Hypothesis-powered tests that encode **mathematical invariants** — things that must be true for any valid dataset. These run alongside existing tests. Nothing is removed yet.

**File to create:** `tests/evaluation/test_evaluation_properties.py`

### Properties to Test

These are the mathematical invariants that prove correctness without depending on any specific dataset:

#### Confusion Matrix Identities

```python
@given(data=evaluation_data_strategy())
@settings(max_examples=100)
def test_confusion_matrix_sums_to_n(data):
    """TP + TN + FP + FN must equal the total number of samples."""
    # Build PyArrow table, attach metadata, run evaluation at threshold 0.5
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    assert r["TP"][0] + r["TN"][0] + r["FP"][0] + r["FN"][0] == len(data)


@given(data=evaluation_data_strategy())
def test_tp_plus_fn_equals_total_positives(data):
    """TP + FN must equal the number of actual positive labels."""
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    total_pos = sum(data["y_label"])
    assert r["TP"][0] + r["FN"][0] == total_pos


@given(data=evaluation_data_strategy())
def test_fp_plus_tn_equals_total_negatives(data):
    """FP + TN must equal the number of actual negative labels."""
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    total_neg = len(data) - sum(data["y_label"])
    assert r["FP"][0] + r["TN"][0] == total_neg
```

#### Threshold Boundary Behaviour

```python
@given(data=evaluation_data_strategy())
def test_threshold_zero_all_predicted_positive(data):
    """At threshold 0.0, every sample is predicted positive, so FN must be 0."""
    results = run_evaluation(data, thresholds=[0.0])
    r = results.to_pydict()
    # Find the row for threshold 0.0
    idx = r["threshold"].index(0.0)
    assert r["FN"][idx] == 0
```

#### Metric Consistency

```python
@given(data=evaluation_data_strategy())
def test_sensitivity_equals_tp_over_tp_plus_fn(data):
    """Sensitivity = TP / (TP + FN) when there are positive labels."""
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    tp, fn = r["TP"][0], r["FN"][0]
    if tp + fn > 0:
        expected = tp / (tp + fn)
        assert r["Sensitivity"][0] == pytest.approx(expected, abs=1e-9)


@given(data=evaluation_data_strategy())
def test_ppv_equals_tp_over_tp_plus_fp(data):
    """PPV = TP / (TP + FP) when there are predicted positives."""
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    tp, fp = r["TP"][0], r["FP"][0]
    if tp + fp > 0:
        expected = tp / (tp + fp)
        assert r["PPV"][0] == pytest.approx(expected, abs=1e-9)
```

#### AUROC / AUPRC Bounds

```python
@given(data=evaluation_data_strategy())
def test_auroc_bounded_zero_one(data):
    """AUROC must be in [0, 1] when both classes are present."""
    if len(set(data["y_label"])) < 2:
        return  # Skip single-class data
    results = run_evaluation(data, thresholds=[0.5])
    r = results.to_pydict()
    assert 0.0 <= r["AUROC"][0] <= 1.0
```

#### Confidence Interval Ordering

```python
@given(data=evaluation_data_strategy(n_rows=st.integers(30, 100)))
def test_ci_lower_le_point_le_upper(data):
    """Lower CI <= point estimate <= Upper CI for all metrics."""
    results = run_evaluation(
        data, thresholds=[0.5], calculate_au_ci=True, bootstrap_rounds=50
    )
    r = results.to_pydict()
    if r["AUROC_Lower_CI"][0] is not None:
        assert r["AUROC_Lower_CI"][0] <= r["AUROC"][0] <= r["AUROC_Upper_CI"][0]
```

### Full List of Properties (aim for 15+)

1. `TP + TN + FP + FN == N`
2. `TP + FN == total positives`
3. `FP + TN == total negatives`
4. `TP + FP == total predicted positives`
5. At threshold 0.0: `FN == 0`
6. At threshold > max(proba): `TP == 0`
7. `Sensitivity == TP / (TP + FN)` (when denominator > 0)
8. `PPV == TP / (TP + FP)` (when denominator > 0)
9. `Specificity == TN / (TN + FP)` (when denominator > 0)
10. `NPV == TN / (TN + FN)` (when denominator > 0)
11. `Accuracy == (TP + TN) / N`
12. `F1 == 2 * PPV * Sensitivity / (PPV + Sensitivity)` (when denominator > 0)
13. `Prevalence == total positives / N`
14. `0 <= AUROC <= 1` (two-class)
15. `0 <= AUPRC <= 1` (two-class)
16. `Lower_CI <= point_estimate <= Upper_CI`
17. Output row count == number of unique thresholds

### Phase 2 Checklist

- [ ] `tests/evaluation/test_evaluation_properties.py` exists with 15+ property tests
- [ ] All property tests pass with `pixi run test`
- [ ] All 245 existing tests still pass (nothing was removed)
- [ ] `pixi run lint` clean
- [ ] Coverage >= 75%

---

## Phase 3: Migration (Issues #50, #51, #52, #56)

**Goal:** Replace static-data-dependent tests with schema/strategy-based tests. After this phase, no test depends on `anonymised_sample.parquet`.

> **Important:** Follow this order. Each step is a separate commit (or PR). Run `pixi run lint && pixi run test` after every change.

### Step 3.1 — Migrate `test_io.py` (`#50`)

**File:** `tests/io/test_io.py`

**What to change:**

1. **Replace `realistic_parquet_path` / `realistic_table` fixtures** (lines 120-141) with a strategy-generated fixture:

   ```python
   @pytest.fixture(scope="module")
   def realistic_generated_data(tmp_path_factory):
       """Generate realistic test data and write to a temp parquet file."""
       from tests.strategies import io_sample_strategy
       df = io_sample_strategy(n_rows=500).example()
       path = tmp_path_factory.mktemp("data") / "generated_sample.parquet"
       df.to_parquet(path)
       return path
   ```

2. **Update `test_load_from_realistic_parquet`** (lines 387-416) to use the new fixture instead of the static file.

3. **Keep all other tests unchanged** — `sample_data_dict` and its derivatives are already inline synthetic data (not from the parquet file), so they stay.

4. **Remove the `TESTS_DIR` / `DATA_DIR` constants** that pointed to the static file directory.

**Validation:**
- `pixi run lint && pixi run test`
- Confirm `test_load_from_realistic_parquet` passes with generated data
- Confirm no test references `anonymised_sample.parquet`

---

### Step 3.2 — Migrate `test_evaluation.py` Integration Tests (`#51`)

**File:** `tests/evaluation/test_evaluation.py`

**What to change:**

1. **Replace `loaded_sample_data` fixture** (lines 1397-1413) with a strategy-generated fixture:

   ```python
   @pytest.fixture(scope="module")
   def loaded_sample_data():
       """Generate evaluation data with schema-validated structure."""
       from tests.strategies import evaluation_data_strategy
       import pysalient.io as csio

       df = evaluation_data_strategy(n_rows=st.just(500)).example()
       # Write to temp parquet and load via pysalient.io to test the full pipeline
       # ... (build PyArrow table with correct metadata)
   ```

2. **Update integration tests** (`test_integration_load_data_adds_correct_metadata`, `test_integration_evaluation_consumes_metadata_successfully`) to work with generated data. These tests check **structural** properties (metadata exists, output schema matches, AUROC in [0,1]) — they already do not check specific values, so the change is straightforward.

3. **Remove `SAMPLE_DATA_PATH` and `SAMPLE_COL_MAP`** constants (lines 34-44).

4. **Keep all synthetic fixture tests unchanged** — the `synth_data_basic`, `synth_data_larger` fixtures and their tests stay for now. They will be replaced by the property tests from Phase 2 once those are confirmed green.

**Validation:**
- `pixi run lint && pixi run test`

---

### Step 3.3 — Fill Stub Test Files (`#52, #56`)

**Files:** `tests/io/test_io_utils.py`, `tests/io/test_io_core.py` (expand), and the empty stubs:
- `tests/adapters/test_adapters.py`
- `tests/config/test_config.py`
- `tests/datasets/test_datasets.py`
- `tests/events/test_events.py`
- `tests/parser/test_parser.py`
- `tests/project/test_project.py`
- `tests/reports/test_reports.py`
- `tests/task/test_task.py`

**What to do:**

For modules that have source code (`io/_io_utils.py`, `io/_io_core.py`):
- Write real unit tests using strategies from `tests/strategies.py`
- Target 80%+ coverage of those source files

For modules that are empty stubs (adapters, config, datasets, events, parser, project, reports, task):
- If the source module has no code yet, add a single placeholder test:
  ```python
  def test_module_importable():
      """Verify the module can be imported."""
      import pysalient.adapters  # noqa: F401
  ```
- Do **not** write fake tests for functionality that does not exist yet

**Validation:**
- `pixi run lint && pixi run test`
- Coverage >= 75%

---

### Step 3.4 — Migrate Remaining Synthetic Fixtures

Once the property tests from Phase 2 are proven stable, migrate the remaining hand-built fixtures:

| Test File | Current Fixture | Replacement |
|-----------|----------------|-------------|
| `test_evaluation.py` | `synth_data_basic` (8-pt) | `evaluation_data_strategy` |
| `test_evaluation.py` | `synth_data_larger` (100-pt) | `evaluation_data_strategy(n_rows=st.just(100))` |
| `test_bootstrap_utils.py` | `sample_data` (100-pt) | `evaluation_data_strategy(n_rows=st.just(100))` |
| `test_evaluation_process.py` | `basic_test_data` (4-pt) | `evaluation_data_strategy(n_rows=st.just(4))` |
| `test_evaluation_process.py` | `larger_test_data` (100-pt) | `evaluation_data_strategy(n_rows=st.just(100))` |
| `test_evaluation_time_to_event.py` | `time_to_event_data` | `temporal_data_strategy` |
| `test_visualisation.py` | `sample_eval_table` | `evaluation_results_strategy` |

**Important:** When replacing hard-coded-value tests (e.g. `assert AUROC == 0.9375`), replace the assertion with the corresponding mathematical invariant from Phase 2. Do not simply delete the assertion.

**Example migration:**

```python
# BEFORE (value-based, coupled to 8-point dataset):
def test_evaluation_basic(synth_table_with_metadata):
    results = evaluation(synth_table_with_metadata, "m", "f", [0.5])
    r = results.to_pydict()
    assert r["TP"][0] == 3
    assert r["AUROC"][0] == pytest.approx(0.9375)

# AFTER (property-based, works on any data):
@given(data=evaluation_data_strategy())
def test_evaluation_basic_properties(data):
    table = build_pyarrow_table(data)
    results = evaluation(table, "m", "f", [0.5])
    r = results.to_pydict()
    # Mathematical invariants instead of specific values
    assert r["TP"][0] + r["TN"][0] + r["FP"][0] + r["FN"][0] == len(data)
    assert r["TP"][0] + r["FN"][0] == sum(data["y_label"])
    if len(set(data["y_label"])) == 2:
        assert 0.0 <= r["AUROC"][0] <= 1.0
```

**Validation:**
- `pixi run lint && pixi run test`
- No test references hard-coded expected values from the old fixtures
- Coverage >= 75%

### Phase 3 Checklist

- [ ] `test_io.py` no longer references `anonymised_sample.parquet`
- [ ] `test_evaluation.py` no longer references `anonymised_sample.parquet`
- [ ] `SAMPLE_DATA_PATH` and `SAMPLE_COL_MAP` removed
- [ ] `realistic_parquet_path` and `realistic_table` fixtures removed
- [ ] I/O core and utils have real tests
- [ ] Stub test files have at least import tests
- [ ] All value-based assertions replaced with invariant-based assertions
- [ ] `pixi run lint` clean
- [ ] `pixi run test` passes
- [ ] Coverage >= 75%

---

## Phase 4: Cleanup and Documentation (Issues #45 final, #54)

### Step 4.1 — Delete the Static Test Data

**Only do this after Phase 3 is fully merged and CI is green.**

```bash
git rm tests/test_data/anonymised_sample.parquet
# If the directory is now empty:
rmdir tests/test_data/
```

Also remove any `conftest.py` or test code that references the deleted path. Search for it:

```bash
pixi run ruff check . --fix
grep -r "anonymised_sample" tests/
grep -r "test_data" tests/
```

All results from those greps should be zero.

**Validation:**
- `pixi run lint && pixi run test`
- `git status` shows the parquet file deleted, no other unexpected changes

---

### Step 4.2 — Write Testing Guide (`#54`)

**File to create:** `docs/testing_guide.md`

Contents:
- How to run tests: `pixi run test`, `pixi run -e pandera-testing pytest`
- How Hypothesis profiles work (dev vs ci)
- How to read a Hypothesis failure (shrinking explanation)
- How to add a new property test (template)
- How to add a new Pandera schema (template)
- Link to Pandera docs and Hypothesis docs

---

### Phase 4 Checklist

- [ ] `tests/test_data/anonymised_sample.parquet` deleted from repo
- [ ] `examples/data/anonymised_sample.parquet` still exists (untouched)
- [ ] Zero grep results for `anonymised_sample` in `tests/`
- [ ] `docs/testing_guide.md` written
- [ ] `pixi run lint` clean
- [ ] `pixi run test` passes on all Python versions
- [ ] CI green on all 4 Python versions
- [ ] Coverage >= 75%

---

## Area-by-Area Impact Summary

### Evaluation (`tests/evaluation/`)

| File | Tests | Impact | Action |
|------|-------|--------|--------|
| `test_evaluation.py` | 39 | **HIGH** — 2 integration tests use parquet, all others use synthetic fixtures | Replace integration tests (Phase 3.2), migrate synthetic fixtures (Phase 3.4) |
| `test_evaluation_utils.py` | 46 | **NONE** — pure logic tests, no data files | No changes needed |
| `test_bootstrap_utils.py` | 17 | **LOW** — uses inline `sample_data` fixture | Migrate fixture to strategy (Phase 3.4) |
| `test_evaluation_process.py` | 19 | **LOW** — uses inline fixtures | Migrate fixtures to strategies (Phase 3.4) |
| `test_evaluation_time_to_event.py` | 20 | **MEDIUM** — structured temporal fixture | Migrate to `temporal_data_strategy` (Phase 3.4) |
| `test_models.py` | 13 | **LOW** — uses inline `sample_table` | Migrate to strategy (Phase 3.4) |
| `test_analytical_ci_utils.py` | 4 | **NONE** — pure math tests | No changes needed |
| `test_evaluation_config_api.py` | 6 | **LOW** — uses inline builder | Migrate builder to strategy (Phase 3.4) |
| `test_bootstrap_utils_parallel.py` | 3 | **LOW** — inline fixture | Migrate (Phase 3.4) |
| `test_evaluation_process_parallel.py` | 2 | **LOW** — inline fixture | Migrate (Phase 3.4) |

### I/O (`tests/io/`)

| File | Tests | Impact | Action |
|------|-------|--------|--------|
| `test_io.py` | 31 | **HIGH** — `realistic_parquet_path` fixture uses static file | Replace fixture (Phase 3.1) |
| `test_io_core.py` | 9 | **LOW** — inline PyArrow tables | Expand with strategy-based tests (Phase 3.3) |
| `test_io_config_api.py` | 2 | **LOW** — inline dict | Migrate to strategy (Phase 3.4) |
| `test_io_utils.py` | 0 | **STUB** | Implement real tests (Phase 3.3) |

### Visualisation (`tests/visualisation/`)

| File | Tests | Impact | Action |
|------|-------|--------|--------|
| `test_visualisation.py` | 34 | **LOW** — uses inline `sample_eval_table` fixture, no parquet dependency | Migrate fixture to `evaluation_results_strategy` (Phase 3.4) |

### Stub Modules

All stubs (`adapters`, `config`, `datasets`, `events`, `parser`, `project`, `reports`, `task`) — **No source code exists yet.** Add import-only tests (Phase 3.3). Do not write tests for functionality that doesn't exist.

---

## Ruff Enforcement Reminder

**Before every commit, every PR, every push:**

```bash
pixi run lint    # Ruff check + autofix
pixi run format  # Ruff format
```

If either command modifies files, stage the changes and include them in your commit. CI runs `pixi run lint` and will **reject** any PR with lint errors.

Ruff is configured in `pyproject.toml` (lines 150-194):
- Line length: 88
- Rules: E, F, W, I, N, UP
- Target: Python 3.11+
- E501 (line too long) is ignored

---

## Issue-to-Phase Mapping

| Issue | Title | Phase | Depends On |
|-------|-------|-------|------------|
| #47 | Expand pandera schemas | Phase 1, Step 1.1 | — |
| #48 | Create hypothesis strategies | Phase 1, Step 1.2 | #47 |
| #49 | Shared fixture library | Phase 1, Step 1.3 | #47, #48 |
| #55 | Hypothesis CI settings | Phase 1, Step 1.4 | #49 |
| #53 | Property-based evaluation tests | Phase 2 | #47, #48 |
| #50 | Migrate test_io.py | Phase 3, Step 3.1 | #47, #48, #49 |
| #51 | Migrate test_evaluation.py | Phase 3, Step 3.2 | #47, #48, #49 |
| #52 | IO core and utils tests | Phase 3, Step 3.3 | #47, #48 |
| #56 | Stub module tests | Phase 3, Step 3.3 | #47, #48 |
| #54 | Documentation | Phase 4 | All above |
| #45 | Epic: close when all done | Phase 4 | All above |

---

## Definition of Done (for closing #45)

- [ ] Zero static binary test files in `tests/`
- [ ] `examples/data/anonymised_sample.parquet` still present for notebooks
- [ ] 40+ property-based tests passing
- [ ] All prior test coverage maintained or improved
- [ ] All tests pass on Python 3.11, 3.12, 3.13, 3.14
- [ ] `pixi run lint` clean
- [ ] Coverage >= 75%
- [ ] `docs/testing_guide.md` exists
- [ ] All sub-issues (#47-#56) closed
