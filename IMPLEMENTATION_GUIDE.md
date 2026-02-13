# pySALIENT Implementation Guide: Type Safety & Coverage

## Current Status (2026-02-13)

**Coverage**: 65% (baseline: 61%) | **Tests**: 270 passed, 6 skipped | **Gate**: ✅ Meeting 65% target

### ✅ Completed Work

#### Pydantic Type Safety (Phases 1-2)
- **Foundation models** (`pysalient/_shared_models.py`):
  - `BaseConfig` with validation settings
  - `TimeUnit` and `ThresholdCIMethod` enums

- **Evaluation models** (`pysalient/evaluation/_models.py`):
  - `ThresholdConfig`, `TimeToEventConfig`, `ConfidenceIntervalConfig`, `EvaluationConfig`
  - Grouped 22 flat parameters into validated config objects
  - Model coverage: 92%

- **IO models** (`pysalient/io/_models.py`):
  - `ColumnConfig`, `LoadConfig`
  - Model coverage: 100%

- **Dual API Pattern** (backward compatible):
  - `calculate_bootstrap_ci(..., config=BootstrapCIConfig(...))`
  - `evaluation(EvaluationConfig(...))` or legacy `evaluation(data, modelid, ...)`
  - `load_evaluation_data(LoadConfig(...))` or legacy signature

- **Test Coverage**:
  - New test files: `test_models.py`, `test_evaluation_config_api.py`, `test_io_config_api.py`
  - Bootstrap utils: 90% coverage (was ~70%)
  - IO: 72% coverage (was 67%)

#### What This Achieved
- ✅ 100% backward compatibility maintained
- ✅ Foundation for 40-50% validation code reduction
- ✅ Type-safe config objects with IDE autocomplete
- ✅ +25 new tests (245 → 270)

### ⚠️ Remaining Work

#### Critical Gaps

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `visualisation.py` | 38% | 80% | 🔴 **HIGH** |
| `io/_io_core.py` | 64% | 85% | 🟡 MEDIUM |
| Parallel modules | 0% | 75% or exclude | 🟡 MEDIUM |
| CI validation block | Not migrated | Move to Pydantic | 🔴 **HIGH** |

#### Test Quality Issues
- **59 warnings** in test output (RuntimeWarning, UserWarning)
- Need explicit warning assertions or suppression

---

## Execution Plan

### Phase 1: Visualization Coverage (Target: 38% → 80%)
**Why first**: Largest untested surface, straightforward deterministic branches, high ROI

#### Tasks
1. **`format_evaluation_table` branch coverage**
   ```python
   # tests/visualisation/test_visualisation.py

   def test_order_by_valid_string():
       # order_by="threshold" works

   def test_order_by_valid_list():
       # order_by=["modelid", "threshold"] works

   def test_order_by_missing_column_raises():
       # order_by="invalid_col" → RuntimeError wrapping ValueError
       with pytest.raises(RuntimeError, match="Failed to sort"):
           ...

   def test_order_by_invalid_type_raises():
       # order_by=123 → RuntimeError wrapping TypeError

   def test_ci_column_true_creates_separate_columns():
       # ci_column=True: creates "AUROC CI" column, drops Lower/Upper

   def test_ci_column_false_inlines_ci_text():
       # ci_column=False: inlines as "0.85 [0.82 - 0.88]"

   def test_float_columns_validation_and_filtering():
       # Custom float_columns list works, invalid columns handled
   ```

2. **Plotting functions**
   ```python
   def test_plot_roc_curve_with_axes():
       # Provide ax, ensure it's used

   def test_plot_roc_curve_without_axes():
       # ax=None creates new figure

   def test_plot_roc_curve_label_with_model_name():
       # label includes model_name

   def test_plot_roc_curve_label_without_model_name():
       # label is default

   def test_plot_roc_curve_bad_input_shapes():
       # Mismatched y_true/y_pred lengths raise ValueError

   def test_precision_recall_curve_variants():
       # Same scenarios as ROC

   @pytest.mark.skipif(matplotlib_available, reason="Need unavailable matplotlib")
   def test_plot_import_error(monkeypatch):
       # Monkeypatch availability flag to trigger ImportError path
   ```

**Exit criteria**: `visualisation.py` ≥80% coverage, all tests pass in default and all envs

---

### Phase 2: Pydantic Refactor Completion

#### Task 1: Move CI Validation to Pydantic
**Current problem**: Lines 340-443 in `evaluation.py` run for both config and legacy paths

**Solution**: Add model validators
```python
# pysalient/evaluation/_models.py

class ConfidenceIntervalConfig(BaseConfig):
    calculate_au_ci: bool = False
    calculate_threshold_ci: bool = False
    threshold_ci_method: ThresholdCIMethod = ThresholdCIMethod.BOOTSTRAP
    alpha: float = Field(default=0.05, gt=0, lt=1)
    bootstrap_rounds: int = Field(default=1000, ge=100)
    bootstrap_seed: int | None = None

    @model_validator(mode='after')
    def validate_ci_dependencies(self) -> Self:
        """Validate CI calculation dependencies."""
        if not (self.calculate_au_ci or self.calculate_threshold_ci):
            return self

        # Validate bootstrap parameters when bootstrap method selected
        if self.calculate_threshold_ci and self.threshold_ci_method == ThresholdCIMethod.BOOTSTRAP:
            if self.bootstrap_rounds < 100:
                warnings.warn(
                    "bootstrap_rounds < 100 may produce unreliable CIs",
                    UserWarning
                )

        # Add other CI dependency validations here
        # (Move logic from evaluation.py:340-443)

        return self
```

**Then in `evaluation.py`**:
```python
# Remove lines 340-443 for config path
if not input_is_config:
    # Keep legacy validation for backward compatibility
    if calculate_au_ci or calculate_threshold_ci:
        # ... existing validation
```

**Expected impact**: Reduce evaluation.py by ~100 lines, achieve 40-50% validation reduction goal

#### Task 2: Extract `_evaluation_impl(config)`
**Goal**: Fully separate config-first logic from legacy path

```python
def evaluation(
    data: pa.Table | EvaluationConfig,
    modelid: str | None = None,
    ...
) -> pa.Table:
    """Public API with dual interface."""
    if isinstance(data, EvaluationConfig):
        return _evaluation_impl(data)

    # Legacy path: validate and convert to config
    # ... existing validation for legacy params
    config = EvaluationConfig(...)
    return _evaluation_impl(config)


def _evaluation_impl(config: EvaluationConfig) -> pa.Table:
    """Internal implementation - validation done by Pydantic."""
    # Direct access to config fields, no validation needed
    # Extract existing calculation logic here
```

---

### Phase 3: IO Robustness (Target: 64% → 85%)

#### Test Plan
```python
# tests/io/test_io_core.py

def test_load_data_to_pyarrow_dataframe_with_conflicting_source_type():
    # DataFrame + source_type="csv" → error

def test_load_data_to_pyarrow_unknown_extension():
    # "file.xyz" → inference error

def test_load_data_to_pyarrow_read_failure(monkeypatch):
    # Mock csv/parquet read to raise exception

def test_perform_aggregation_missing_columns():
    # Aggregation column not in table → error

def test_perform_aggregation_pandas_failure(monkeypatch):
    # Mock pandas aggregation failure

def test_export_evaluation_results_invalid_format():
    # format="invalid" → ValueError

def test_export_evaluation_results_missing_output_path():
    # format="csv" without output_path → ValueError

def test_export_formatted_results_invalid_styler_type():
    # styler=123 → TypeError

def test_export_formatted_results_all_format_branches():
    # Test "dataframe", "csv", "parquet" branches
```

**Exit criteria**: `io.py` ≥85%, `_io_core.py` ≥85%

---

### Phase 4: Parallel Modules Policy

**Decision required**: Test or exclude?

#### Option A (Recommended): Test parallel modules
```python
# tests/evaluation/test_bootstrap_utils_parallel.py

def test_parallel_bootstrap_small_deterministic_workload():
    # n_jobs=2, fixed seed, tiny dataset

def test_parallel_worker_distribution():
    # Verify split behavior for different n_jobs

def test_parallel_metric_failure_returns_nan():
    # Broken metric_func → NaN handling
```

**Target**: Parallel modules ≥75% each → total coverage ~75%

#### Option B: Exclude and document
```toml
# pyproject.toml or .coveragerc
[tool.coverage.run]
omit = [
    "pysalient/evaluation/*_parallel.py",
]
```

Add to README:
```markdown
## Experimental Features
Parallel evaluation modules are currently experimental and excluded from coverage gates.
Owner: @maintainer | Sunset: 2026-Q2
```

**Choose one and implement by end of Phase 4.**

---

### Phase 5: Test Quality & CI Integration

#### Task 1: Clean Test Warnings
```python
# Suppress expected warnings
@pytest.mark.filterwarnings("ignore:Bootstrap CI calculation failed")
def test_bootstrap_edge_case():
    ...

# Or assert warnings explicitly
def test_low_bootstrap_rounds_warns():
    with pytest.warns(UserWarning, match="bootstrap_rounds.*100"):
        config = ConfidenceIntervalConfig(bootstrap_rounds=50)
```

**Goal**: Reduce 59 warnings to <10

#### Task 2: Enable Coverage Gate
```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage
  run: pixi run pytest --cov=pysalient --cov-report=term-missing --cov-fail-under=65 tests/

- name: Run tests in all environments (informational)
  run: pixi run -e all pytest --cov=pysalient --cov-report=term-missing tests/
  continue-on-error: true  # Non-blocking until stable
```

#### Task 3: Ratchet Coverage Targets
- After Phase 1 (viz): `--cov-fail-under=68`
- After Phase 2 (IO): `--cov-fail-under=72`
- After Phase 3 (parallel): `--cov-fail-under=75`

---

## Verification Checklist

Before each commit:
- [ ] `pixi run test` passes (all tests green)
- [ ] `pixi run coverage` shows expected increase
- [ ] `pixi run lint` passes (no new lint errors)
- [ ] Backward compatibility: existing code examples still work
- [ ] No new warnings introduced (or explicitly handled)

---

## Success Metrics

| Metric | Baseline | Current | Target | Status |
|--------|----------|---------|--------|--------|
| **Total Coverage** | 61% | 65% | 75% | 🟡 In Progress |
| **Tests Passing** | 245 | 270 | - | ✅ |
| **Visualization** | 38% | 38% | 80% | 🔴 Blocked |
| **Bootstrap Utils** | ~70% | 90% | - | ✅ |
| **IO** | 67% | 72% | 85% | 🟡 In Progress |
| **Validation Code** | ~200 checks | ~180 | ~100-120 | 🟡 40% complete |
| **Test Warnings** | ~50 | 59 | <10 | 🔴 Increased |
| **CI Gate Active** | ❌ | ❌ | ✅ | 🔴 Not Started |

---

## Quick Reference

### Running Tests
```bash
# Full test suite
pixi run test

# With coverage
pixi run coverage

# All environments (includes plotting tests)
pixi run -e all test
pixi run -e all coverage

# Type checking
pixi run typecheck
```

### Adding Tests (TDD Workflow)
```bash
# 1. Write failing test
vim tests/visualisation/test_visualisation.py

# 2. Run specific test
pixi run pytest tests/visualisation/test_visualisation.py::test_name -v

# 3. Implement minimal fix
vim pysalient/visualisation/visualisation.py

# 4. Verify fix
pixi run pytest tests/visualisation/test_visualisation.py::test_name -v

# 5. Run full suite
pixi run test
```

---

## Priority Roadmap

### This Week
1. ✅ Commit current Pydantic work (Phases 1-2)
2. 🎨 Start Phase 1: Add visualization tests (38% → 80%)
3. 🚦 Enable CI coverage gate at 65%

### Next Week
4. ♻️ Move CI validation to Pydantic validators
5. 🧪 Complete IO robustness tests (64% → 85%)
6. 🔕 Clean test warnings (<10 target)

### Following Week
7. ⚡ Parallel modules: decide policy and implement
8. 📈 Ratchet coverage gate to 75%
9. 📚 Add documentation for config APIs

---

## Notes

- **Backward Compatibility**: Non-negotiable. All existing code must work.
- **TDD Discipline**: Write failing test first, then minimal implementation.
- **Coverage Philosophy**: Cover risk areas (error paths, edge cases, branches), not just line count.
- **Parallel modules**: Don't let this linger - test or exclude within 2 weeks.

---

*Last Updated: 2026-02-13*
*Next Review: After Phase 1 completion*
