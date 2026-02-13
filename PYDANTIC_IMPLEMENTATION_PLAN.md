# Pydantic Implementation Plan for pySALIENT Type Safety

## Context

**Problem**: pySALIENT has strong type annotations (95%+ coverage) but relies on ~200 manual validation checks scattered across the codebase. The `evaluation()` function alone has 22-26 parameters with 136 validation checks. Pydantic is listed as a dependency but completely unused.

**Goal**: Introduce Pydantic models to consolidate validation, reduce code complexity by 40-50%, improve maintainability, and enable future config file support—all while maintaining backward compatibility.

**Impact**:
- Reduce validation code from ~200 manual checks to ~100 declarative validations
- Group related parameters (e.g., 8 CI params → `CIConfig` object)
- Enable IDE autocomplete with string enums (TimeUnit, ThresholdCIMethod)
- Foundation for YAML/JSON config loading for reproducible research

## Key Discovery from ty Review

Running `ty check pysalient` revealed 33 type issues, validating the need for better type safety:
- Callable `__name__` access issues (bootstrap_utils.py)
- Invalid None assignments in try/except blocks (evaluation.py)
- Missing validation for dict structures (read_options, time_to_event_cols)

## Implementation Phases

### Phase 1: Foundation (Create Models)

**Create new files:**

1. **pysalient/_shared_models.py** (~100 lines)
   ```python
   from enum import Enum
   from pydantic import BaseModel, ConfigDict

   class TimeUnit(str, Enum):
       SECOND = "second"; HOUR = "hour"; DAY = "day"
       # ... all 19 variants including abbreviations

   class ThresholdCIMethod(str, Enum):
       BOOTSTRAP = "bootstrap"; NORMAL = "normal"
       WILSON = "wilson"; AGRESTI_COULL = "agresti-coull"

   class BaseConfig(BaseModel):
       model_config = ConfigDict(
           arbitrary_types_allowed=True,  # For PyArrow types
           validate_assignment=True,
       )
   ```

2. **pysalient/evaluation/_models.py** (~400 lines)
   ```python
   # Core groupings from evaluation() function
   class ThresholdConfig(BaseConfig):
       values: list[float] | tuple[float, float, float]
       force_threshold_zero: bool = True

   class TimeToEventConfig(BaseConfig):
       event_columns: dict[str, str]
       aggregation_func: str = "median"
       time_unit: TimeUnit = TimeUnit.HOUR
       fillna: float | None = None

   class ConfidenceIntervalConfig(BaseConfig):
       calculate_au_ci: bool = False
       calculate_threshold_ci: bool = False
       threshold_ci_method: ThresholdCIMethod = ThresholdCIMethod.BOOTSTRAP
       alpha: float = Field(default=0.05, gt=0, lt=1)
       bootstrap_rounds: int = Field(default=1000, ge=100)
       bootstrap_seed: int | None = None

   class EvaluationConfig(BaseConfig):
       data: pa.Table
       modelid: str
       filter_desc: str
       thresholds: ThresholdConfig
       time_to_event: TimeToEventConfig | None = None
       confidence_intervals: ConfidenceIntervalConfig = Field(default_factory=...)
       # ... grouped parameters replace 22 flat params
   ```

3. **pysalient/io/_models.py** (~250 lines)
   ```python
   class ColumnConfig(BaseConfig):
       y_proba_col: str
       y_label_col: str
       timeseries_col: str
       aggregation_cols: str | list[str]

   class LoadConfig(BaseConfig):
       source: str | pd.DataFrame
       columns: ColumnConfig
       read_options: ReadOptions = Field(default_factory=ReadOptions)
       # Replaces load_evaluation_data's 9 params + **kwargs
   ```

**Testing**: Write 50+ model validation tests covering:
- Valid inputs create models successfully
- Invalid inputs raise ValidationError with clear messages
- Field validators work (e.g., alpha range, numpy function names)
- Cross-field validation (CI dependencies)

### Phase 2: Internal Adoption (Proof of Concept)

**Refactor**: `pysalient/evaluation/_bootstrap_utils.py`

Current signature:
```python
def calculate_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    verbosity: int = 0,
) -> tuple[float, float]:
```

New approach (maintain backward compatibility):
```python
class BootstrapCIConfig(BaseConfig):
    n_rounds: int = Field(default=1000, ge=100)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    seed: int | None = None
    verbosity: int = 0

def calculate_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    config: BootstrapCIConfig | None = None,
    # Legacy params for backward compatibility
    n_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
    verbosity: int = 0,
) -> tuple[float, float]:
    if config is None:
        config = BootstrapCIConfig(
            n_rounds=n_rounds, alpha=alpha, seed=seed, verbosity=verbosity
        )
    # Use config.n_rounds, config.alpha, etc.
    # Remove manual validation (lines with isinstance checks)
```

**Benefit**: Reduces 10-15 manual validation checks, proves Pydantic integration works.

### Phase 3: Evaluation Module (Main Target)

**Refactor**: `pysalient/evaluation/evaluation.py` (evaluation.py:65-91, 220-495)

**Current state**: 22-26 parameters, 136 validation checks (lines 220-495)

**Approach**: Dual API with wrapper

```python
def evaluation(
    # Accept either config object OR individual params
    data: pa.Table | EvaluationConfig,
    modelid: str | None = None,
    filter_desc: str | None = None,
    thresholds: list[float] | tuple | None = None,
    # ... all 22 existing params with defaults
) -> pa.Table:
    """
    Public API - supports both legacy and config-based usage.

    Examples:
        # Legacy API (backward compatible)
        >>> results = evaluation(
        ...     data=table, modelid="v1", filter_desc="test",
        ...     thresholds=[0.5, 0.9], calculate_au_ci=True
        ... )

        # New config API (recommended)
        >>> config = EvaluationConfig(
        ...     data=table,
        ...     modelid="v1",
        ...     filter_desc="test",
        ...     thresholds=ThresholdConfig(values=[0.5, 0.9]),
        ...     confidence_intervals=ConfidenceIntervalConfig(calculate_au_ci=True)
        ... )
        >>> results = evaluation(config)
    """
    # Detect config object
    if isinstance(data, EvaluationConfig):
        return _evaluation_impl(data)

    # Legacy path: convert params to config
    config = EvaluationConfig(
        data=data,
        modelid=modelid,
        filter_desc=filter_desc,
        thresholds=ThresholdConfig(values=thresholds),
        # ... map all params to config groups
    )
    return _evaluation_impl(config)


def _evaluation_impl(config: EvaluationConfig) -> pa.Table:
    """Internal implementation - validation already done by Pydantic."""
    # Remove lines 220-495 (275 lines of validation)
    # Access config.data, config.modelid, config.thresholds.values, etc.
    # Existing calculation logic remains unchanged
```

**Impact**:
- Lines 220-495 (275 lines of validation) → ~50 lines using config
- 50% reduction in validation code
- Both APIs produce identical results (verified by tests)

### Phase 4: IO Module

**Refactor**: `pysalient/io/io.py` (io.py:19-36)

Similar dual-API approach for `load_evaluation_data()`:
- Current: 9 params + **kwargs, 44 validation checks
- New: `LoadConfig` object with nested configs
- Maintain backward compatibility

### Phase 5: Integration & ty Fixes

**Address ty check issues** identified earlier:
1. Fix `metric_func.__name__` access (bootstrap_utils.py:125, 132, 140, etc.)
   - Use `getattr(metric_func, '__name__', '<callable>')` or validate function type
2. Fix None assignments (evaluation.py:46, 51)
   - Use proper Optional types with conditional imports
3. Add matplotlib to optional dependencies properly

**Run**: `pixi run typecheck` to verify all 33 issues resolved

## Critical Files

1. `/home/schnetlerr/dev/pysalient/pysalient/evaluation/evaluation.py` - Main refactoring target; 22-26 params → grouped config (lines 65-91, 220-495)
2. `/home/schnetlerr/dev/pysalient/pysalient/io/io.py` - Secondary target; 9 params + **kwargs → LoadConfig (lines 19-36)
3. `/home/schnetlerr/dev/pysalient/pysalient/_shared_models.py` (NEW) - Foundation: BaseConfig, TimeUnit, ThresholdCIMethod enums
4. `/home/schnetlerr/dev/pysalient/pysalient/evaluation/_models.py` (NEW) - EvaluationConfig and sub-configs (ThresholdConfig, TimeToEventConfig, CIConfig)
5. `/home/schnetlerr/dev/pysalient/pysalient/evaluation/_bootstrap_utils.py` - Phase 2 proof-of-concept (lines 85-165)

## Verification Strategy

**Per-phase testing**:
1. Model validation tests (pytest): 50+ tests for pydantic models
2. Integration tests: Both legacy and config APIs produce identical results
3. Backward compatibility: All existing tests pass without modification
4. Performance: < 5% overhead (benchmark critical paths)
5. Type checking: `ty check pysalient` shows reduction in errors

**Success metrics**:
- ✅ 40-50% reduction in validation code (200 → ~100-120 checks)
- ✅ 100% backward compatibility (all existing tests pass)
- ✅ 33 ty errors reduced to 0
- ✅ IDE autocomplete works for string enums (TimeUnit.HOUR vs "houuur")

## Trade-offs

**Benefits**:
- Consolidated validation (DRY principle)
- Better error messages (Pydantic shows field paths)
- Self-documenting code (Field descriptions)
- Foundation for config files (YAML/JSON loading)

**Costs**:
- Pydantic dependency (already present, just unused)
- Dual code paths during transition
- Learning curve for contributors

**Risk mitigation**:
- Gradual rollout (Phase 2 proves concept before Phase 3)
- Maintain backward compatibility (no breaking changes)
- Comprehensive testing (150+ new tests)
- Clear migration examples in docs

## Future Enhancements

Once established:
- Config file support: `EvaluationConfig.parse_file("config.yaml")`
- Schema export for documentation
- CLI config validation: `pysalient validate-config config.yaml`
