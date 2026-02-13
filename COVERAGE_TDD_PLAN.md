# Coverage Improvement Plan (TDD)

## Scope and Objective
This plan defines a test-first path to increase and stabilize code coverage in `pysalient` while preserving behavior and keeping tests maintainable.

Primary objective:
- Raise total coverage from current baseline to >=75% in default environment.
- Eliminate major untested risk areas in visualization, IO error handling, and parallel evaluation modules.

Secondary objective:
- Make coverage checks a first-class part of local and CI workflows.

## Baseline (Measured 2026-02-13)
Commands executed:
- `pixi run test`
- `pixi run coverage`
- `pixi run -e all coverage`

Results:
- Default env (`pixi run coverage`):
  - Tests: `245 passed, 6 skipped`
  - Total coverage: `61%`
- All env (`pixi run -e all coverage`):
  - Tests: `249 passed, 2 skipped`
  - Total coverage: `64%`

Low-coverage hotspots:
- `pysalient/evaluation/_bootstrap_utils_parallel.py`: `0%`
- `pysalient/evaluation/_evaluation_process_parallel.py`: `0%`
- `pysalient/visualisation/visualisation.py`: `38%` (default), `59%` (all)
- `pysalient/io/_io_core.py`: `64%`
- `pysalient/io/io.py`: `67%`
- `pysalient/evaluation/evaluation.py`: `76%`
- `pysalient/evaluation/_evaluation_process.py`: `77%`

## Guiding TDD Principles
- For every gap, start with a failing test that captures expected behavior.
- Prefer behavior-level assertions over implementation details.
- Add regression tests for every bug or edge case discovered.
- Keep tests deterministic:
  - Seed random paths.
  - Avoid timing-sensitive assertions.
  - Use focused fixtures with minimal data.
- Only update production code after a failing test demonstrates the gap.

## Coverage Policy Decisions (Required Early)
1. Parallel modules policy:
- Option A (recommended): Treat parallel modules as supported functionality and test them directly.
- Option B: Mark them experimental and exclude from coverage gates until stabilized.

2. Environment policy for coverage gate:
- Gate on `default` only (fast, reliable).
- Run informational coverage on `all` for plotting paths.

3. Skip policy:
- Any skipped test must have explicit reason and owner.
- Reduce dependency-based skips by ensuring required deps in the right pixi environment.

## Execution Plan

### Phase 0: Workflow hardening (completed)
- Added `pytest-cov` dependency to dev feature.
- Added `pixi` coverage task:
  - `coverage = "pytest --cov=pysalient --cov-report=term-missing tests/"`

Exit criteria:
- `pixi run coverage` works locally and in CI environments.

### Phase 1: Visualisation module (highest ROI)
Target file:
- `pysalient/visualisation/visualisation.py`

Why first:
- Large uncovered surface with straightforward deterministic branches.

Test plan:
1. `format_evaluation_table` branch coverage
- `order_by`:
  - valid string
  - valid list
  - missing column -> `RuntimeError` wrapping `ValueError`
  - invalid type -> `RuntimeError` wrapping `TypeError`
- CI formatting modes:
  - `ci_column=True` creates `"<Metric> CI"` columns and drops raw CI bounds.
  - `ci_column=False` inlines CI text in metric columns.
- `float_columns` validation and filtering behavior.
- column reorder logic with metrics + CI columns.
- fallback behavior when formatting/reindex operations encounter edge inputs.

2. Plotting functions
- `plot_roc_curve`:
  - with and without provided axes
  - label behavior with and without `model_name`
  - shape/type validation via bad inputs
- `plot_precision_recall_curve`:
  - same scenarios as ROC
- Import error pathways:
  - monkeypatch availability flags to validate `ImportError` paths.

Expected impact:
- Raise `visualisation.py` toward >=80%.

Exit criteria:
- New tests pass in `default` and `all` as appropriate.
- `visualisation.py` >=80% coverage.

### Phase 2: IO robustness (second ROI)
Target files:
- `pysalient/io/_io_core.py`
- `pysalient/io/io.py`
- `pysalient/io/_io_utils.py`

Test plan:
1. `_load_data_to_pyarrow`
- DataFrame + conflicting `source_type` mismatch.
- unknown file extension inference error.
- forced CSV/Parquet read exceptions (mock read functions).
- unsupported source type path.

2. `_perform_aggregation`
- missing aggregation columns.
- pandas aggregation failure handling.
- conversion back to pyarrow failure handling.
- multi-column aggregation ordering guarantees.

3. `export_evaluation_results`
- invalid `results_table` type.
- invalid format.
- missing output path for file formats.
- csv/parquet writer kwargs pass-through behavior.

4. `export_formatted_results`
- invalid `styler` type.
- all format branches (`dataframe`, `csv`, `parquet`).
- output path requirement.

Expected impact:
- Raise `io.py` and `_io_core.py` to >=85%.

Exit criteria:
- IO test additions are green and deterministic.
- `io.py` and `_io_core.py` >=85% coverage.

### Phase 3: Parallel evaluation modules (risk-heavy)
Target files:
- `pysalient/evaluation/_bootstrap_utils_parallel.py`
- `pysalient/evaluation/_evaluation_process_parallel.py`

Test plan:
1. Unit tests around input validation and fallback behavior.
2. Deterministic bootstrap tests with fixed seed and small rounds.
3. Worker distribution logic tests (`n_jobs`, threshold counts, split behavior).
4. Monkey-patch enablement path tests for `enable_parallel_evaluation`.
5. Error-path tests where metric functions fail and return NaNs.

Alternative (if excluded):
- Add explicit `omit` entries to coverage config and document rationale, owner, and sunset date.

Expected impact:
- If tested: substantial increase in total coverage and confidence.
- If excluded: clearer policy but lower functional confidence for parallel paths.

Exit criteria:
- Either modules reach >=75% each, or formal exclusion policy is merged with owner/date.

### Phase 4: Evaluation edge cases and warning paths
Target files:
- `pysalient/evaluation/evaluation.py`
- `pysalient/evaluation/_evaluation_process.py`
- `pysalient/evaluation/_bootstrap_utils.py`

Test plan:
- warning pathways for low bootstrap rounds and undefined class metrics.
- force-eval threshold guardrails and exact boundary behavior.
- time-to-event edge combinations currently not explicitly covered.
- bootstrap failure-rate warning thresholds and all-fail behavior.

Expected impact:
- Raise these modules from mid/high-70s to >=85% while locking down behavior.

Exit criteria:
- Key warning and boundary branches covered.
- No behavior regressions in existing suite.

## Milestones and Targets
- Milestone 1 (after Phase 1): total >=68%
- Milestone 2 (after Phase 2): total >=72%
- Milestone 3 (after Phase 3): total >=75%
- Milestone 4 (after Phase 4): total >=80% (stretch)

## Proposed CI Gating
1. Required checks:
- `pixi run test`
- `pixi run coverage`

2. Coverage thresholds (incremental):
- Initial gate: `--cov-fail-under=65`
- After Milestone 2: `--cov-fail-under=72`
- After Milestone 3: `--cov-fail-under=75`

3. Optional informational job:
- `pixi run -e all coverage` (non-blocking until stable)

## Task Breakdown (Implementation Queue)
1. Add tests for `format_evaluation_table` ordering + CI mode branches.
2. Add tests for plotting error and labeling branches.
3. Add IO export branch tests.
4. Add `_perform_aggregation` failure-path tests.
5. Add parallel module validation tests.
6. Decide test-vs-exclude policy for parallel modules and implement.
7. Add first coverage gate (`cov-fail-under=65`).

## Risks and Mitigations
- Risk: brittle formatting assertions in HTML output.
  - Mitigation: assert structural invariants and key tokens, not full HTML snapshots.
- Risk: multiprocessing tests flaky in constrained CI.
  - Mitigation: use tiny deterministic workloads and explicit seeds.
- Risk: optional dependency drift changes skip behavior.
  - Mitigation: separate default and all-env expectations in CI.

## Definition of Done
- Coverage command reproducible via `pixi run coverage`.
- Coverage trend documented across milestones.
- Agreed coverage gate active in CI.
- High-risk modules (visualisation, IO, parallel or explicit exclusion policy) are resolved.
- Test suite remains stable and fast enough for regular TDD cycles.
