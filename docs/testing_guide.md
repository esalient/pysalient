# Testing Guide

This project uses `pytest` for test execution, with Pandera schemas and Hypothesis
property-based tests for data-driven validation.

## Running Tests

- Run the standard suite:
  - `pixi run test`
- Run tests with coverage details:
  - `pixi run coverage`
- Run Hypothesis/Pandera-focused tests in the dedicated environment:
  - `pixi run -e pandera-testing pytest`

## Hypothesis Profiles

Profiles are configured in `tests/conftest.py` and loaded from
`HYPOTHESIS_PROFILE`:

- `dev` (default): faster local feedback
  - `max_examples=50`
- `ci`: more thorough checking in CI
  - `max_examples=200`

Examples:

- Local default:
  - `pixi run -e pandera-testing pytest tests/evaluation/test_evaluation_properties.py`
- CI-style locally:
  - `HYPOTHESIS_PROFILE=ci pixi run -e pandera-testing pytest`

## Reading Hypothesis Failures

When a property test fails, Hypothesis prints:

- A minimal counterexample (`Falsifying example`) that reproduces the bug.
- A seed to reproduce (`--hypothesis-seed=...`).

Typical workflow:

1. Re-run exactly with the provided seed.
2. Debug using the shrunk counterexample.
3. Fix code or relax/tighten the property if the invariant was incorrect.

## Adding a New Property Test

Template:

```python
from hypothesis import given, settings
import hypothesis.strategies as st

@settings(max_examples=25, deadline=None)
@given(data=my_strategy())
def test_some_invariant(data):
    result = my_function(data)
    assert invariant_holds(result)
```

Guidelines:

- Assert invariants (always true), not hard-coded outputs from one dataset.
- Use `assume(...)` sparingly; high filtering makes tests slow and less effective.
- Use numeric tolerances for floating-point boundaries.

## Adding a New Pandera Schema

Template:

```python
import pandera as pa

my_schema = pa.DataFrameSchema(
    {
        "id": pa.Column(str, nullable=False),
        "score": pa.Column(float, checks=pa.Check.in_range(0.0, 1.0)),
    },
    strict=False,
    coerce=True,
)
```

Guidelines:

- Put shared schemas in `tests/schemas.py`.
- Add acceptance and rejection tests in `tests/test_schemas.py`.
- Keep checks aligned with library contracts (types, ranges, nullability).

## Useful References

- Pandera docs: <https://pandera.readthedocs.io/>
- Hypothesis docs: <https://hypothesis.readthedocs.io/>
