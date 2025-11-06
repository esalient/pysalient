"""
Pandera schemas for pysalient test data validation.

This module defines schemas using pandera for validating test data structures,
replacing the need for static parquet files in tests.
"""

import pandas as pd
import pandera as pa

# Use DataFrameSchema for better compatibility
evaluation_data_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba_1": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)


# Minimal schema for basic testing
minimal_evaluation_data_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_probability": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
    },
    coerce=True,
    strict=True
)


evaluation_with_temporal_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(
            "datetime64[ns]",
            pa.Check.in_range(
                min_value=pd.Timestamp("2020-01-01"),
                max_value=pd.Timestamp("2025-12-31 23:59:59")
            ),
            nullable=False
        ),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba_1": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_with_temporal_schema.__doc__ = """
Schema for evaluation data with datetime timestamps.

This schema validates evaluation data identical to evaluation_data_schema,
but with event_timestamp as a pandas datetime64 type instead of float.
This enables temporal analysis and ensures timestamps are within a reasonable
range for medical data (2020-01-01 to 2025-12-31).

Columns:
    encounter_id: String identifier for each encounter (required)
    event_timestamp: Datetime timestamp validated to be between 2020-01-01 and
        2025-12-31 (required)
    culture_event: Binary indicator (0.0 or 1.0) for culture events (optional)
    suspected_infection: Binary indicator (0.0 or 1.0) for suspected infections (optional)
    true_label: Binary ground truth label (0 or 1) (required)
    prediction_proba_1: Model probability prediction, range [0.0, 1.0] (required)
    prediction_proba_2: Alternative model probability prediction, range [0.0, 1.0] (optional)

Schema properties:
    coerce: True - automatically coerce types to match schema
    strict: False - allow additional columns not defined in schema
"""


io_csv_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "timestamp": pa.Column(str, nullable=False),
        "label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "probability": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
io_csv_input_schema.__doc__ = """
Schema for validating raw CSV input data before transformation.

This schema is designed to validate data as it comes from CSV files, where all
values are typically strings. The coerce=True parameter enables automatic type
conversion from string representations (e.g., "0", "1", "0.85") to the proper
types (int, float). This is essential for CSV input validation as CSV readers
initially parse all data as strings.

The schema enforces data quality constraints while allowing flexibility for
typical CSV variations through strict=False, which permits additional columns
commonly found in raw data exports.

Columns:
    encounter_id: String identifier for each encounter (required)
    timestamp: String timestamp from CSV - accepts string representations (required)
    label: Binary ground truth label (0 or 1), coerced from string "0"/"1" to int (required)
    probability: Model probability prediction, range [0.0, 1.0], coerced from string to float (required)
    culture_event: Binary indicator (0.0 or 1.0) for culture events, coerced from string to float (optional)
    suspected_infection: Binary indicator (0.0 or 1.0) for suspected infections, coerced from string to float (optional)
    prediction_proba_2: Alternative model probability prediction, range [0.0, 1.0], coerced from string to float (optional)

Schema properties:
    coerce: True - automatically coerce string values to proper types (essential for CSV)
    strict: False - allow extra columns typical in raw CSV exports

Use cases:
    - Validating CSV files before data transformation pipelines
    - Ensuring raw data meets quality standards at ingestion
    - Type-safe conversion from CSV string data to proper Python types
"""


io_parquet_input_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "probability": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
io_parquet_input_schema.__doc__ = """
Schema for validating raw Parquet input data before transformation.

This schema is designed to validate data as it comes from Parquet files, which
preserve numeric types during serialization unlike CSV. Parquet's columnar
storage format maintains type information, so this schema expects properly typed
numeric values (float, int) rather than string representations.

The coerce=True parameter allows some flexibility for type conversions while
maintaining strict validation of data constraints. The strict=False setting
permits additional metadata columns commonly found in raw Parquet exports.

This schema is structurally similar to io_csv_input_schema but differs in the
timestamp column type: Parquet preserves the numeric float type directly,
whereas CSV stores it as a string requiring parsing.

Columns:
    encounter_id: String identifier for each encounter (required)
    timestamp: Numeric timestamp as float, must be >= 0.0 (required)
    label: Binary ground truth label (0 or 1) as integer (required)
    probability: Model probability prediction, range [0.0, 1.0] as float (required)
    culture_event: Binary indicator (0.0 or 1.0) for culture events as float (optional)
    suspected_infection: Binary indicator (0.0 or 1.0) for suspected infections as float (optional)
    prediction_proba_2: Alternative model probability prediction, range [0.0, 1.0] as float (optional)

Schema properties:
    coerce: True - allow type coercion for compatibility
    strict: False - allow extra metadata columns in raw Parquet files

Use cases:
    - Validating Parquet files before data transformation pipelines
    - Ensuring type preservation and data quality from binary formats
    - Efficient validation of large datasets with maintained type information
"""


time_to_event_data_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "time_to_event": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "event_occurred": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "baseline_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "prediction_proba": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "risk_score": pa.Column(float, nullable=True, required=False),
        "censoring_reason": pa.Column(str, nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
time_to_event_data_schema.__doc__ = """
Schema for time-to-event (survival analysis) data validation.

This schema validates data used in survival analysis and time-to-event studies,
ensuring that temporal data, event indicators, and predictions meet quality
standards for statistical modeling. It enforces constraints on time values
(non-negative), event flags (binary), and probability predictions while allowing
optional risk scores and censoring information.

The schema supports both uncensored events (event_occurred=1) where the outcome
was observed and censored observations (event_occurred=0) where the subject was
lost to follow-up or the study ended before the event occurred. This is essential
for Kaplan-Meier estimation, Cox proportional hazards models, and other survival
analysis techniques.

Columns:
    encounter_id: String identifier for each encounter/subject (required)
    time_to_event: Time until event or censoring in hours/days, must be >= 0.0 (required)
    event_occurred: Binary indicator (0 or 1) - 1 if event happened, 0 if censored (required)
    baseline_timestamp: Starting time reference, must be >= 0.0 (required)
    prediction_proba: Predicted probability of event occurrence, range [0.0, 1.0] (required)
    risk_score: Optional risk score for the subject (optional)
    censoring_reason: Optional string describing why observation was censored (optional)

Schema properties:
    coerce: True - automatically coerce types to match schema
    strict: False - allow additional columns not defined in schema

Use cases:
    - Validating survival analysis datasets before modeling
    - Ensuring time-to-event data meets statistical requirements
    - Quality control for clinical trial and observational study data
    - Preprocessing for Kaplan-Meier curves and Cox regression
"""


evaluation_results_schema = pa.DataFrameSchema(
    {
        "metric_name": pa.Column(str, nullable=False),
        "metric_value": pa.Column(float, nullable=False),  # No range constraint - metrics can exceed [0,1]
        "ci_lower": pa.Column(float, nullable=False),  # No range constraint - depends on metric type
        "ci_upper": pa.Column(float, nullable=False),  # No range constraint - depends on metric type
        "n_samples": pa.Column(int, pa.Check.greater_than(0), nullable=False),
        "threshold": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
        "prevalence": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
        "model_name": pa.Column(str, nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_results_schema.__doc__ = """
Schema for validating evaluation output/results data containing computed metrics and statistics.

This schema validates the output of model evaluation pipelines, ensuring that
computed metrics, confidence intervals, and associated statistics meet quality
standards. It enforces range constraints on metric values and confidence bounds
while allowing flexibility for optional metadata fields.

The schema supports both simple metric reporting (name, value, confidence intervals)
and enriched results that include threshold information, prevalence rates, and
model identification. This flexibility enables validation across different
evaluation contexts from basic performance reporting to detailed comparative analyses.

Columns:
    metric_name: String identifier for the metric (e.g., "AUROC", "log_loss", "MSE") (required)
    metric_value: Computed metric value (no range constraint - some metrics like MSE, log loss can exceed 1.0) (required)
    ci_lower: Lower bound of confidence interval (range depends on metric type) (required)
    ci_upper: Upper bound of confidence interval (range depends on metric type) (required)
    n_samples: Number of samples used in metric computation, must be > 0 (required)
    threshold: Classification threshold used for metric computation, range [0.0, 1.0] (optional)
    prevalence: Class prevalence in the evaluation set, range [0.0, 1.0] (optional)
    model_name: Identifier for the model being evaluated (optional)

Schema properties:
    coerce: True - automatically coerce types to match schema
    strict: False - allow additional metadata columns in evaluation results

Use cases:
    - Validating model evaluation outputs before storage or reporting
    - Ensuring metric computations meet statistical validity requirements
    - Quality control for automated evaluation pipelines
    - Standardizing evaluation results across different models or experiments
"""


evaluation_multi_model_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "model_name": pa.Column(str, nullable=False),
        "prediction_proba": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "model_version": pa.Column(str, nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_multi_model_schema.__doc__ = """
Schema for validating multi-model comparison data on the same dataset.

This schema validates data structured for comparing multiple models on the same
encounters, where each encounter can have multiple rows (one per model). This
design enables head-to-head model performance comparisons on identical test sets
and facilitates analyses like model agreement, calibration differences, and
ensemble method development.

The schema enforces that each row represents a single model's prediction for a
specific encounter at a specific timestamp. Unlike single-model evaluation schemas,
this structure allows the same encounter_id to appear multiple times with different
model_name values, enabling direct comparison of how different models perform on
the exact same clinical scenarios.

Columns:
    encounter_id: String identifier for each encounter (required)
    event_timestamp: Numeric timestamp as float, must be >= 0.0 (required)
    true_label: Binary ground truth label (0 or 1) (required)
    model_name: String identifier for the model making this prediction (required)
    prediction_proba: Model probability prediction, range [0.0, 1.0] (required)
    culture_event: Binary indicator (0.0 or 1.0) for culture events (optional)
    suspected_infection: Binary indicator (0.0 or 1.0) for suspected infections (optional)
    model_version: String identifier for model version (optional)

Schema properties:
    coerce: True - automatically coerce types to match schema
    strict: False - allow additional metadata columns for model comparison analyses

Use cases:
    - Comparing multiple models on the same evaluation dataset
    - Analyzing model agreement and disagreement patterns
    - Developing ensemble methods by combining multiple model predictions
    - Evaluating model calibration differences across different architectures
    - A/B testing different model versions on identical data

Example structure:
    | encounter_id | event_timestamp | true_label | model_name | prediction_proba |
    |--------------|-----------------|------------|------------|------------------|
    | enc_1        | 10.5            | 1          | model_a    | 0.85             |
    | enc_1        | 10.5            | 1          | model_b    | 0.72             |
    | enc_2        | 15.0            | 0          | model_a    | 0.15             |
    | enc_2        | 15.0            | 0          | model_b    | 0.23             |
"""


evaluation_multi_task_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "task_name": pa.Column(str, nullable=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "task_weight": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_multi_task_schema.__doc__ = """
Schema for validating multi-task learning evaluation data.

This schema validates data structured for multi-task learning scenarios where a
single model simultaneously predicts multiple related tasks. Each encounter has
multiple rows (one per task), enabling evaluation of models that learn shared
representations across related prediction problems such as mortality, readmission,
and infection risk.

The schema enforces that each row represents one task's prediction for a specific
encounter at a specific timestamp. Unlike single-task evaluation schemas, this
structure requires the same encounter_id to appear multiple times with different
task_name values, representing the model's predictions for different but related
clinical outcomes on the same patient encounter.

Multi-task learning leverages relationships between tasks to improve generalization
and efficiency compared to training separate single-task models. This schema
supports evaluation of such models by allowing flexible task weighting, task-specific
ground truth labels, and task-specific predictions all within a unified structure.

Columns:
    encounter_id: String identifier for each encounter (required)
    event_timestamp: Numeric timestamp as float, must be >= 0.0 (required)
    task_name: String identifier for the task (e.g., "infection", "mortality", "readmission") (required)
    true_label: Binary ground truth label (0 or 1) for this specific task (required)
    prediction_proba: Model probability prediction for this task, range [0.0, 1.0] (required)
    culture_event: Binary indicator (0.0 or 1.0) for culture events (optional)
    suspected_infection: Binary indicator (0.0 or 1.0) for suspected infections (optional)
    task_weight: Weight or importance of this task, must be >= 0.0 (optional)

Schema properties:
    coerce: True - automatically coerce types to match schema
    strict: False - allow additional metadata columns for multi-task analyses

Use cases:
    - Evaluating multi-task learning models on multiple related prediction problems
    - Analyzing task-specific performance within a shared model architecture
    - Computing weighted multi-task loss functions and aggregate metrics
    - Studying transfer learning effects between related clinical tasks
    - Optimizing task weights for improved overall model performance

Example structure:
    | encounter_id | event_timestamp | task_name    | true_label | prediction_proba | task_weight |
    |--------------|-----------------|--------------|------------|------------------|-------------|
    | enc_1        | 10.5            | infection    | 1          | 0.85             | 1.0         |
    | enc_1        | 10.5            | mortality    | 0          | 0.12             | 2.0         |
    | enc_1        | 10.5            | readmission  | 1          | 0.68             | 1.5         |
    | enc_2        | 15.0            | infection    | 0          | 0.15             | 1.0         |
    | enc_2        | 15.0            | mortality    | 0          | 0.08             | 2.0         |
    | enc_2        | 15.0            | readmission  | 0          | 0.23             | 1.5         |
"""


evaluation_data_event_level_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba_1": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_data_event_level_schema.__doc__ = """
Schema for event-level evaluation data with multiple rows per encounter.

This schema validates time-series medical data where each encounter is represented
by multiple rows, with each row capturing a specific event or measurement at a
particular timestamp within that encounter. This is the typical format for raw
clinical data before aggregation.

**Use this schema for:**
    - Validating raw clinical time-series data with multiple observations per patient
    - Multi-step temporal sequences within a single encounter (e.g., vital sign measurements,
      lab results at different time points)
    - Data with repeated events (cultures, suspected infections) at different times
    - Pre-aggregation evaluation where the evaluation() function processes temporal sequences

**Data Structure:**
    - encounter_id: String identifier allowing duplicates - same encounter appears multiple times
    - event_timestamp: Numeric timestamp (float >= 0.0) marking when each event occurred
    - culture_event: Binary indicator (0.0 or 1.0) for culture event presence (optional)
    - suspected_infection: Binary indicator (0.0 or 1.0) for suspected infection (optional)
    - true_label: Binary ground truth label (0 or 1) for the clinical outcome (required)
    - prediction_proba_1: Model probability prediction, range [0.0, 1.0] (required)
    - prediction_proba_2: Alternative model probability prediction, range [0.0, 1.0] (optional)

**Key Characteristics:**
    - Multiple rows per encounter (e.g., 212 rows/encounter in anonymised_sample.parquet)
    - ALLOWS encounter_id duplicates (no uniqueness constraint)
    - Preserves temporal granularity of clinical events
    - pysalient.evaluation.evaluation() accepts this format directly without internal aggregation

**Example:**
    | encounter_id | event_timestamp | true_label | prediction_proba_1 |
    |--------------|-----------------|------------|------------------|
    | ENC001       | 1.0             | 0          | 0.3               |
    | ENC001       | 2.0             | 0          | 0.5               |  <- Same encounter, different time
    | ENC001       | 3.5             | 1          | 0.7               |
    | ENC002       | 1.2             | 1          | 0.8               |

**Schema Properties:**
    - coerce: True (automatic type conversion for data ingestion)
    - strict: False (allow extra columns like metadata or raw measurements)

**Related schemas:**
    - evaluation_data_encounter_level_schema: For aggregated data with one row per encounter
    - evaluation_data_schema: Legacy event-level schema, similar structure
"""


evaluation_data_encounter_level_schema = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(str, nullable=False, unique=True),
        "event_timestamp": pa.Column(float, pa.Check.greater_than_or_equal_to(0.0), nullable=False),
        "culture_event": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "suspected_infection": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, required=False),
        "true_label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "prediction_proba_1": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        "prediction_proba_2": pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=True, required=False),
    },
    coerce=True,
    strict=False
)
evaluation_data_encounter_level_schema.__doc__ = """
Schema for encounter-level evaluation data with one row per encounter (aggregated).

This schema validates aggregated clinical data where each encounter is represented
by exactly one row. This format is produced after temporal aggregation of event-level
data, where multiple observations per encounter have been collapsed into summary
statistics or final values (e.g., final prediction probability, aggregated label).

**Use this schema for:**
    - Validating aggregated/summarized evaluation data with one row per encounter
    - Post-processing clinical datasets after temporal aggregation
    - Comparison data where multiple encounters must be distinct entities
    - Data where temporal sequences have been reduced to single representative values

**Data Structure:**
    - encounter_id: String identifier with UNIQUE constraint - each encounter appears exactly once
    - event_timestamp: Numeric timestamp (float >= 0.0), typically represents aggregation time
    - culture_event: Binary indicator (0.0 or 1.0) for culture event presence (optional)
    - suspected_infection: Binary indicator (0.0 or 1.0) for suspected infection (optional)
    - true_label: Binary ground truth label (0 or 1) aggregated across the encounter (required)
    - prediction_proba_1: Aggregated model probability prediction, range [0.0, 1.0] (required)
    - prediction_proba_2: Alternative aggregated probability prediction, range [0.0, 1.0] (optional)

**Key Characteristics:**
    - Exactly one row per encounter (uniqueness enforced at schema validation)
    - ENFORCES encounter_id uniqueness constraint
    - Represents collapsed temporal sequences from event-level data
    - pysalient.evaluation.evaluation() can accept pre-aggregated data in this format

**Example:**
    | encounter_id | event_timestamp | true_label | prediction_proba_1 |
    |--------------|-----------------|------------|------------------|
    | ENC001       | 3.5             | 1          | 0.7               |  <- Aggregated across 3 earlier events
    | ENC002       | 1.2             | 1          | 0.8               |  <- Aggregated across N events
    | ENC003       | 5.0             | 0          | 0.2               |

**Schema Properties:**
    - coerce: True (automatic type conversion for data ingestion)
    - strict: False (allow extra columns like aggregation metadata)
    - unique=True on encounter_id (enforces one row per encounter)

**Relationship to event-level data:**
    - event-level: 500 encounters × 212 rows/encounter = 106,000 rows total
    - encounter-level: 500 encounters × 1 row/encounter = 500 rows total

**Related schemas:**
    - evaluation_data_event_level_schema: For multi-row event-level data before aggregation
    - evaluation_data_schema: Legacy event-level schema, similar but without uniqueness constraint
"""
