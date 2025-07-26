import os
import sys

sys.path.insert(0, '..')

import pyarrow.parquet as pq

from pysalient import io as io
from pysalient import visualisation as vis
from pysalient.evaluation import evaluation

sample_data_path = os.path.join("data", "anonymised_sample.parquet")
# count rows
table = pq.read_table(sample_data_path)
print(f"Number of rows: {table.num_rows}")
# print column names
print(table.column_names)
# Convert the 'true_label' column to a pandas Series
true_label_series = table["true_label"].to_pandas()

# Count the number of true labels (1)
true_count = (true_label_series == 1).sum()

print(f"Number of true labels (1): {true_count}")

# Convert the table to a pandas DataFrame for easier grouping
df = table.to_pandas()

# show table
print(df.head(5))


grouped = df.groupby("encounter_id")

# Count the number of unique groups (encounters)
num_groups = df["encounter_id"].nunique()
print(f"Number of unique encounter groups: {num_groups}")

# Calculate the sum of 'true_label' for each group
group_sums = grouped["true_label"].sum()

# Count how many groups have at least one true positive (sum > 0)
groups_with_positives = (group_sums > 0).sum()
print(
    f"Number of encounter groups with at least one true positive: {groups_with_positives}"
)

# Define the path relative to the project root
# Assuming the notebook is run from the project root or examples/ directory
sample_data_path = os.path.join("data", "anonymised_sample.parquet")

assigned_table_events = None

if os.path.exists(sample_data_path):
    # Use the actual column names identified during inspection directly
    # Ensure these names actually exist based on the printout above!
    assigned_table_events = io.load_evaluation_data(
        source=sample_data_path,
        y_proba_col="prediction_proba_1",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        timeseries_col="event_timestamp",
        perform_aggregation=False,

        # We don't provide task_col or model_col from the source
        # assign_task_name="AKI",  # Assign this name to the new 'task' column
        # assign_model_name="LogRegress",  # Assign this name to the new 'model' column
    )

    print("\nSuccessfully loaded data with assigned names (Example 1):")
    print(assigned_table_events.schema)
    print(f"\nNumber of rows: {assigned_table_events.num_rows}")

    # Display first few rows to verify new columns
    print("\nFirst 5 rows (with added 'task' and 'model' columns):")
    print(assigned_table_events.slice(0, 5).to_pandas())

else:
    print(
        f"Skipping data loading (Example 1) as file was not found: {sample_data_path}"
    )

# Define evaluation parameters
eval_modelid = "LogRegress_01"  # Use a generic ID as model wasn't assigned here
eval_filter = "ExampleFilterDummy"  # Describe the data subset
eval_thresholds = (0.01, 0.1, 0.01)  # Range: 0.1, 0.2, ..., 0.9
# eval_thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # Example: List of thresholds

# Run the evaluation
evaluation_results = evaluation(
    data=assigned_table_events,  # Use the table loaded with col_map
    modelid=eval_modelid,
    filter_desc=eval_filter,
    thresholds=eval_thresholds,
    decimal_places=3,  # Control rounding of output floats # check that -1 is no rounding.
    calculate_au_ci=True,  # Enable AU CI calculation (uses bootstrap)
    calculate_threshold_ci=True,
    threshold_ci_method="bootstrap",  # Method for threshold CIs (ignored if calculate_threshold_ci=False)
    ci_alpha=0.05,  # 95% CI
    bootstrap_seed=42,  # For reproducible CIs
    bootstrap_rounds=1000,  # Fewer rounds for notebook speed
    force_threshold_zero=True,
    verbosity=1,
    time_to_event_cols={'bc': 'culture_event', 'sofa': 'suspected_infection'},
    aggregation_func='median',
    time_to_event_fillna=0,
    time_unit='hour',
)

# Visualisation
styled_results = vis.format_evaluation_table(
    evaluation_results, decimal_places=3, ci_column=False
)
print(styled_results)

sample_data_path = os.path.join("data", "anonymised_sample.parquet")

assigned_table_agg = None

if os.path.exists(sample_data_path):
    # Use the actual column names identified during inspection directly
    # Ensure these names actually exist based on the printout above!
    assigned_table_agg = io.load_evaluation_data(
        source=sample_data_path,
        y_proba_col="prediction_proba_1",
        y_label_col="true_label",
        aggregation_cols="encounter_id",
        proba_agg_func="max",
        label_agg_func="max",
        timeseries_col="event_timestamp",
        perform_aggregation=True,  # Explicitly enable aggregation
        # We don't provide task_col or model_col from the source
        # assign_task_name="AKI",  # Assign this name to the new 'task' column
        # assign_model_name="LogRegress",  # Assign this name to the new 'model' column
    )

    print("\nSuccessfully loaded data with assigned names (Example 1):")
    print(assigned_table_agg.schema)
    print(f"\nNumber of rows: {assigned_table_agg.num_rows}")

    # Display first few rows to verify new columns
    print("\nFirst 5 rows (with added 'task' and 'model' columns):")
    print(assigned_table_agg.slice(0, 5).to_pandas())

else:
    print(
        f"Skipping data loading (Example 1) as file was not found: {sample_data_path}"
    )

evaluation_results_agg = evaluation(
    data=assigned_table_agg,  # Use the table loaded with col_map
    modelid=eval_modelid,
    filter_desc=eval_filter,
    thresholds=eval_thresholds,
    decimal_places=3,  # Control rounding of output floats # check that -1 is no rounding.
    calculate_au_ci=True,  # Enable AU CI calculation (uses bootstrap)
    calculate_threshold_ci=True,
    threshold_ci_method="bootstrap",  # Method for threshold CIs (ignored if calculate_threshold_ci=False)
    ci_alpha=0.05,  # 95% CI
    bootstrap_seed=42,  # For reproducible CIs
    bootstrap_rounds=1000,  # Fewer rounds for notebook speed
    force_threshold_zero=True,
    verbosity=1,
    time_to_event_cols={'bc': 'culture_event', 'sofa': 'suspected_infection'},
    aggregation_func='median',
    time_to_event_fillna=0,
    time_unit='hour',
)

# Visualisation
styled_results = vis.format_evaluation_table(
    evaluation_results_agg, decimal_places=3, ci_column=False
)
print(styled_results)

