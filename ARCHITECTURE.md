# Refactoring `pysalient/evaluation` for Time-to-Event Metrics

## 1. High-Level Goal

The primary objective is to refactor the `pysalient/evaluation` module to correctly calculate complex, grouped "median time to event" metrics, aligning with the logic demonstrated in the `6_model_evaluation.ipynb` notebook.

## 2. Architectural Principles

### Separation of Concerns: `io` vs. `evaluation`
The `pysalient` library follows a strict separation of concerns:
-   **`pysalient.io`**: This module is responsible for all data ingress and egress. It handles loading data from various sources and performs optional initial aggregations (e.g., grouping by an encounter ID). The aggregation column(s) used are stored in the table's schema metadata under the key `pysalient.io.aggregation_cols`.
-   **`pysalient.evaluation`**: This module consumes a prepared PyArrow Table from the `io` module. Its sole responsibility is to perform metric calculations. It reads metadata from the input table to inform its calculations.

### Internal Technology Stack
All internal data manipulation within the `pysalient` library must prioritize the use of **PyArrow** and **NumPy**. **Pandas** should only be used as a boundary layer for interoperability.

## 3. Key Discrepancies & The Core Problem

The central issue is that the `pysalient.evaluation` module currently lacks the capability to perform the specific, multi-step calculation required for the "median time to event" metrics.

-   **Calculation Logic Scope**:
    -   **Standard Metrics** (e.g., Sensitivity, Specificity, TP, TN, FP, FN) are calculated for each threshold using the **entire dataset** provided.
    -   **Time-to-Event Metrics** (e.g., `med_hrs_from_first_alert_to_bc`) are calculated for each threshold based *only* on the subset of data that are **true positives**.

-   **Notebook Logic (for time-to-event):**
    1.  For each threshold, filter the data to include only **true positives**.
    2.  For this subset, calculate the time difference between the alert timestamp and various clinical event timestamps.
    3.  Group these results by an encounter identifier.
    4.  Within each group, find the **maximum** time difference (to identify the earliest effective alert).
    5.  Calculate the **median** of these maximum time differences across all true positive encounters.

-   **Current `pysalient/evaluation` Logic:**
    -   It correctly calculates standard metrics.
    -   **Crucially, it is missing the logic to:**
        -   Conditionally perform calculations based on metadata.
        -   Read the `pysalient.io.aggregation_cols` from the metadata to identify the grouping column.
        -   Accept multiple, arbitrary event timestamp columns for comparison.
        -   Perform the `filter (true positives) -> groupby -> max -> median` calculation sequence.

## 4. Proposed Architecture & Design

The `evaluation` function will be enhanced to support this complex calculation, making it strictly conditional on the presence of aggregation metadata from the `io` module.

-   **Parameters for `evaluation` function:**
    -   A new `time_to_event_cols` (`dict[str, str] | None`) parameter will be added.
        -   **Key**: The desired name for the output metric column (e.g., `'med_hrs_from_first_alert_to_bc'`).
        -   **Value**: The name of the column in the input data containing the clinical event's timestamp (e.g., `'first_blood_cult_or_flag_dt'`).

-   **Workflow Diagram (Mermaid):**
    ```mermaid
    graph TD
        subgraph pysalient.io
            A[io.load_evaluation_data] --> B{Perform Initial Aggregation?};
            B -- Yes --> C[Group & Aggregate Data];
            B -- No --> D[Use Raw Data];
            C --> E[Attach Metadata, including aggregation_cols];
            D --> E;
        end

        subgraph pysalient.evaluation
            F[evaluation function] --> G{Receive PyArrow Table};
            G --> H[Read Metadata for aggregation_cols];
            H --> I[For each threshold...];
            I --> J[Calculate Standard Metrics on ALL data];
            J --> K{Metadata has aggregation_cols AND time_to_event_cols provided?};
            K -- Yes --> L[Perform Time-to-Event Calculation on TRUE POSITIVES only];
            K -- No --> M[Skip];
            L --> N[Append Time-to-Event Metrics to Results];
            M --> N;
            N --> O[Collect All Results];
        end

        E --> F;
        O --> P[Return Final Results Table];
    ```

## 5. Step-by-Step Implementation Plan

1.  **Modify `evaluation` function signature** in `pysalient/evaluation/evaluation.py`:
    -   Add the `time_to_event_cols: dict[str, str] | None = None` parameter.
    -   Update docstrings to reflect the new parameter and the conditional logic.
    -   Inside the function, read `pysalient.io.aggregation_cols` from the table's schema metadata.
    -   If aggregation metadata and `time_to_event_cols` are present, extract the required columns (`grouping_col`, `timeseries_col`, and event timestamp columns) into NumPy arrays.

2.  **Update `_process_single_evaluation` function** in `pysalient/evaluation/_evaluation_process.py`:
    -   Modify its signature to accept the NumPy array for grouping IDs and the dictionary of event time arrays (both can be `None`).
    -   Inside the main threshold loop, after calculating standard metrics, add a new logic block.
    -   **Condition**: This block only runs if the grouping IDs and time-to-event columns are not `None`.
    -   **Filter**: Create a boolean mask for the **true positives**.
    -   **Calculate**: If there are any true positives:
        -   For each metric in `time_to_event_cols`:
            1.  Apply the true positive mask to all necessary NumPy arrays.
            2.  Calculate the time difference in hours.
            3.  Use NumPy to perform the `groupby().max().median()` equivalent on the filtered data.
            4.  Add the resulting median and counts to the `row_data` dictionary.

3.  **Update the output schema** in `pysalient/evaluation/evaluation.py`:
    -   Dynamically add the new time-to-event metric fields to the output schema by inspecting the keys of the first result dictionary.

4.  **Add Unit Tests** in `tests/evaluation/test_evaluation.py`:
    -   Create a test that calls `pysalient.io.load_evaluation_data` with `perform_aggregation=True`.
    -   Pass the resulting table to the `evaluation` function with `time_to_event_cols`.
    -   Assert that the time-to-event metrics are correctly calculated.
    -   Create another test where `perform_aggregation=False` and assert that the time-to-event metrics are *not* present in the output.
