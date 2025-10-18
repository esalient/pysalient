import numpy as np


def _generate_thresholds(
    threshold_spec: list[float] | tuple[float, ...] | tuple[float, float, float],
    include_zero: bool = True,
) -> list[float]:
    """
    Parses or generates a list of thresholds based on the specification.

    Args:
        threshold_spec: The specification for thresholds. Can be a list/tuple
                        of explicit values or a tuple (start, stop, step) for a range.
        include_zero: If True (default), ensures 0.0 is included in the final list
                      if it wasn't already generated.

    Returns:
        A sorted list of unique threshold values between 0.0 and 1.0.
    """
    thresholds_out: list[float] = []

    # Case 1: Tuple of 3 numbers potentially representing a range
    if (
        isinstance(threshold_spec, tuple)
        and len(threshold_spec) == 3
        and all(isinstance(x, int | float) for x in threshold_spec)
    ):
        start, stop, step = threshold_spec

        # Heuristic: Treat as explicit list if step is significantly larger compared to range
        # This handles cases like (0.2, 0.6, 0.8) which should be explicit values
        # Use a tolerance to handle floating point precision issues
        range_size = abs(stop - start)
        if (
            step > range_size * 1.1 and range_size > 0
        ):  # 10% tolerance for floating point
            # Treat as explicit list instead of range
            thresholds_out = [float(t) for t in threshold_spec]
        else:
            # Treat as range specification
            if step == 0:
                # Explicitly disallow zero step when it looks like a range spec
                raise ValueError(
                    "Threshold step cannot be zero for range specification."
                )
            if step < 0:
                # We decided to only support positive steps for ranges
                raise ValueError(
                    "Threshold step must be positive for range specification."
                )
            if start > stop:
                raise ValueError(
                    "Threshold range start cannot be greater than stop with a positive step."
                )

            # Use np.linspace for potentially better precision with float steps
            # Add small epsilon to step comparison to handle potential float issues near zero
            if step <= 1e-9:  # Avoid division by zero or tiny steps
                raise ValueError(
                    "Threshold step must be positive and non-negligible for range specification."
                )
            # Ensure we handle the case where start and stop are the same
            if np.isclose(start, stop):
                num_points = 1
            else:
                # Calculate number of points, rounding to handle potential float inaccuracies
                num_points = int(round((stop - start) / step)) + 1
                # Basic sanity check on calculated points
                if num_points <= 0:
                    raise ValueError(
                        f"Calculated number of threshold points ({num_points}) is not positive. Check start/stop/step."
                    )
                # Consider adding an upper bound check if very large num_points are unexpected

            thresholds_np = np.linspace(start, stop, num_points)
            thresholds_out = np.clip(thresholds_np, 0.0, 1.0).tolist()

    # Case 2: Any other list or tuple (including 3-element tuple with non-numeric or non-tuple type)
    elif isinstance(threshold_spec, list | tuple):
        if len(threshold_spec) == 0:
            raise ValueError("Threshold specification cannot be empty.")
        if not all(isinstance(x, int | float) for x in threshold_spec):
            raise ValueError("Explicit threshold list/tuple must contain numbers.")
        thresholds_out = [float(t) for t in threshold_spec]

    # Case 3: Invalid type
    else:
        raise TypeError("Thresholds must be specified as a list or tuple.")

    # Final validation and processing
    if (
        not thresholds_out
    ):  # Check if list ended up empty (e.g., range produced nothing)
        raise ValueError("Threshold specification resulted in an empty list.")

    if not all(0.0 <= t <= 1.0 for t in thresholds_out):
        raise ValueError("All thresholds must be between 0.0 and 1.0.")

    # Remove duplicates and sort
    thresholds_final = sorted(list(set(thresholds_out)))

    # Optionally include 0.0 if not already present
    if include_zero and (not thresholds_final or thresholds_final[0] != 0.0):
        # Check if list is empty or if first element isn't 0.0
        if 0.0 not in thresholds_final:  # Double check it wasn't added by range/clip
            thresholds_final.insert(0, 0.0)

    return thresholds_final
