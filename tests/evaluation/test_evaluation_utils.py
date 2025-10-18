"""Tests for pysalient.evaluation._evaluation_utils internal functions."""

import pytest

from pysalient.evaluation._evaluation_utils import _generate_thresholds


class TestGenerateThresholdsValidInputs:
    """Test valid threshold specifications."""

    def test_explicit_list_basic(self):
        """Test basic explicit list of thresholds."""
        result = _generate_thresholds([0.1, 0.5, 0.9])
        expected = [0.0, 0.1, 0.5, 0.9]  # 0.0 added by default
        assert result == pytest.approx(expected)

    def test_explicit_tuple_basic(self):
        """Test basic explicit tuple of thresholds."""
        result = _generate_thresholds((0.2, 0.6, 0.8))
        expected = [0.0, 0.2, 0.6, 0.8]  # 0.0 added by default
        assert result == pytest.approx(expected)

    def test_explicit_list_with_duplicates(self):
        """Test that duplicates are removed."""
        result = _generate_thresholds([0.1, 0.5, 0.1, 0.5, 0.9])
        expected = [0.0, 0.1, 0.5, 0.9]  # Duplicates removed, 0.0 added
        assert result == pytest.approx(expected)

    def test_explicit_list_unsorted(self):
        """Test that results are sorted."""
        result = _generate_thresholds([0.9, 0.1, 0.5, 0.3])
        expected = [0.0, 0.1, 0.3, 0.5, 0.9]  # Sorted, 0.0 added
        assert result == pytest.approx(expected)

    def test_explicit_list_with_zero(self):
        """Test that 0.0 is not duplicated when already present."""
        result = _generate_thresholds([0.0, 0.5, 0.8])
        expected = [0.0, 0.5, 0.8]  # 0.0 not duplicated
        assert result == pytest.approx(expected)

    def test_explicit_list_boundary_values(self):
        """Test boundary values 0.0 and 1.0."""
        result = _generate_thresholds([0.0, 0.5, 1.0])
        expected = [0.0, 0.5, 1.0]
        assert result == pytest.approx(expected)

    def test_single_value_list(self):
        """Test list with single value."""
        result = _generate_thresholds([0.5])
        expected = [0.0, 0.5]  # 0.0 added
        assert result == pytest.approx(expected)

    def test_single_value_tuple(self):
        """Test tuple with single value."""
        result = _generate_thresholds((0.7,))
        expected = [0.0, 0.7]  # 0.0 added
        assert result == pytest.approx(expected)


class TestGenerateThresholdsRangeSpecification:
    """Test range specification (3-tuple) functionality."""

    def test_basic_range(self):
        """Test basic range specification."""
        result = _generate_thresholds((0.1, 0.5, 0.1))
        expected = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0.0 added
        assert result == pytest.approx(expected)

    def test_range_without_zero_inclusion(self):
        """Test range that naturally includes 0.0."""
        result = _generate_thresholds((0.0, 0.4, 0.2))
        expected = [0.0, 0.2, 0.4]  # 0.0 from range, not added
        assert result == pytest.approx(expected)

    def test_range_small_step(self):
        """Test range with small step size."""
        result = _generate_thresholds((0.1, 0.3, 0.05))
        expected = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]  # 0.0 added
        assert result == pytest.approx(expected)

    def test_range_single_point(self):
        """Test range where start equals stop."""
        result = _generate_thresholds((0.5, 0.5, 0.1))
        expected = [0.0, 0.5]  # Single point, 0.0 added
        assert result == pytest.approx(expected)

    def test_range_boundary_to_boundary(self):
        """Test range from 0.0 to 1.0."""
        result = _generate_thresholds((0.0, 1.0, 0.25))
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert result == pytest.approx(expected)

    def test_range_clipping(self):
        """Test that range values are clipped to [0,1]."""
        # This test assumes the function clips values outside [0,1]
        result = _generate_thresholds((-0.1, 1.1, 0.5))
        expected = [0.0, 0.5, 1.0]  # Clipped to valid range
        assert result == pytest.approx(expected)

    def test_range_with_float_precision(self):
        """Test range with float precision considerations."""
        result = _generate_thresholds((0.0, 0.1, 0.03))
        # Should handle float precision gracefully
        assert len(result) >= 4  # At least 0.0, 0.03, 0.06, 0.09
        assert result[0] == pytest.approx(0.0)
        assert result[-1] <= 0.1


class TestGenerateThresholdsIncludeZeroFlag:
    """Test the include_zero parameter functionality."""

    def test_include_zero_true_default(self):
        """Test default behavior (include_zero=True)."""
        result = _generate_thresholds([0.5, 0.8])
        assert 0.0 in result
        assert result[0] == 0.0

    def test_include_zero_true_explicit(self):
        """Test explicit include_zero=True."""
        result = _generate_thresholds([0.5, 0.8], include_zero=True)
        assert 0.0 in result
        assert result[0] == 0.0

    def test_include_zero_false(self):
        """Test include_zero=False when 0.0 not in specification."""
        result = _generate_thresholds([0.5, 0.8], include_zero=False)
        assert 0.0 not in result
        assert result == pytest.approx([0.5, 0.8])

    def test_include_zero_false_with_explicit_zero(self):
        """Test include_zero=False when 0.0 is explicitly in specification."""
        result = _generate_thresholds([0.0, 0.5, 0.8], include_zero=False)
        assert 0.0 in result  # Explicit 0.0 should remain
        assert result == pytest.approx([0.0, 0.5, 0.8])

    def test_include_zero_false_range_with_zero(self):
        """Test include_zero=False with range that generates 0.0."""
        result = _generate_thresholds((0.0, 0.4, 0.2), include_zero=False)
        assert 0.0 in result  # Generated by range, should remain
        assert result == pytest.approx([0.0, 0.2, 0.4])

    def test_include_zero_false_range_without_zero(self):
        """Test include_zero=False with range that doesn't generate 0.0."""
        result = _generate_thresholds((0.1, 0.5, 0.2), include_zero=False)
        assert 0.0 not in result
        assert result == pytest.approx([0.1, 0.3, 0.5])


class TestGenerateThresholdsInvalidInputs:
    """Test invalid input handling and error cases."""

    def test_empty_list(self):
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="Threshold specification cannot be empty"):
            _generate_thresholds([])

    def test_empty_tuple(self):
        """Test empty tuple raises ValueError."""
        with pytest.raises(ValueError, match="Threshold specification cannot be empty"):
            _generate_thresholds(())

    def test_invalid_type_input(self):
        """Test invalid input types raise TypeError."""
        with pytest.raises(
            TypeError, match="Thresholds must be specified as a list or tuple"
        ):
            _generate_thresholds(0.5)  # Single number

        with pytest.raises(
            TypeError, match="Thresholds must be specified as a list or tuple"
        ):
            _generate_thresholds("0.5")  # String

        with pytest.raises(
            TypeError, match="Thresholds must be specified as a list or tuple"
        ):
            _generate_thresholds({0.5, 0.8})  # Set

    def test_non_numeric_values_in_list(self):
        """Test non-numeric values in list raise ValueError."""
        with pytest.raises(
            ValueError, match="Explicit threshold list/tuple must contain numbers"
        ):
            _generate_thresholds([0.1, "0.5", 0.9])

        with pytest.raises(
            ValueError, match="Explicit threshold list/tuple must contain numbers"
        ):
            _generate_thresholds([0.1, None, 0.9])

    def test_values_out_of_range(self):
        """Test values outside [0,1] raise ValueError."""
        with pytest.raises(
            ValueError, match="All thresholds must be between 0.0 and 1.0"
        ):
            _generate_thresholds([0.1, 1.5, 0.9])

        with pytest.raises(
            ValueError, match="All thresholds must be between 0.0 and 1.0"
        ):
            _generate_thresholds([-0.1, 0.5, 0.9])

    def test_range_invalid_step_zero(self):
        """Test range with zero step raises ValueError."""
        with pytest.raises(ValueError, match="Threshold step cannot be zero"):
            _generate_thresholds((0.1, 0.5, 0.0))

    def test_range_invalid_step_negative(self):
        """Test range with negative step raises ValueError."""
        with pytest.raises(ValueError, match="Threshold step must be positive"):
            _generate_thresholds((0.1, 0.5, -0.1))

    def test_range_invalid_start_greater_than_stop(self):
        """Test range where start > stop raises ValueError."""
        with pytest.raises(
            ValueError, match="Threshold range start cannot be greater than stop"
        ):
            _generate_thresholds((0.8, 0.3, 0.1))

    def test_range_step_too_small(self):
        """Test range with extremely small step raises ValueError."""
        with pytest.raises(
            ValueError, match="Threshold step must be positive and non-negligible"
        ):
            _generate_thresholds((0.1, 0.5, 1e-12))

    def test_range_non_numeric_values(self):
        """Test range with non-numeric values raises ValueError."""
        with pytest.raises(
            ValueError, match="Explicit threshold list/tuple must contain numbers"
        ):
            _generate_thresholds((0.1, "0.5", 0.1))

        with pytest.raises(
            ValueError, match="Explicit threshold list/tuple must contain numbers"
        ):
            _generate_thresholds(("start", 0.5, 0.1))


class TestGenerateThresholdsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_only_zero_threshold(self):
        """Test specification with only 0.0."""
        result = _generate_thresholds([0.0])
        expected = [0.0]
        assert result == pytest.approx(expected)

    def test_only_one_threshold(self):
        """Test specification with only 1.0."""
        result = _generate_thresholds([1.0])
        expected = [0.0, 1.0]  # 0.0 added
        assert result == pytest.approx(expected)

    def test_very_close_values(self):
        """Test thresholds that are very close together."""
        result = _generate_thresholds([0.5, 0.5000001, 0.5000002])
        # Should handle close values without issues
        assert len(result) >= 2  # At least 0.0 + unique values
        assert result[0] == pytest.approx(0.0)

    def test_integer_values(self):
        """Test integer values in threshold specification."""
        result = _generate_thresholds([0, 1])  # Integers 0 and 1
        expected = [0.0, 1.0]
        assert result == pytest.approx(expected)

    def test_mixed_int_float_values(self):
        """Test mixed integer and float values."""
        result = _generate_thresholds([0, 0.5, 1])  # Mixed int/float
        expected = [0.0, 0.5, 1.0]
        assert result == pytest.approx(expected)

    def test_range_extremely_small_interval(self):
        """Test range with extremely small interval."""
        result = _generate_thresholds((0.49999, 0.50001, 0.00001))
        # Should generate valid thresholds in small interval
        assert len(result) >= 2  # At least start and end points
        assert all(0.0 <= x <= 1.0 for x in result)

    def test_range_single_step(self):
        """Test range that produces exactly two points."""
        result = _generate_thresholds((0.3, 0.7, 0.4))
        expected = [0.0, 0.3, 0.7]  # Start, end, 0.0 added
        assert result == pytest.approx(expected)

    def test_range_many_points(self):
        """Test range that produces many points."""
        result = _generate_thresholds((0.0, 1.0, 0.01))
        # Should produce 101 points (0.00, 0.01, ..., 1.00)
        assert len(result) == 101
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        # Test case that might expose floating point precision problems
        result = _generate_thresholds((0.1, 0.3, 0.1))
        expected = [0.0, 0.1, 0.2, 0.3]
        assert result == pytest.approx(expected, abs=1e-10)


class TestGenerateThresholdsReturnProperties:
    """Test properties of the return value."""

    def test_return_type_is_list(self):
        """Test that return type is always list."""
        result = _generate_thresholds([0.5])
        assert isinstance(result, list)

    def test_return_values_are_floats(self):
        """Test that all returned values are floats."""
        result = _generate_thresholds([0, 1])  # Input integers
        assert all(isinstance(x, float) for x in result)

    def test_return_values_sorted(self):
        """Test that return values are always sorted."""
        result = _generate_thresholds([0.9, 0.1, 0.5, 0.3])
        assert result == sorted(result)

    def test_return_values_unique(self):
        """Test that return values are unique."""
        result = _generate_thresholds([0.5, 0.5, 0.5, 0.3, 0.3])
        assert len(result) == len(set(result))

    def test_return_values_in_range(self):
        """Test that all return values are in [0,1]."""
        result = _generate_thresholds((0.0, 1.0, 0.1))
        assert all(0.0 <= x <= 1.0 for x in result)

    def test_return_non_empty(self):
        """Test that return is never empty for valid inputs."""
        result = _generate_thresholds([0.5])
        assert len(result) > 0

        result = _generate_thresholds((0.2, 0.8, 0.3))
        assert len(result) > 0
