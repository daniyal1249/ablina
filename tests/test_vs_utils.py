"""
Unit tests for the ablina.vs_utils module.
"""

import unittest
from ablina import *


class TestToNsMatrix(unittest.TestCase):
    """Test to_ns_matrix function."""

    def test_to_ns_matrix_single_constraint(self):
        """Test to_ns_matrix with a single constraint."""
        constraints = ["v0 + v1 == 0"]
        result = to_ns_matrix(2, constraints)
        self.assertIsInstance(result, M)
        # Should have one row representing v0 + v1 = 0
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.cols, 2)

    def test_to_ns_matrix_multiple_constraints(self):
        """Test to_ns_matrix with multiple constraints."""
        constraints = ["v0 + v1 == 0", "v1 + v2 == 0"]
        result = to_ns_matrix(3, constraints)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 3)

    def test_to_ns_matrix_duplicate_constraints(self):
        """Test to_ns_matrix with duplicate constraints."""
        constraints = ["v0 + v1 == 0", "v0 + v1 == 0"]
        result = to_ns_matrix(2, constraints)
        # Duplicates should be removed
        self.assertIsInstance(result, M)

    def test_to_ns_matrix_empty_constraints(self):
        """Test to_ns_matrix with empty constraints."""
        result = to_ns_matrix(3, [])
        self.assertIsInstance(result, M)
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.cols, 3)

    def test_to_ns_matrix_zero_dimension(self):
        """Test to_ns_matrix with zero dimension."""
        result = to_ns_matrix(0, [])
        self.assertIsInstance(result, M)
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.cols, 0)

    def test_to_ns_matrix_complex_constraint(self):
        """Test to_ns_matrix with complex constraint."""
        constraints = ["2*v0 + 3*v1 - v2 == 0"]
        result = to_ns_matrix(3, constraints)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 3)

    def test_to_ns_matrix_multiple_variables(self):
        """Test to_ns_matrix with multiple variables in constraint."""
        constraints = ["v0 + 2*v1 + 3*v2 == 0"]
        result = to_ns_matrix(3, constraints)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 3)

    def test_to_ns_matrix_invalid_constraint(self):
        """Test to_ns_matrix with invalid constraint raises ConstraintError."""
        constraints = ["invalid constraint"]
        with self.assertRaises(ConstraintError):
            to_ns_matrix(2, constraints)

    def test_to_ns_matrix_invalid_variable(self):
        """Test to_ns_matrix with invalid variable name."""
        constraints = ["x + y == 0"]  # Should use v0, v1, etc.
        with self.assertRaises(ConstraintError):
            to_ns_matrix(2, constraints)

    def test_to_ns_matrix_rref_applied(self):
        """Test that to_ns_matrix applies RREF."""
        constraints = ["v0 + v1 == 0", "2*v0 + 2*v1 == 0"]
        result = to_ns_matrix(2, constraints)
        # RREF should remove redundant rows
        self.assertIsInstance(result, M)

    def test_to_ns_matrix_large_dimension(self):
        """Test to_ns_matrix with large dimension."""
        constraints = ["v0 + v10 == 0"]
        result = to_ns_matrix(11, constraints)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 11)

    def test_to_ns_matrix_negative_coefficients(self):
        """Test to_ns_matrix with negative coefficients."""
        constraints = ["v0 - v1 == 0"]
        result = to_ns_matrix(2, constraints)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 2)


class TestToComplement(unittest.TestCase):
    """Test to_complement function."""

    def test_to_complement_identity_matrix(self):
        """Test to_complement with identity matrix."""
        mat = M.eye(3)
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Complement of full space is empty space
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.cols, 3)

    def test_to_complement_zero_matrix(self):
        """Test to_complement with zero matrix."""
        mat = M.zeros(2, 3)
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Complement of zero space is full space
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.cols, 3)

    def test_to_complement_empty_matrix(self):
        """Test to_complement with empty matrix (0 rows)."""
        mat = M.zeros(0, 3)
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Complement of empty matrix should be identity
        self.assertEqual(result.rows, 3)
        self.assertEqual(result.cols, 3)
        # Should be identity matrix
        self.assertTrue(result.equals(M.eye(3)))

    def test_to_complement_single_row(self):
        """Test to_complement with single row matrix."""
        mat = M([[1, 0, 0]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Null space of [1, 0, 0] is span of [0, 1, 0] and [0, 0, 1]
        self.assertEqual(result.cols, 3)

    def test_to_complement_square_matrix(self):
        """Test to_complement with square matrix."""
        mat = M([[1, 1], [0, 1]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 2)

    def test_to_complement_rectangular_matrix(self):
        """Test to_complement with rectangular matrix."""
        mat = M([[1, 0, 0], [0, 1, 0]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 3)

    def test_to_complement_bidirectional(self):
        """Test to_complement bidirectional property."""
        # Start with a null space matrix
        null_space = M([[1, 1, 0], [1, 0, 1]])
        row_space = to_complement(null_space)
        
        # Complement of the row space should give back something related
        # (not necessarily the same, but should be valid)
        result = to_complement(row_space)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 3)

    def test_to_complement_list_input(self):
        """Test to_complement with list input."""
        mat = [[1, 0], [0, 1]]
        result = to_complement(mat)
        self.assertIsInstance(result, M)

    def test_to_complement_rref_applied(self):
        """Test that to_complement applies RREF."""
        mat = M([[1, 1, 0], [2, 2, 0]])  # Second row is multiple of first
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # RREF should remove redundant rows
        self.assertEqual(result.cols, 3)

    def test_to_complement_full_rank(self):
        """Test to_complement with full rank matrix."""
        mat = M([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Full rank matrix has empty null space
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.cols, 3)

    def test_to_complement_single_column(self):
        """Test to_complement with single column matrix."""
        mat = M([[1], [0], [0]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 1)

    def test_to_complement_single_row_single_col(self):
        """Test to_complement with 1x1 matrix."""
        mat = M([[1]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        # Null space of [1] is empty
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.cols, 1)

    def test_to_complement_zero_row(self):
        """Test to_complement with matrix containing zero row."""
        mat = M([[1, 0], [0, 0]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 2)

    def test_to_complement_complex_values(self):
        """Test to_complement with complex matrix values."""
        from sympy import I
        mat = M([[1, I], [0, 1]])
        result = to_complement(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.cols, 2)


if __name__ == '__main__':
    unittest.main()

