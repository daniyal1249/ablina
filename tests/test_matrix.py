"""
Unit tests for the ablina.matrix module.
"""

import unittest
import sympy as sp
from ablina import *


class TestMatrix(unittest.TestCase):
    """Test cases for the Matrix class."""

    def test_matrix_creation_from_list(self):
        """Test creating a matrix from a list."""
        mat = Matrix([[1, 2], [3, 4]])
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)
        self.assertEqual(mat[0, 0], 1)
        self.assertEqual(mat[0, 1], 2)
        self.assertEqual(mat[1, 0], 3)
        self.assertEqual(mat[1, 1], 4)

    def test_matrix_creation_from_nested_list(self):
        """Test creating a matrix from nested lists."""
        mat = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 3)

    def test_matrix_creation_from_tuple(self):
        """Test creating a matrix from a tuple."""
        mat = Matrix(((1, 2), (3, 4)))
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)

    def test_matrix_creation_from_single_list(self):
        """Test creating a column vector from a single list."""
        mat = Matrix([1, 2, 3])
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 1)

    def test_matrix_creation_from_sympy_matrix(self):
        """Test creating a Matrix from a sympy Matrix."""
        sympy_mat = sp.Matrix([[1, 2], [3, 4]])
        mat = Matrix(sympy_mat)
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)

    def test_matrix_identity_check(self):
        """Test that passing a Matrix instance returns the same instance."""
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix(mat1)
        self.assertIs(mat1, mat2)

    def test_matrix_identity_check_different_instances(self):
        """Test that different Matrix instances are not the same."""
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[1, 2], [3, 4]])
        self.assertIsNot(mat1, mat2)
        # But they should be equal
        self.assertEqual(mat1, mat2)

    def test_matrix_repr_column_vector(self):
        """Test __repr__ for a column vector (cols == 1)."""
        mat = Matrix([1, 2, 3])
        # For column vectors, should return flat() representation
        repr_str = repr(mat)
        # Should be a string representation of the flat list
        self.assertIsInstance(repr_str, str)
        # Check that it contains the values
        self.assertIn('1', repr_str)
        self.assertIn('2', repr_str)
        self.assertIn('3', repr_str)

    def test_matrix_repr_matrix(self):
        """Test __repr__ for a regular matrix (cols > 1)."""
        mat = Matrix([[1, 2], [3, 4]])
        repr_str = repr(mat)
        # Should be a string representation of the tolist()
        self.assertIsInstance(repr_str, str)
        # Should contain the matrix structure
        self.assertIn('1', repr_str)
        self.assertIn('2', repr_str)
        self.assertIn('3', repr_str)
        self.assertIn('4', repr_str)

    def test_matrix_str(self):
        """Test __str__ method (should call __repr__)."""
        mat = Matrix([[1, 2], [3, 4]])
        str_repr = str(mat)
        repr_repr = repr(mat)
        self.assertEqual(str_repr, repr_repr)

    def test_matrix_str_column_vector(self):
        """Test __str__ for column vector."""
        mat = Matrix([1, 2, 3])
        str_repr = str(mat)
        repr_repr = repr(mat)
        self.assertEqual(str_repr, repr_repr)

    def test_matrix_inherits_sympy_functionality(self):
        """Test that Matrix inherits sympy Matrix functionality."""
        mat = Matrix([[1, 2], [3, 4]])
        # Test some sympy Matrix methods
        self.assertEqual(mat.det(), -2)
        self.assertEqual(mat.trace(), 5)
        self.assertTrue(mat.is_square)

    def test_matrix_arithmetic_operations(self):
        """Test that Matrix supports arithmetic operations from sympy."""
        mat1 = Matrix([[1, 2], [3, 4]])
        mat2 = Matrix([[5, 6], [7, 8]])
        # Addition
        result = mat1 + mat2
        self.assertEqual(result[0, 0], 6)
        self.assertEqual(result[0, 1], 8)
        # Multiplication
        result = mat1 * mat2
        self.assertEqual(result[0, 0], 19)
        self.assertEqual(result[0, 1], 22)

    def test_matrix_scalar_multiplication(self):
        """Test scalar multiplication."""
        mat = Matrix([[1, 2], [3, 4]])
        result = 2 * mat
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[0, 1], 4)
        self.assertEqual(result[1, 0], 6)
        self.assertEqual(result[1, 1], 8)

    def test_matrix_transpose(self):
        """Test matrix transpose."""
        mat = Matrix([[1, 2], [3, 4]])
        transposed = mat.T
        self.assertEqual(transposed[0, 0], 1)
        self.assertEqual(transposed[0, 1], 3)
        self.assertEqual(transposed[1, 0], 2)
        self.assertEqual(transposed[1, 1], 4)

    def test_matrix_creation_with_symbols(self):
        """Test creating a matrix with symbolic elements."""
        x, y = sp.symbols('x y')
        mat = Matrix([[x, y], [1, 2]])
        self.assertEqual(mat[0, 0], x)
        self.assertEqual(mat[0, 1], y)
        self.assertEqual(mat[1, 0], 1)
        self.assertEqual(mat[1, 1], 2)

    def test_matrix_creation_empty(self):
        """Test creating an empty matrix."""
        mat = Matrix([])
        self.assertEqual(mat.rows, 0)
        self.assertEqual(mat.cols, 0)

    def test_matrix_creation_single_element(self):
        """Test creating a 1x1 matrix."""
        mat = Matrix([[5]])
        self.assertEqual(mat.rows, 1)
        self.assertEqual(mat.cols, 1)
        self.assertEqual(mat[0, 0], 5)

    def test_matrix_creation_from_flat_list(self):
        """Test creating a matrix from a flat list (creates column vector)."""
        mat = Matrix([1, 2, 3, 4])
        self.assertEqual(mat.rows, 4)
        self.assertEqual(mat.cols, 1)


class TestMatrixClassGetItem(unittest.TestCase):
    """Test cases for Matrix.__class_getitem__."""

    def test_class_getitem_with_tuple(self):
        """Test Matrix[tuple] creates a matrix from tuple."""
        mat = Matrix[(1, 2), (3, 4)]
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)
        self.assertEqual(mat[0, 0], 1)
        self.assertEqual(mat[0, 1], 2)

    def test_class_getitem_with_single_value(self):
        """Test Matrix[single_value] creates a column vector."""
        mat = Matrix[1, 2, 3]
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 1)
        self.assertEqual(mat[0, 0], 1)
        self.assertEqual(mat[1, 0], 2)
        self.assertEqual(mat[2, 0], 3)

    def test_class_getitem_with_nested_tuple(self):
        """Test Matrix with nested tuple structure."""
        mat = Matrix[((1, 2), (3, 4))]
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)

    def test_class_getitem_single_element(self):
        """Test Matrix with single element."""
        mat = Matrix[5]
        self.assertEqual(mat.rows, 1)
        self.assertEqual(mat.cols, 1)
        self.assertEqual(mat[0, 0], 5)


class TestMAlias(unittest.TestCase):
    """Test cases for the M alias."""

    def test_m_is_matrix(self):
        """Test that M is the Matrix class."""
        self.assertIs(M, Matrix)

    def test_m_creation(self):
        """Test creating a matrix using M alias."""
        mat = M([[1, 2], [3, 4]])
        self.assertIsInstance(mat, Matrix)
        self.assertEqual(mat.rows, 2)
        self.assertEqual(mat.cols, 2)

    def test_m_class_getitem(self):
        """Test using M with __class_getitem__."""
        mat = M[1, 2, 3]
        self.assertIsInstance(mat, Matrix)
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 1)


class TestMatrixEdgeCases(unittest.TestCase):
    """Test edge cases for Matrix."""

    def test_matrix_repr_zero_matrix(self):
        """Test __repr__ for zero matrix."""
        mat = Matrix([[0, 0], [0, 0]])
        repr_str = repr(mat)
        self.assertIsInstance(repr_str, str)

    def test_matrix_repr_large_matrix(self):
        """Test __repr__ for larger matrix."""
        data = [[i + j for j in range(5)] for i in range(3)]
        mat = Matrix(data)
        repr_str = repr(mat)
        self.assertIsInstance(repr_str, str)
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 5)

    def test_matrix_identity_preservation(self):
        """Test that identity check works with various inputs."""
        mat1 = Matrix([[1, 2]])
        # Creating from the same matrix should return the same instance
        mat2 = Matrix(mat1)
        self.assertIs(mat1, mat2)
        # But creating from the data should create a new instance
        mat3 = Matrix([[1, 2]])
        self.assertIsNot(mat1, mat3)

    def test_matrix_with_fractions(self):
        """Test matrix with fractional entries."""
        mat = Matrix([[sp.Rational(1, 2), sp.Rational(3, 4)]])
        self.assertEqual(mat[0, 0], sp.Rational(1, 2))
        self.assertEqual(mat[0, 1], sp.Rational(3, 4))

    def test_matrix_with_decimals(self):
        """Test matrix with decimal entries."""
        from decimal import Decimal
        mat = Matrix([[Decimal('1.5'), Decimal('2.5')]])
        # sympy converts Decimal to Float, so we check float equivalence
        self.assertAlmostEqual(float(mat[0, 0]), 1.5)
        self.assertAlmostEqual(float(mat[0, 1]), 2.5)


if __name__ == '__main__':
    unittest.main()

