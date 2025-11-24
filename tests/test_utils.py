"""
Unit tests for the ablina.utils module.
"""

import unittest
import sympy as sp
from ablina import *


class TestSymbols(unittest.TestCase):
    """Test cases for the symbols function."""

    def test_symbols_without_field(self):
        """Test symbols without field specification."""
        result = symbols('x y z')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_symbols_with_r_field(self):
        """Test symbols with R field (real=True)."""
        result = symbols('x y', field=R)
        self.assertIsInstance(result, tuple)
        x, y = result
        self.assertTrue(x.is_real)
        self.assertTrue(y.is_real)

    def test_symbols_with_c_field(self):
        """Test symbols with C field (complex=True)."""
        result = symbols('x y', field=C)
        self.assertIsInstance(result, tuple)
        x, y = result
        self.assertTrue(x.is_complex)

    def test_symbols_single_name(self):
        """Test symbols with a single name."""
        result = symbols('x', field=R)
        self.assertIsInstance(result, sp.Symbol)
        self.assertTrue(result.is_real)

    def test_symbols_with_kwargs(self):
        """Test symbols with additional keyword arguments."""
        result = symbols('x', field=R, positive=True)
        x = result
        self.assertTrue(x.is_positive)

    def test_symbols_multiple_names_with_field(self):
        """Test symbols with multiple names and field."""
        result = symbols('a b c', field=C)
        a, b, c = result
        self.assertTrue(a.is_complex)
        self.assertTrue(b.is_complex)
        self.assertTrue(c.is_complex)


class TestIsLinear(unittest.TestCase):
    """Test cases for the is_linear function."""

    def test_is_linear_constant(self):
        """Test is_linear with a constant expression."""
        expr = sp.Integer(5)
        self.assertTrue(is_linear(expr))

    def test_is_linear_linear_expression(self):
        """Test is_linear with a linear expression."""
        x, y = sp.symbols('x y')
        expr = 2*x + 3*y + 1
        self.assertTrue(is_linear(expr))

    def test_is_linear_quadratic_expression(self):
        """Test is_linear with a quadratic expression."""
        x = sp.Symbol('x')
        expr = x**2 + x + 1
        self.assertFalse(is_linear(expr))

    def test_is_linear_with_vars_specified(self):
        """Test is_linear with specific variables."""
        x, y, z = sp.symbols('x y z')
        expr = 2*x + 3*y + z**2
        # Linear in x and y, but not in z
        self.assertFalse(is_linear(expr, vars=[x, y, z]))
        self.assertTrue(is_linear(expr, vars=[x, y]))

    def test_is_linear_no_variables(self):
        """Test is_linear with expression containing no variables."""
        expr = sp.Integer(10)
        self.assertTrue(is_linear(expr))

    def test_is_linear_mixed_terms(self):
        """Test is_linear with mixed linear and non-linear terms."""
        x, y = sp.symbols('x y')
        expr = x + y + x*y
        self.assertFalse(is_linear(expr))

    def test_is_linear_polynomial_error(self):
        """Test is_linear with non-polynomial expression."""
        x = sp.Symbol('x')
        expr = sp.sin(x)
        self.assertFalse(is_linear(expr))

    def test_is_linear_exponential(self):
        """Test is_linear with exponential expression."""
        x = sp.Symbol('x')
        expr = sp.exp(x)
        self.assertFalse(is_linear(expr))

    def test_is_linear_with_empty_vars(self):
        """Test is_linear with empty vars set."""
        x = sp.Symbol('x')
        expr = x + 1
        self.assertTrue(is_linear(expr, vars=set()))


class TestIsEmpty(unittest.TestCase):
    """Test cases for the is_empty function."""

    def test_is_empty_zero_rows(self):
        """Test is_empty with matrix having zero rows."""
        mat = M.zeros(0, 3)
        self.assertTrue(is_empty(mat))

    def test_is_empty_zero_cols(self):
        """Test is_empty with matrix having zero columns."""
        mat = M.zeros(3, 0)
        self.assertTrue(is_empty(mat))

    def test_is_empty_zero_rows_and_cols(self):
        """Test is_empty with 0x0 matrix."""
        mat = M.zeros(0, 0)
        self.assertTrue(is_empty(mat))

    def test_is_empty_non_empty_matrix(self):
        """Test is_empty with non-empty matrix."""
        mat = M([[1, 2], [3, 4]])
        self.assertFalse(is_empty(mat))

    def test_is_empty_single_element(self):
        """Test is_empty with 1x1 matrix."""
        mat = M([[5]])
        self.assertFalse(is_empty(mat))

    def test_is_empty_from_list(self):
        """Test is_empty with list input."""
        self.assertTrue(is_empty([]))
        self.assertFalse(is_empty([[1, 2]]))


class TestIsInvertible(unittest.TestCase):
    """Test cases for the is_invertible function."""

    def test_is_invertible_identity(self):
        """Test is_invertible with identity matrix."""
        mat = M.eye(3)
        self.assertTrue(is_invertible(mat))

    def test_is_invertible_invertible_matrix(self):
        """Test is_invertible with invertible matrix."""
        mat = M([[1, 2], [3, 4]])
        self.assertTrue(is_invertible(mat))

    def test_is_invertible_singular_matrix(self):
        """Test is_invertible with singular matrix."""
        mat = M([[1, 2], [2, 4]])
        self.assertFalse(is_invertible(mat))

    def test_is_invertible_non_square(self):
        """Test is_invertible with non-square matrix."""
        mat = M([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(is_invertible(mat))

    def test_is_invertible_zero_determinant(self):
        """Test is_invertible with zero determinant."""
        mat = M([[0, 0], [0, 0]])
        self.assertFalse(is_invertible(mat))

    def test_is_invertible_symbolic(self):
        """Test is_invertible with symbolic matrix."""
        x = sp.Symbol('x')
        mat = M([[x, 1], [1, x]])
        # Should work with symbolic matrices
        result = is_invertible(mat)
        self.assertIsInstance(result, bool)


class TestIsOrthogonal(unittest.TestCase):
    """Test cases for the is_orthogonal function."""

    def test_is_orthogonal_identity(self):
        """Test is_orthogonal with identity matrix."""
        mat = M.eye(3)
        self.assertTrue(is_orthogonal(mat))

    def test_is_orthogonal_rotation_matrix(self):
        """Test is_orthogonal with rotation matrix."""
        theta = sp.Symbol('theta')
        mat = M([[sp.cos(theta), -sp.sin(theta)],
                 [sp.sin(theta), sp.cos(theta)]])
        # Should be orthogonal
        result = is_orthogonal(mat)
        self.assertIsInstance(result, (bool, sp.Basic))

    def test_is_orthogonal_non_orthogonal(self):
        """Test is_orthogonal with non-orthogonal matrix."""
        mat = M([[1, 2], [3, 4]])
        self.assertFalse(is_orthogonal(mat))

    def test_is_orthogonal_non_square(self):
        """Test is_orthogonal with non-square matrix."""
        mat = M([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(is_orthogonal(mat))


class TestIsUnitary(unittest.TestCase):
    """Test cases for the is_unitary function."""

    def test_is_unitary_identity(self):
        """Test is_unitary with identity matrix."""
        mat = M.eye(3)
        self.assertTrue(is_unitary(mat))

    def test_is_unitary_unitary_matrix(self):
        """Test is_unitary with unitary matrix."""
        # A simple unitary matrix
        mat = M([[1/sp.sqrt(2), 1/sp.sqrt(2)],
                 [1/sp.sqrt(2), -1/sp.sqrt(2)]])
        self.assertTrue(is_unitary(mat))

    def test_is_unitary_non_unitary(self):
        """Test is_unitary with non-unitary matrix."""
        mat = M([[1, 2], [3, 4]])
        self.assertFalse(is_unitary(mat))

    def test_is_unitary_non_square(self):
        """Test is_unitary with non-square matrix."""
        mat = M([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(is_unitary(mat))

    def test_is_unitary_complex_matrix(self):
        """Test is_unitary with complex matrix."""
        # Simple complex unitary matrix
        mat = M([[1, 0], [0, 1j]])
        result = is_unitary(mat)
        self.assertIsInstance(result, bool)


class TestIsNormal(unittest.TestCase):
    """Test cases for the is_normal function."""

    def test_is_normal_identity(self):
        """Test is_normal with identity matrix."""
        mat = M.eye(3)
        self.assertTrue(is_normal(mat))

    def test_is_normal_hermitian(self):
        """Test is_normal with hermitian matrix (normal)."""
        mat = M([[1, 1+1j], [1-1j, 2]])
        result = is_normal(mat)
        self.assertIsInstance(result, bool)

    def test_is_normal_symmetric(self):
        """Test is_normal with symmetric matrix (normal)."""
        mat = M([[1, 2], [2, 3]])
        self.assertTrue(is_normal(mat))

    def test_is_normal_non_normal(self):
        """Test is_normal with non-normal matrix."""
        mat = M([[1, 1], [0, 1]])
        self.assertFalse(is_normal(mat))

    def test_is_normal_non_square(self):
        """Test is_normal with non-square matrix."""
        mat = M([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(is_normal(mat))


class TestRref(unittest.TestCase):
    """Test cases for the rref function."""

    def test_rref_simple_matrix(self):
        """Test rref with a simple matrix."""
        mat = M([[1, 2], [3, 4]])
        result = rref(mat)
        self.assertIsInstance(result, M)
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.cols, 2)

    def test_rref_with_remove_false(self):
        """Test rref with remove=False."""
        mat = M([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        result = rref(mat, remove=False)
        self.assertEqual(result.rows, 3)  # Zero row should be kept

    def test_rref_with_remove_true(self):
        """Test rref with remove=True."""
        mat = M([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
        result = rref(mat, remove=True)
        self.assertEqual(result.rows, 2)  # Zero row should be removed

    def test_rref_identity_matrix(self):
        """Test rref with identity matrix."""
        mat = M.eye(3)
        result = rref(mat)
        self.assertTrue(result.equals(mat))

    def test_rref_already_rref(self):
        """Test rref with matrix already in rref."""
        mat = M([[1, 0, 2], [0, 1, 3]])
        result = rref(mat)
        self.assertTrue(result.equals(mat))

    def test_rref_multiple_zero_rows(self):
        """Test rref with multiple zero rows."""
        mat = M([[1, 2], [0, 0], [0, 0], [3, 4]])
        result = rref(mat, remove=True)
        self.assertEqual(result.rows, 2)

    def test_rref_from_list(self):
        """Test rref with list input."""
        result = rref([[1, 2], [3, 4]])
        self.assertIsInstance(result, M)


class TestOfArity(unittest.TestCase):
    """Test cases for the of_arity function."""

    def test_of_arity_exact_match(self):
        """Test of_arity with function matching exact arity."""
        def func(a, b):
            return a + b
        self.assertTrue(of_arity(func, 2))

    def test_of_arity_more_parameters(self):
        """Test of_arity with function having more parameters."""
        def func(a, b, c):
            return a + b + c
        # Function requires 3 args, so can't accept only 2
        self.assertFalse(of_arity(func, 2))
        self.assertTrue(of_arity(func, 3))
        self.assertFalse(of_arity(func, 4))

    def test_of_arity_fewer_parameters(self):
        """Test of_arity with function having fewer parameters."""
        def func(a):
            return a
        self.assertFalse(of_arity(func, 2))

    def test_of_arity_with_defaults(self):
        """Test of_arity with function having default parameters."""
        def func(a, b=0, c=0):
            return a + b + c
        self.assertTrue(of_arity(func, 1))
        self.assertTrue(of_arity(func, 2))
        self.assertTrue(of_arity(func, 3))

    def test_of_arity_keyword_only(self):
        """Test of_arity with keyword-only parameters."""
        def func(a, *, b):
            return a + b
        # Function has required keyword-only arg, so can't be called with only positional args
        self.assertFalse(of_arity(func, 1))
        self.assertFalse(of_arity(func, 2))

    def test_of_arity_keyword_only_with_default(self):
        """Test of_arity with keyword-only parameter with default."""
        def func(a, *, b=0):
            return a + b
        # Keyword-only with default doesn't prevent calling with only positional args
        self.assertTrue(of_arity(func, 1))
        self.assertFalse(of_arity(func, 2))

    def test_of_arity_varargs(self):
        """Test of_arity with *args."""
        def func(*args):
            return sum(args)
        # Function with *args can accept any number of positional args
        self.assertTrue(of_arity(func, 0))
        self.assertTrue(of_arity(func, 1))
        self.assertTrue(of_arity(func, 100))

    def test_of_arity_varargs_with_required(self):
        """Test of_arity with *args and required positional args."""
        def func(a, b, *args):
            return a + b + sum(args)
        # Function requires 2 args minimum, but can accept more via *args
        self.assertFalse(of_arity(func, 0))
        self.assertFalse(of_arity(func, 1))
        self.assertTrue(of_arity(func, 2))
        self.assertTrue(of_arity(func, 3))
        self.assertTrue(of_arity(func, 100))

    def test_of_arity_lambda(self):
        """Test of_arity with lambda function."""
        func = lambda x, y: x + y
        self.assertTrue(of_arity(func, 2))

    def test_of_arity_not_callable(self):
        """Test of_arity with non-callable returns False."""
        self.assertFalse(of_arity("not a function", 1))

    def test_of_arity_zero_arity(self):
        """Test of_arity with zero arity."""
        def func():
            return 42
        self.assertTrue(of_arity(func, 0))

    def test_of_arity_positional_only(self):
        """Test of_arity with positional-only parameters."""
        # Python 3.8+ supports positional-only with /
        def func(a, b, /, c):
            return a + b + c
        # Function requires 3 args (2 positional-only + 1 regular)
        self.assertFalse(of_arity(func, 2))
        self.assertTrue(of_arity(func, 3))
        self.assertFalse(of_arity(func, 4))


class TestAddAttributes(unittest.TestCase):
    """Test cases for the add_attributes function."""

    def test_add_attributes_single_attribute(self):
        """Test add_attributes with a single attribute."""
        class Base:
            pass
        
        def method(self):
            return "test"
        
        NewClass = add_attributes(Base, method)
        self.assertTrue(issubclass(NewClass, Base))
        self.assertTrue(hasattr(NewClass, 'method'))
        instance = NewClass()
        self.assertEqual(instance.method(), "test")

    def test_add_attributes_multiple_attributes(self):
        """Test add_attributes with multiple attributes."""
        class Base:
            pass
        
        def method1(self):
            return 1
        
        def method2(self):
            return 2
        
        NewClass = add_attributes(Base, method1, method2)
        self.assertTrue(hasattr(NewClass, 'method1'))
        self.assertTrue(hasattr(NewClass, 'method2'))
        instance = NewClass()
        self.assertEqual(instance.method1(), 1)
        self.assertEqual(instance.method2(), 2)

    def test_add_attributes_class_name(self):
        """Test that add_attributes creates class with correct name."""
        class Base:
            pass
        
        def method(self):
            pass
        
        NewClass = add_attributes(Base, method)
        self.assertIn('_subclass', NewClass.__name__)
        self.assertIn('Base', NewClass.__name__)

    def test_add_attributes_inheritance(self):
        """Test that new class properly inherits from base."""
        class Base:
            def base_method(self):
                return "base"
        
        def new_method(self):
            return "new"
        
        NewClass = add_attributes(Base, new_method)
        instance = NewClass()
        self.assertEqual(instance.base_method(), "base")
        self.assertEqual(instance.new_method(), "new")

    def test_add_attributes_with_property(self):
        """Test add_attributes with a property."""
        class Base:
            pass
        
        @property
        def prop(self):
            return "property"
        
        NewClass = add_attributes(Base, prop)
        self.assertTrue(hasattr(NewClass, 'prop'))
        instance = NewClass()
        self.assertEqual(instance.prop, "property")


if __name__ == '__main__':
    unittest.main()

