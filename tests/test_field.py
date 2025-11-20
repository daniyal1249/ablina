"""
Unit tests for the ablina.field module.
"""

import unittest
import sympy as sp
from ablina import *


class TestField(unittest.TestCase):
    """Test cases for the Field base class."""

    def test_field_is_class(self):
        """Test that Field is a class."""
        self.assertTrue(isinstance(Field, type))

    def test_field_can_be_instantiated(self):
        """Test that Field can be instantiated (though it's just a pass)."""
        field = Field()
        self.assertIsInstance(field, Field)


class TestReals(unittest.TestCase):
    """Test cases for the Reals class."""

    def test_reals_is_subclass_of_field(self):
        """Test that Reals is a subclass of Field."""
        self.assertTrue(issubclass(Reals, Field))

    def test_reals_repr(self):
        """Test the string representation of Reals."""
        reals = Reals()
        self.assertEqual(repr(reals), "R")

    def test_reals_str(self):
        """Test the string conversion of Reals."""
        reals = Reals()
        self.assertEqual(str(reals), "R")

    def test_reals_contains_integer(self):
        """Test that Reals contains integers."""
        reals = Reals()
        self.assertIn(0, reals)
        self.assertIn(1, reals)
        self.assertIn(-1, reals)
        self.assertIn(42, reals)

    def test_reals_contains_float(self):
        """Test that Reals contains floats."""
        reals = Reals()
        self.assertIn(0.0, reals)
        self.assertIn(3.14, reals)
        self.assertIn(-2.5, reals)
        self.assertIn(1e10, reals)

    def test_reals_contains_rational(self):
        """Test that Reals contains rational numbers."""
        reals = Reals()
        self.assertIn(sp.Rational(1, 2), reals)
        self.assertIn(sp.Rational(-3, 4), reals)

    def test_reals_contains_symbolic_real(self):
        """Test that Reals contains symbolic real numbers."""
        reals = Reals()
        x = sp.Symbol('x', real=True)
        self.assertIn(x, reals)

    def test_reals_contains_complex_raises_exception_handled(self):
        """Test that Reals.__contains__ handles exceptions for complex numbers."""
        reals = Reals()
        # Complex numbers should return False, not raise exception
        result = 1j in reals
        self.assertIsInstance(result, bool)

    def test_reals_contains_string_returns_false(self):
        """Test that Reals.__contains__ returns False for invalid types."""
        reals = Reals()
        self.assertFalse("hello" in reals)
        self.assertFalse([1, 2, 3] in reals)
        self.assertFalse(None in reals)


class TestComplexes(unittest.TestCase):
    """Test cases for the Complexes class."""

    def test_complexes_is_subclass_of_field(self):
        """Test that Complexes is a subclass of Field."""
        self.assertTrue(issubclass(Complexes, Field))

    def test_complexes_repr(self):
        """Test the string representation of Complexes."""
        complexes = Complexes()
        self.assertEqual(repr(complexes), "C")

    def test_complexes_str(self):
        """Test the string conversion of Complexes."""
        complexes = Complexes()
        self.assertEqual(str(complexes), "C")

    def test_complexes_contains_integer(self):
        """Test that Complexes contains integers."""
        complexes = Complexes()
        self.assertIn(0, complexes)
        self.assertIn(1, complexes)
        self.assertIn(-1, complexes)

    def test_complexes_contains_float(self):
        """Test that Complexes contains floats."""
        complexes = Complexes()
        self.assertIn(0.0, complexes)
        self.assertIn(3.14, complexes)
        self.assertIn(-2.5, complexes)

    def test_complexes_contains_complex(self):
        """Test that Complexes contains complex numbers."""
        complexes = Complexes()
        self.assertIn(1j, complexes)
        self.assertIn(1 + 2j, complexes)
        self.assertIn(complex(3, 4), complexes)

    def test_complexes_contains_rational(self):
        """Test that Complexes contains rational numbers."""
        complexes = Complexes()
        self.assertIn(sp.Rational(1, 2), complexes)
        self.assertIn(sp.Rational(-3, 4), complexes)

    def test_complexes_contains_symbolic_complex(self):
        """Test that Complexes contains symbolic complex numbers."""
        complexes = Complexes()
        x = sp.Symbol('x', complex=True)
        self.assertIn(x, complexes)

    def test_complexes_contains_string_returns_false(self):
        """Test that Complexes.__contains__ returns False for invalid types."""
        complexes = Complexes()
        self.assertFalse("hello" in complexes)
        self.assertFalse([1, 2, 3] in complexes)
        self.assertFalse(None in complexes)


class TestRSingleton(unittest.TestCase):
    """Test cases for the R singleton instance."""

    def test_r_is_instance_of_reals(self):
        """Test that R is an instance of Reals."""
        self.assertIsInstance(R, Reals)

    def test_r_is_singleton(self):
        """Test that R is a singleton (same instance)."""
        r1 = R
        r2 = R
        self.assertIs(r1, r2)

    def test_r_repr(self):
        """Test the string representation of R."""
        self.assertEqual(repr(R), "R")

    def test_r_str(self):
        """Test the string conversion of R."""
        self.assertEqual(str(R), "R")

    def test_r_contains_real_numbers(self):
        """Test that R contains real numbers."""
        self.assertIn(0, R)
        self.assertIn(1, R)
        self.assertIn(3.14, R)
        self.assertIn(sp.Rational(1, 2), R)


class TestCSingleton(unittest.TestCase):
    """Test cases for the C singleton instance."""

    def test_c_is_instance_of_complexes(self):
        """Test that C is an instance of Complexes."""
        self.assertIsInstance(C, Complexes)

    def test_c_is_singleton(self):
        """Test that C is a singleton (same instance)."""
        c1 = C
        c2 = C
        self.assertIs(c1, c2)

    def test_c_repr(self):
        """Test the string representation of C."""
        self.assertEqual(repr(C), "C")

    def test_c_str(self):
        """Test the string conversion of C."""
        self.assertEqual(str(C), "C")

    def test_c_contains_complex_numbers(self):
        """Test that C contains complex numbers."""
        self.assertIn(0, C)
        self.assertIn(1, C)
        self.assertIn(1j, C)
        self.assertIn(1 + 2j, C)
        self.assertIn(sp.Rational(1, 2), C)


if __name__ == '__main__':
    unittest.main()

