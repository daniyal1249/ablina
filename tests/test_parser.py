"""
Unit tests for the ablina.parser module.
"""

import unittest
import sympy as sp
from ablina import *


class TestParsingError(unittest.TestCase):
    """Test cases for the ParsingError exception."""

    def test_parsing_error_is_exception(self):
        """Test that ParsingError is an Exception."""
        self.assertTrue(issubclass(ParsingError, Exception))

    def test_parsing_error_creation_with_message(self):
        """Test creating ParsingError with a message."""
        error = ParsingError("Test error message")
        self.assertEqual(str(error), "Test error message")

    def test_parsing_error_creation_without_message(self):
        """Test creating ParsingError without a message."""
        error = ParsingError()
        self.assertEqual(str(error), "")


class TestConstraintError(unittest.TestCase):
    """Test cases for the ConstraintError exception."""

    def test_constraint_error_is_exception(self):
        """Test that ConstraintError is an Exception."""
        self.assertTrue(issubclass(ConstraintError, Exception))

    def test_constraint_error_creation_with_message(self):
        """Test creating ConstraintError with a message."""
        error = ConstraintError("Test error message")
        self.assertEqual(str(error), "Test error message")

    def test_constraint_error_creation_without_message(self):
        """Test creating ConstraintError without a message."""
        error = ConstraintError()
        self.assertEqual(str(error), "")


class TestSympify(unittest.TestCase):
    """Test cases for the sympify function."""

    def test_sympify_simple_expression(self):
        """Test sympify with a simple expression."""
        result = sympify("x + 1")
        self.assertIsInstance(result, sp.Basic)
        self.assertEqual(result, sp.Symbol('x') + 1)

    def test_sympify_numeric_expression(self):
        """Test sympify with a numeric expression."""
        result = sympify("2 + 3")
        self.assertEqual(result, 5)

    def test_sympify_rational_expression(self):
        """Test sympify with rational=True flag."""
        result = sympify("1/2")
        self.assertEqual(result, sp.Rational(1, 2))

    def test_sympify_with_allowed_vars_valid(self):
        """Test sympify with allowed variables that match."""
        x, y = sp.symbols('x y')
        allowed = [x, y]
        result = sympify("x + y", allowed_vars=allowed)
        self.assertEqual(result, x + y)

    def test_sympify_with_allowed_vars_invalid(self):
        """Test sympify with variables not in allowed_vars."""
        x = sp.symbols('x')
        allowed = [x]
        with self.assertRaises(ParsingError) as context:
            sympify("x + y", allowed_vars=allowed)
        self.assertIn("Unrecognized variables", str(context.exception))

    def test_sympify_with_allowed_vars_multiple_invalid(self):
        """Test sympify with multiple invalid variables."""
        x = sp.symbols('x')
        allowed = [x]
        with self.assertRaises(ParsingError) as context:
            sympify("x + y + z", allowed_vars=allowed)
        error_msg = str(context.exception)
        self.assertIn("Unrecognized variables", error_msg)
        # Should mention y and z
        self.assertTrue('y' in error_msg or 'z' in error_msg)

    def test_sympify_with_allowed_vars_none(self):
        """Test sympify with allowed_vars=None (no restriction)."""
        result = sympify("x + y + z", allowed_vars=None)
        self.assertIsInstance(result, sp.Basic)

    def test_sympify_with_allowed_vars_empty_set(self):
        """Test sympify with empty allowed_vars set."""
        with self.assertRaises(ParsingError):
            sympify("x", allowed_vars=set())

    def test_sympify_complex_expression(self):
        """Test sympify with a complex expression."""
        result = sympify("x**2 + 2*x + 1")
        x = sp.Symbol('x')
        self.assertEqual(result, x**2 + 2*x + 1)

    def test_sympify_with_functions(self):
        """Test sympify with mathematical functions."""
        result = sympify("sin(x) + cos(y)")
        x, y = sp.symbols('x y')
        self.assertIsInstance(result, sp.Basic)

    def test_sympify_with_constants(self):
        """Test sympify with constants like pi, e."""
        result = sympify("pi + E")
        self.assertIsInstance(result, sp.Basic)

    def test_sympify_allowed_vars_with_symbols(self):
        """Test sympify with allowed_vars as sympy symbols."""
        v0, v1, v2 = sp.symbols('v0 v1 v2')
        allowed = [v0, v1, v2]
        result = sympify("v0 + 2*v1", allowed_vars=allowed)
        self.assertEqual(result, v0 + 2*v1)

    def test_sympify_allowed_vars_with_strings(self):
        """Test sympify with allowed_vars as strings (should work if converted)."""
        # Note: This might not work directly, but let's test the behavior
        v0, v1 = sp.symbols('v0 v1')
        allowed = ['v0', 'v1']  # Strings won't match symbols
        with self.assertRaises(ParsingError):
            sympify("v0 + v1", allowed_vars=allowed)

    def test_sympify_no_variables(self):
        """Test sympify with expression containing no variables."""
        result = sympify("1 + 2 + 3")
        self.assertEqual(result, 6)

    def test_sympify_no_variables_with_allowed_vars(self):
        """Test sympify with no variables and allowed_vars."""
        x = sp.symbols('x')
        result = sympify("5", allowed_vars=[x])
        self.assertEqual(result, 5)


class TestSplitConstraint(unittest.TestCase):
    """Test cases for the split_constraint function."""

    def test_split_constraint_single_equals(self):
        """Test split_constraint with a single == operator."""
        result = split_constraint("x == 0")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 1)
        self.assertIn("x  - ( 0)", result)

    def test_split_constraint_multiple_equals(self):
        """Test split_constraint with multiple == operators."""
        result = split_constraint("x == y == z")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 2)
        self.assertIn("x  - ( y )", result)
        self.assertIn(" y  - ( z)", result)

    def test_split_constraint_three_equals(self):
        """Test split_constraint with three == operators."""
        result = split_constraint("a == b == c == d")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 3)
        self.assertIn("a  - ( b )", result)
        self.assertIn(" b  - ( c )", result)
        self.assertIn(" c  - ( d)", result)

    def test_split_constraint_with_expressions(self):
        """Test split_constraint with complex expressions."""
        result = split_constraint("x + 1 == 2*y == z - 3")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 2)
        self.assertIn("x + 1  - ( 2*y )", result)
        self.assertIn(" 2*y  - ( z - 3)", result)

    def test_split_constraint_with_spaces(self):
        """Test split_constraint with various spacing."""
        result = split_constraint("x == 0")
        self.assertIn("x  - ( 0)", result)
        
        result2 = split_constraint("x==0")
        # No spaces in input, so no spaces in output
        self.assertIn("x - (0)", result2)

    def test_split_constraint_no_equals(self):
        """Test split_constraint with no == operator raises error."""
        with self.assertRaises(ConstraintError) as context:
            split_constraint("x + 1")
        self.assertIn('at least one "=="', str(context.exception))

    def test_split_constraint_empty_string(self):
        """Test split_constraint with empty string raises error."""
        with self.assertRaises(ConstraintError):
            split_constraint("")

    def test_split_constraint_equals_at_start(self):
        """Test split_constraint with == at the start."""
        result = split_constraint("== x")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 1)
        self.assertIn(" - ( x)", result)

    def test_split_constraint_equals_at_end(self):
        """Test split_constraint with == at the end."""
        result = split_constraint("x ==")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 1)
        self.assertIn("x  - ()", result)

    def test_split_constraint_consecutive_equals(self):
        """Test split_constraint with consecutive == operators."""
        result = split_constraint("x === y")
        self.assertIsInstance(result, set)
        # Should split on each ==
        self.assertGreaterEqual(len(result), 1)

    def test_split_constraint_with_nested_parentheses(self):
        """Test split_constraint with nested parentheses in expressions."""
        result = split_constraint("(x + 1) == (y - 2)")
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 1)
        self.assertIn("(x + 1)  - ( (y - 2))", result)

    def test_split_constraint_result_is_set(self):
        """Test that split_constraint returns a set (no duplicates)."""
        # Even if we have the same expression, it should be a set
        result = split_constraint("x == 0")
        self.assertIsInstance(result, set)

    def test_split_constraint_multiple_same_relations(self):
        """Test split_constraint with multiple identical relations."""
        # "x == x == x" creates "x  - ( x )" and " x  - ( x)"
        # These are different strings due to spacing, so set has 2 elements
        result = split_constraint("x == x == x")
        self.assertIsInstance(result, set)
        # Note: Due to spacing differences, these are treated as different strings
        self.assertEqual(len(result), 2)
        self.assertIn("x  - ( x )", result)
        self.assertIn(" x  - ( x)", result)


if __name__ == '__main__':
    unittest.main()

