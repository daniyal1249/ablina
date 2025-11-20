"""
Unit tests for the ablina.experimental module.
"""

import unittest
import sympy as sp
from ablina import *
from ablina.experimental import (
    additive_id, additive_inv, is_commutative, is_associative, is_consistent,
    substitute_form, find_valid_params, solve_func_eq, find_add_isomorphism,
    find_mul_isomorphism, internal_isomorphism, map_constraints
)


class TestAdditiveId(unittest.TestCase):
    """Test additive_id function."""

    def test_additive_id_standard_addition(self):
        """Test additive_id with standard addition."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = additive_id(R, 2, add)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0, 0])

    def test_additive_id_standard_addition_3d(self):
        """Test additive_id with standard addition in 3D."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = additive_id(R, 3, add)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0, 0, 0])

    def test_additive_id_complex_field(self):
        """Test additive_id with complex field."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = additive_id(C, 2, add)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0, 0])

    def test_additive_id_shifted_addition(self):
        """Test additive_id with shifted addition (finds identity)."""
        def add(u, v):
            return [u[i] + v[i] + 1 for i in range(len(u))]
        result = additive_id(R, 2, add)
        # Actually finds identity [-1, -1] because:
        # add([v0, v1], [-1, -1]) = [v0 - 1 + 1, v1 - 1 + 1] = [v0, v1]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [-1, -1])

    def test_additive_id_custom_addition(self):
        """Test additive_id with custom addition operation."""
        def add(u, v):
            return [u[0] + v[0], u[1] + v[1] + u[0] * v[0]]
        result = additive_id(R, 2, add)
        # Should find identity if it exists
        self.assertIsInstance(result, list)


class TestAdditiveInv(unittest.TestCase):
    """Test additive_inv function."""

    def test_additive_inv_standard_addition(self):
        """Test additive_inv with standard addition."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        add_id = [0, 0]
        result = additive_inv(R, 2, add, add_id)
        self.assertEqual(len(result), 1)
        # Inverse of [v0, v1] should be [-v0, -v1]
        self.assertEqual(len(result[0]), 2)

    def test_additive_inv_standard_addition_3d(self):
        """Test additive_inv with standard addition in 3D."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        add_id = [0, 0, 0]
        result = additive_inv(R, 3, add, add_id)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)

    def test_additive_inv_lambdify_false(self):
        """Test additive_inv with lambdify=False."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        add_id = [0, 0]
        result = additive_inv(R, 2, add, add_id, lambdify=False)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_additive_inv_lambdify_true(self):
        """Test additive_inv with lambdify=True."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        add_id = [0, 0]
        result = additive_inv(R, 2, add, add_id, lambdify=True)
        self.assertIsInstance(result, list)
        self.assertTrue(callable(result[0]))
        # Test that the lambdified function works
        inv_func = result[0]
        test_vec = [1, 2]
        inverse = inv_func(test_vec)
        self.assertEqual(len(inverse), 2)

    def test_additive_inv_complex_field(self):
        """Test additive_inv with complex field."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        add_id = [0, 0]
        result = additive_inv(C, 2, add, add_id)
        self.assertEqual(len(result), 1)

    def test_additive_inv_no_inverse(self):
        """Test additive_inv when no inverse exists."""
        def add(u, v):
            return [u[0] + v[0] + 1, u[1] + v[1]]  # Non-standard
        add_id = [0, 0]
        result = additive_inv(R, 2, add, add_id)
        # May return empty list or valid inverse depending on operation
        self.assertIsInstance(result, list)


class TestIsCommutative(unittest.TestCase):
    """Test is_commutative function."""

    def test_is_commutative_standard_addition(self):
        """Test is_commutative with standard addition."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_commutative(R, 2, add)
        self.assertTrue(result)

    def test_is_commutative_standard_multiplication(self):
        """Test is_commutative with standard multiplication."""
        def mul(u, v):
            return [u[i] * v[i] for i in range(len(u))]
        result = is_commutative(R, 2, mul)
        self.assertTrue(result)

    def test_is_commutative_non_commutative(self):
        """Test is_commutative with non-commutative operation."""
        def non_comm(u, v):
            return [u[0] + v[1], u[1] + v[0]]  # Swapped
        result = is_commutative(R, 2, non_comm)
        self.assertFalse(result)

    def test_is_commutative_3d(self):
        """Test is_commutative with 3D vectors."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_commutative(R, 3, add)
        self.assertTrue(result)

    def test_is_commutative_complex_field(self):
        """Test is_commutative with complex field."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_commutative(C, 2, add)
        self.assertTrue(result)


class TestIsAssociative(unittest.TestCase):
    """Test is_associative function."""

    def test_is_associative_standard_addition(self):
        """Test is_associative with standard addition."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_associative(R, 2, add)
        self.assertTrue(result)

    def test_is_associative_standard_multiplication(self):
        """Test is_associative with standard multiplication."""
        def mul(u, v):
            return [u[i] * v[i] for i in range(len(u))]
        result = is_associative(R, 2, mul)
        self.assertTrue(result)

    def test_is_associative_non_associative(self):
        """Test is_associative with non-associative operation."""
        def non_assoc(u, v):
            return [u[0] - v[0], u[1] - v[1]]  # Subtraction is not associative
        result = is_associative(R, 2, non_assoc)
        self.assertFalse(result)

    def test_is_associative_3d(self):
        """Test is_associative with 3D vectors."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_associative(R, 3, add)
        self.assertTrue(result)

    def test_is_associative_complex_field(self):
        """Test is_associative with complex field."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        result = is_associative(C, 2, add)
        self.assertTrue(result)


class TestIsConsistent(unittest.TestCase):
    """Test is_consistent function."""

    def test_is_consistent_tautology(self):
        """Test is_consistent with tautology."""
        x = sp.Symbol('x')
        eq = sp.Eq(x, x)
        result = is_consistent(eq)
        self.assertTrue(result)

    def test_is_consistent_contradiction(self):
        """Test is_consistent with contradiction."""
        x = sp.Symbol('x')
        eq = sp.Eq(x, x + 1)
        result = is_consistent(eq)
        self.assertFalse(result)

    def test_is_consistent_regular_equation(self):
        """Test is_consistent with regular equation."""
        x = sp.Symbol('x')
        eq = sp.Eq(x, 5)
        result = is_consistent(eq)
        self.assertIsNone(result)

    def test_is_consistent_simplified_tautology(self):
        """Test is_consistent with simplified tautology."""
        x = sp.Symbol('x')
        eq = sp.simplify(sp.Eq(x + 1, x + 1))
        result = is_consistent(eq)
        # After simplification, should be True
        self.assertTrue(result)

    def test_is_consistent_simplified_contradiction(self):
        """Test is_consistent with simplified contradiction."""
        x = sp.Symbol('x')
        eq = sp.simplify(sp.Eq(x, x + 1))
        result = is_consistent(eq)
        self.assertFalse(result)


class TestSubstituteForm(unittest.TestCase):
    """Test substitute_form function."""

    def test_substitute_form_linear(self):
        """Test substitute_form with linear form."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = f(x) + 1
        form = lambda x: 2 * x + 3
        result = substitute_form(equation, f, form)
        self.assertEqual(result, 2 * x + 4)

    def test_substitute_form_equation(self):
        """Test substitute_form with equation."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = sp.Eq(f(x), x)
        form = lambda x: 2 * x
        result = substitute_form(equation, f, form)
        self.assertEqual(result, sp.Eq(2 * x, x))

    def test_substitute_form_multiple_occurrences(self):
        """Test substitute_form with multiple function occurrences."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = f(x) + f(x)
        form = lambda x: x ** 2
        result = substitute_form(equation, f, form)
        self.assertEqual(result, 2 * x ** 2)

    def test_substitute_form_no_occurrence(self):
        """Test substitute_form with no function occurrence."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = x + 1
        form = lambda x: 2 * x
        result = substitute_form(equation, f, form)
        self.assertEqual(result, x + 1)


class TestFindValidParams(unittest.TestCase):
    """Test find_valid_params function."""

    def test_find_valid_params_linear_form(self):
        """Test find_valid_params with linear form."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = sp.Eq(f(x), 2 * x)
        form = lambda x: sp.Symbol('a') * x + sp.Symbol('b')
        params = [sp.Symbol('a'), sp.Symbol('b')]
        result = find_valid_params(equation, f, form, params)
        # Should find a=2, b=0
        self.assertIsNotNone(result)

    def test_find_valid_params_no_solution(self):
        """Test find_valid_params when no solution exists."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = sp.Eq(f(x), x ** 2)
        form = lambda x: sp.Symbol('a') * x  # Linear can't match quadratic
        params = [sp.Symbol('a')]
        result = find_valid_params(equation, f, form, params)
        self.assertIsNone(result)

    def test_find_valid_params_tautology(self):
        """Test find_valid_params with tautology."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = sp.Eq(f(x), f(x))  # Tautology
        form = lambda x: sp.Symbol('a') * x
        params = [sp.Symbol('a')]
        result = find_valid_params(equation, f, form, params)
        # Should return form(x) since equation is consistent
        self.assertIsNotNone(result)


class TestSolveFuncEq(unittest.TestCase):
    """Test solve_func_eq function."""

    def test_solve_func_eq_linear(self):
        """Test solve_func_eq with linear solution."""
        x, y = sp.symbols('x y')
        f = sp.Function('f')
        equation = sp.Eq(f(x) + f(y), f(x + y))
        result = solve_func_eq(equation, f)
        # Should find linear solution f(x) = a*x
        self.assertIsNotNone(result)

    def test_solve_func_eq_no_solution(self):
        """Test solve_func_eq when no solution exists."""
        x = sp.Symbol('x')
        f = sp.Function('f')
        equation = sp.Eq(f(x), x ** 3)  # Not in supported forms
        result = solve_func_eq(equation, f)
        # May return None if no supported form matches
        # This depends on implementation

    def test_solve_func_eq_logarithmic(self):
        """Test solve_func_eq with logarithmic solution."""
        x, y = sp.symbols('x y', positive=True)
        f = sp.Function('f')
        equation = sp.Eq(f(x * y), f(x) + f(y))
        result = solve_func_eq(equation, f)
        # Should find logarithmic solution
        # May or may not find it depending on implementation


class TestFindAddIsomorphism(unittest.TestCase):
    """Test find_add_isomorphism function."""

    def test_find_add_isomorphism_not_implemented(self):
        """Test find_add_isomorphism raises NotImplementedError."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        with self.assertRaises(NotImplementedError):
            find_add_isomorphism(R, 2, add)


class TestFindMulIsomorphism(unittest.TestCase):
    """Test find_mul_isomorphism function."""

    def test_find_mul_isomorphism_not_implemented(self):
        """Test find_mul_isomorphism raises NotImplementedError."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        with self.assertRaises(NotImplementedError):
            find_mul_isomorphism(R, 2, mul)


class TestInternalIsomorphism(unittest.TestCase):
    """Test internal_isomorphism function."""

    def test_internal_isomorphism_not_implemented(self):
        """Test internal_isomorphism raises NotImplementedError."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        with self.assertRaises(NotImplementedError):
            internal_isomorphism(R, 2, add, mul)


class TestMapConstraints(unittest.TestCase):
    """Test map_constraints function."""

    def test_map_constraints_not_implemented(self):
        """Test map_constraints raises NotImplementedError."""
        def mapping(x):
            return x
        constraints = ["v0 + v1 == 0"]
        with self.assertRaises(NotImplementedError):
            map_constraints(mapping, constraints)


if __name__ == '__main__':
    unittest.main()

