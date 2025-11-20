"""
Unit tests for the ablina.operations module.
"""

import unittest
from ablina import *
from ablina.operations import OperationError, Operation, VectorAdd, ScalarMul


class TestOperationError(unittest.TestCase):
    """Test OperationError exception."""

    def test_operation_error_init_no_message(self):
        """Test OperationError initialization with no message."""
        error = OperationError()
        self.assertEqual(str(error), "")

    def test_operation_error_init_with_message(self):
        """Test OperationError initialization with message."""
        error = OperationError("Custom error message")
        self.assertEqual(str(error), "Custom error message")

    def test_operation_error_is_exception(self):
        """Test OperationError is an Exception."""
        self.assertIsInstance(OperationError(), Exception)


class TestOperation(unittest.TestCase):
    """Test Operation class."""

    def test_operation_init_valid(self):
        """Test Operation initialization with valid function and arity."""
        def add(x, y):
            return x + y
        op = Operation(add, 2)
        self.assertEqual(op.arity, 2)
        self.assertEqual(op.func, add)

    def test_operation_init_wrong_arity(self):
        """Test Operation initialization with wrong arity raises OperationError."""
        def add(x, y):
            return x + y
        with self.assertRaises(OperationError):
            Operation(add, 1)
        with self.assertRaises(OperationError):
            Operation(add, 3)

    def test_operation_init_zero_arity(self):
        """Test Operation initialization with zero arity."""
        def constant():
            return 42
        op = Operation(constant, 0)
        self.assertEqual(op.arity, 0)
        self.assertEqual(op.func, constant)

    def test_operation_init_with_defaults(self):
        """Test Operation initialization with function having default parameters."""
        def func(x, y=0):
            return x + y
        # Function can accept 1 or 2 args, so arity 1 should work
        op = Operation(func, 1)
        self.assertEqual(op.arity, 1)

    def test_operation_init_with_keyword_only(self):
        """Test Operation initialization with keyword-only parameters."""
        def func(x, *, y):
            return x + y
        # Function has required keyword-only param, so can't accept only 1 positional arg
        with self.assertRaises(OperationError):
            Operation(func, 1)

    def test_operation_init_with_varargs(self):
        """Test Operation initialization with *args."""
        def func(*args):
            return sum(args)
        # Function with *args can accept any number >= 0
        op0 = Operation(func, 0)
        self.assertEqual(op0.arity, 0)
        op1 = Operation(func, 1)
        self.assertEqual(op1.arity, 1)
        op100 = Operation(func, 100)
        self.assertEqual(op100.arity, 100)

    def test_operation_func_property(self):
        """Test func property."""
        def my_func(x):
            return x * 2
        op = Operation(my_func, 1)
        self.assertEqual(op.func, my_func)

    def test_operation_arity_property(self):
        """Test arity property."""
        def my_func(x, y, z):
            return x + y + z
        op = Operation(my_func, 3)
        self.assertEqual(op.arity, 3)

    def test_operation_call(self):
        """Test Operation.__call__ method."""
        def add(x, y):
            return x + y
        op = Operation(add, 2)
        self.assertEqual(op(3, 5), 8)
        self.assertEqual(op(10, 20), 30)

    def test_operation_call_zero_arity(self):
        """Test Operation.__call__ with zero arity."""
        def constant():
            return 42
        op = Operation(constant, 0)
        self.assertEqual(op(), 42)

    def test_operation_call_varargs(self):
        """Test Operation.__call__ with *args function."""
        def sum_all(*args):
            return sum(args)
        op = Operation(sum_all, 3)
        self.assertEqual(op(1, 2, 3), 6)
        self.assertEqual(op(10, 20, 30), 60)


class TestVectorAdd(unittest.TestCase):
    """Test VectorAdd class."""

    def test_vector_add_init_valid(self):
        """Test VectorAdd initialization with valid function."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 3, add)
        self.assertEqual(vec_add.field, R)
        self.assertEqual(vec_add.n, 3)
        self.assertEqual(vec_add.arity, 2)

    def test_vector_add_init_wrong_arity(self):
        """Test VectorAdd initialization with wrong arity raises OperationError."""
        def wrong_func(u):
            return u
        with self.assertRaises(OperationError):
            VectorAdd(R, 3, wrong_func)

    def test_vector_add_field_property(self):
        """Test field property."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(C, 2, add)
        self.assertEqual(vec_add.field, C)

    def test_vector_add_n_property(self):
        """Test n property."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 5, add)
        self.assertEqual(vec_add.n, 5)

    def test_vector_add_call(self):
        """Test VectorAdd.__call__ method."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 3, add)
        result = vec_add([1, 2, 3], [4, 5, 6])
        self.assertEqual(result, [5, 7, 9])

    def test_vector_add_eq_same_instance(self):
        """Test VectorAdd.__eq__ with same instance."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 2, add)
        self.assertEqual(vec_add, vec_add)

    def test_vector_add_eq_identical_operations(self):
        """Test VectorAdd.__eq__ with identical operations."""
        def add1(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        def add2(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add1 = VectorAdd(R, 2, add1)
        vec_add2 = VectorAdd(R, 2, add2)
        # Should return True for identical operations
        self.assertEqual(vec_add1, vec_add2)

    def test_vector_add_eq_different_operations(self):
        """Test VectorAdd.__eq__ with different operations."""
        def add1(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        def add2(u, v):
            return [u[i] * v[i] for i in range(len(u))]  # Different operation
        vec_add1 = VectorAdd(R, 2, add1)
        vec_add2 = VectorAdd(R, 2, add2)
        # Should return False for different operations
        self.assertNotEqual(vec_add1, vec_add2)

    def test_vector_add_eq_different_n(self):
        """Test VectorAdd.__eq__ with different n."""
        def add1(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        def add2(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add1 = VectorAdd(R, 2, add1)
        vec_add2 = VectorAdd(R, 3, add2)
        # Different n should return False
        self.assertFalse(vec_add1 == vec_add2)

    def test_vector_add_eq_different_field(self):
        """Test VectorAdd.__eq__ with different field."""
        def add1(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        def add2(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add1 = VectorAdd(R, 2, add1)
        vec_add2 = VectorAdd(C, 2, add2)
        # Different field should return False
        self.assertFalse(vec_add1 == vec_add2)

    def test_vector_add_eq_with_non_vector_add(self):
        """Test VectorAdd.__eq__ with non-VectorAdd object."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 2, add)
        # Should return False for non-VectorAdd objects
        self.assertFalse(vec_add == "not a VectorAdd")
        self.assertFalse(vec_add == 42)

    def test_vector_add_inheritance(self):
        """Test VectorAdd inherits from Operation."""
        def add(u, v):
            return [u[i] + v[i] for i in range(len(u))]
        vec_add = VectorAdd(R, 2, add)
        self.assertIsInstance(vec_add, Operation)


class TestScalarMul(unittest.TestCase):
    """Test ScalarMul class."""

    def test_scalar_mul_init_valid(self):
        """Test ScalarMul initialization with valid function."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 3, mul)
        self.assertEqual(scalar_mul.field, R)
        self.assertEqual(scalar_mul.n, 3)
        self.assertEqual(scalar_mul.arity, 2)

    def test_scalar_mul_init_wrong_arity(self):
        """Test ScalarMul initialization with wrong arity raises OperationError."""
        def wrong_func(v):
            return v
        with self.assertRaises(OperationError):
            ScalarMul(R, 3, wrong_func)

    def test_scalar_mul_field_property(self):
        """Test field property."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(C, 2, mul)
        self.assertEqual(scalar_mul.field, C)

    def test_scalar_mul_n_property(self):
        """Test n property."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 5, mul)
        self.assertEqual(scalar_mul.n, 5)

    def test_scalar_mul_call(self):
        """Test ScalarMul.__call__ method."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 3, mul)
        result = scalar_mul(2, [1, 2, 3])
        self.assertEqual(result, [2, 4, 6])

    def test_scalar_mul_eq_same_instance(self):
        """Test ScalarMul.__eq__ with same instance."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 2, mul)
        self.assertEqual(scalar_mul, scalar_mul)

    def test_scalar_mul_eq_identical_operations(self):
        """Test ScalarMul.__eq__ with identical operations."""
        def mul1(c, v):
            return [c * v[i] for i in range(len(v))]
        def mul2(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul1 = ScalarMul(R, 2, mul1)
        scalar_mul2 = ScalarMul(R, 2, mul2)
        # Should return True for identical operations
        self.assertEqual(scalar_mul1, scalar_mul2)

    def test_scalar_mul_eq_different_operations(self):
        """Test ScalarMul.__eq__ with different operations."""
        def mul1(c, v):
            return [c * v[i] for i in range(len(v))]
        def mul2(c, v):
            return [c + v[i] for i in range(len(v))]  # Different operation
        scalar_mul1 = ScalarMul(R, 2, mul1)
        scalar_mul2 = ScalarMul(R, 2, mul2)
        # Should return False for different operations
        self.assertNotEqual(scalar_mul1, scalar_mul2)

    def test_scalar_mul_eq_different_n(self):
        """Test ScalarMul.__eq__ with different n."""
        def mul1(c, v):
            return [c * v[i] for i in range(len(v))]
        def mul2(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul1 = ScalarMul(R, 2, mul1)
        scalar_mul2 = ScalarMul(R, 3, mul2)
        # Different n should return False
        self.assertFalse(scalar_mul1 == scalar_mul2)

    def test_scalar_mul_eq_different_field(self):
        """Test ScalarMul.__eq__ with different field."""
        def mul1(c, v):
            return [c * v[i] for i in range(len(v))]
        def mul2(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul1 = ScalarMul(R, 2, mul1)
        scalar_mul2 = ScalarMul(C, 2, mul2)
        # Different field should return False
        self.assertFalse(scalar_mul1 == scalar_mul2)

    def test_scalar_mul_eq_with_non_scalar_mul(self):
        """Test ScalarMul.__eq__ with non-ScalarMul object."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 2, mul)
        # Should return False for non-ScalarMul objects
        self.assertFalse(scalar_mul == "not a ScalarMul")
        self.assertFalse(scalar_mul == 42)

    def test_scalar_mul_inheritance(self):
        """Test ScalarMul inherits from Operation."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(R, 2, mul)
        self.assertIsInstance(scalar_mul, Operation)

    def test_scalar_mul_complex_field(self):
        """Test ScalarMul with complex field."""
        def mul(c, v):
            return [c * v[i] for i in range(len(v))]
        scalar_mul = ScalarMul(C, 2, mul)
        result = scalar_mul(1+1j, [1+2j, 3+4j])
        # (1+1j) * (1+2j) = 1+2j+1j+2j^2 = 1+3j-2 = -1+3j
        # (1+1j) * (3+4j) = 3+4j+3j+4j^2 = 3+7j-4 = -1+7j
        self.assertEqual(result[0], (-1+3j))
        self.assertEqual(result[1], (-1+7j))


if __name__ == '__main__':
    unittest.main()

