"""
Unit tests for the ablina.vectorspace module.
"""

import unittest
import sympy as sp
from ablina import *


class TestFnInit(unittest.TestCase):
    """Test Fn.__init__ method."""

    def test_init_basic_real_field(self):
        """Test Fn initialization with real field and no constraints."""
        fn = Fn(R, 3)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_basic_complex_field(self):
        """Test Fn initialization with complex field and no constraints."""
        fn = Fn(C, 2)
        self.assertEqual(fn.field, C)
        self.assertEqual(fn.n, 2)

    def test_init_with_constraints(self):
        """Test Fn initialization with valid constraints."""
        constraints = ["v0 + 2*v1 == 0"]
        fn = Fn(R, 2, constraints=constraints)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 2)

    def test_init_with_multiple_constraints(self):
        """Test Fn initialization with multiple valid constraints."""
        constraints = ["v0 + v1 == 0", "v1 + v2 == 0"]
        fn = Fn(R, 3, constraints=constraints)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_with_empty_constraints(self):
        """Test Fn initialization with empty constraints list."""
        fn = Fn(R, 2, constraints=[])
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 2)

    def test_init_with_none_constraints(self):
        """Test Fn initialization with None constraints."""
        fn = Fn(R, 2, constraints=None)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 2)

    def test_init_with_ns_matrix(self):
        """Test Fn initialization with ns_matrix."""
        ns_matrix = M([[1, 1, 0]])
        fn = Fn(R, 3, ns_matrix=ns_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_with_rs_matrix(self):
        """Test Fn initialization with rs_matrix."""
        rs_matrix = M([[1, 0, 0], [0, 1, 0]])
        fn = Fn(R, 3, rs_matrix=rs_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_with_both_matrices(self):
        """Test Fn initialization with both ns_matrix and rs_matrix."""
        ns_matrix = M([[1, 1, 0]])
        rs_matrix = M([[1, -1, 0], [0, 0, 1]])
        fn = Fn(R, 3, ns_matrix=ns_matrix, rs_matrix=rs_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_zero_dimension(self):
        """Test Fn initialization with zero dimension."""
        fn = Fn(R, 0)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 0)

    def test_init_one_dimension(self):
        """Test Fn initialization with one dimension."""
        fn = Fn(R, 1)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 1)

    def test_init_large_dimension(self):
        """Test Fn initialization with large dimension."""
        fn = Fn(R, 100)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 100)

    def test_init_invalid_field_type(self):
        """Test Fn initialization with invalid field type raises TypeError."""
        with self.assertRaises(TypeError):
            Fn("not a field", 2)

    def test_init_invalid_field_value(self):
        """Test Fn initialization with invalid field value raises TypeError."""
        with self.assertRaises(TypeError):
            Fn(int, 2)

    def test_init_invalid_constraints_non_linear(self):
        """Test Fn initialization with non-linear constraints raises NotAVectorSpaceError."""
        constraints = ["v0 * v1 == 0"]  # Non-linear
        with self.assertRaises(NotAVectorSpaceError):
            Fn(R, 2, constraints=constraints)

    def test_init_invalid_constraints_nonzero_constant(self):
        """Test Fn initialization with constraints having nonzero constant raises NotAVectorSpaceError."""
        constraints = ["v0 + v1 == 1"]  # Nonzero constant
        with self.assertRaises(NotAVectorSpaceError):
            Fn(R, 2, constraints=constraints)

    def test_init_invalid_constraints_quadratic(self):
        """Test Fn initialization with quadratic constraints raises NotAVectorSpaceError."""
        constraints = ["v0**2 == 0"]  # Quadratic
        with self.assertRaises(NotAVectorSpaceError):
            Fn(R, 2, constraints=constraints)

    def test_init_valid_constraints_linear_homogeneous(self):
        """Test Fn initialization with valid linear homogeneous constraints."""
        constraints = ["v0 + 2*v1 == 0", "v1 - v2 == 0"]
        fn = Fn(R, 3, constraints=constraints)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_with_ns_matrix_bypasses_validation(self):
        """Test that providing ns_matrix bypasses constraint validation."""
        # This should work even though constraint would be invalid
        ns_matrix = M([[1, 1]])
        fn = Fn(R, 2, constraints=["invalid"], ns_matrix=ns_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 2)

    def test_init_with_rs_matrix_bypasses_validation(self):
        """Test that providing rs_matrix bypasses constraint validation."""
        # This should work even though constraint would be invalid
        rs_matrix = M([[1, 0]])
        fn = Fn(R, 2, constraints=["invalid"], rs_matrix=rs_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 2)

    def test_init_empty_ns_matrix(self):
        """Test Fn initialization with empty ns_matrix."""
        ns_matrix = M.zeros(0, 3)
        fn = Fn(R, 3, ns_matrix=ns_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_empty_rs_matrix(self):
        """Test Fn initialization with empty rs_matrix."""
        rs_matrix = M.zeros(0, 3)
        fn = Fn(R, 3, rs_matrix=rs_matrix)
        self.assertEqual(fn.field, R)
        self.assertEqual(fn.n, 3)

    def test_init_complex_field_with_constraints(self):
        """Test Fn initialization with complex field and constraints."""
        constraints = ["v0 + v1 == 0"]
        fn = Fn(C, 2, constraints=constraints)
        self.assertEqual(fn.field, C)
        self.assertEqual(fn.n, 2)


class TestVectorSpaceInit(unittest.TestCase):
    """Test VectorSpace.__init__ method."""

    @classmethod
    def setUpClass(cls):
        """Create a test VectorSpace subclass for testing."""
        # Create a simple VectorSpace subclass for testing
        class TestVS(VectorSpace):
            set = Set("TestVS", object, lambda vec: isinstance(vec, list) and len(vec) == 2)
            fn = Fn(R, 2)
            
            @staticmethod
            def __push__(vec):
                return M(vec)
            
            @staticmethod
            def __pull__(vec):
                return vec.tolist()
        
        cls.TestVS = TestVS

    def test_init_basic(self):
        """Test VectorSpace initialization with just a name."""
        vs = self.TestVS("V")
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_constraints(self):
        """Test VectorSpace initialization with constraints."""
        constraints = ["v0 + v1 == 0"]
        vs = self.TestVS("V", constraints=constraints)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_basis(self):
        """Test VectorSpace initialization with valid basis."""
        basis = [[1, 0], [0, 1]]
        vs = self.TestVS("V", basis=basis)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_fn(self):
        """Test VectorSpace initialization with fn parameter."""
        fn = Fn(R, 2, constraints=["v0 + v1 == 0"])
        vs = self.TestVS("V", fn=fn)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)
        self.assertEqual(vs.fn, fn)

    def test_init_with_fn_bypasses_constraints(self):
        """Test that providing fn bypasses constraints and basis."""
        fn = Fn(R, 2)
        vs = self.TestVS("V", constraints=["invalid"], basis=[[1, 1]], fn=fn)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.fn, fn)

    def test_init_with_constraints_and_basis(self):
        """Test VectorSpace initialization with both constraints and basis."""
        constraints = ["v0 + v1 == 0"]
        basis = [[1, -1]]
        vs = self.TestVS("V", constraints=constraints, basis=basis)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_dependent_basis_raises_error(self):
        """Test VectorSpace initialization with linearly dependent basis raises ValueError."""
        basis = [[1, 0], [2, 0]]  # Linearly dependent
        with self.assertRaises(ValueError) as context:
            self.TestVS("V", basis=basis)
        self.assertIn("linearly independent", str(context.exception))

    def test_init_with_empty_basis(self):
        """Test VectorSpace initialization with empty basis."""
        basis = []
        vs = self.TestVS("V", basis=basis)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_none_constraints(self):
        """Test VectorSpace initialization with None constraints."""
        vs = self.TestVS("V", constraints=None)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_with_none_basis(self):
        """Test VectorSpace initialization with None basis."""
        vs = self.TestVS("V", basis=None)
        self.assertEqual(vs.name, "V")
        self.assertEqual(vs.field, R)

    def test_init_name_assignment(self):
        """Test that name is correctly assigned."""
        vs = self.TestVS("MyVectorSpace")
        self.assertEqual(vs.name, "MyVectorSpace")

    def test_init_set_creation(self):
        """Test that set is correctly created."""
        vs = self.TestVS("V")
        self.assertIsNotNone(vs.set)
        self.assertEqual(vs.set.name, "V")


class TestVectorSpaceProperties(unittest.TestCase):
    """Test VectorSpace properties."""

    def test_field_property(self):
        """Test field property."""
        V = fn("V", R, 3)
        self.assertEqual(V.field, R)
        
        V = fn("V", C, 2)
        self.assertEqual(V.field, C)
        
        M_space = matrix_space("M", R, (2, 2))
        self.assertEqual(M_space.field, R)
        
        P = poly_space("P", R, 3)
        self.assertEqual(P.field, R)

    def test_add_property(self):
        """Test add property returns callable."""
        V = fn("V", R, 3)
        self.assertTrue(callable(V.add))
        
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        result = V.add(v1, v2)
        self.assertEqual(M(result), M([5, 7, 9]))

    def test_mul_property(self):
        """Test mul property returns callable."""
        V = fn("V", R, 3)
        self.assertTrue(callable(V.mul))
        
        v = [1, 2, 3]
        result = V.mul(2, v)
        self.assertEqual(M(result), M([2, 4, 6]))
        
        result = V.mul(0, v)
        self.assertEqual(M(result), M([0, 0, 0]))
        
        result = V.mul(-1, v)
        self.assertEqual(M(result), M([-1, -2, -3]))

    def test_additive_inv_property(self):
        """Test additive_inv property returns callable."""
        V = fn("V", R, 3)
        self.assertTrue(callable(V.additive_inv))
        
        v = [1, 2, 3]
        result = V.additive_inv(v)
        self.assertEqual(M(result), M([-1, -2, -3]))
        
        result = V.additive_inv([0, 0, 0])
        self.assertEqual(M(result), M([0, 0, 0]))

    def test_additive_id_property(self):
        """Test additive_id property."""
        V = fn("V", R, 3)
        zero = V.additive_id
        self.assertEqual(M(zero), M([0, 0, 0]))
        
        V = fn("V", R, 2, constraints=["v0 + v1 == 0"])
        zero = V.additive_id
        self.assertEqual(M(zero), M([0, 0]))
        
        V = fn("V", R, 0)
        zero = V.additive_id
        self.assertEqual(zero.shape, (0, 1))
        
        M_space = matrix_space("M", R, (2, 2))
        zero = M_space.additive_id
        self.assertEqual(M(zero).shape, (2, 2))
        self.assertEqual(M(zero), M.zeros(2, 2))

    def test_basis_property(self):
        """Test basis property."""
        V = fn("V", R, 3)
        basis = V.basis
        self.assertEqual(len(basis), 3)
        self.assertTrue(V.is_basis(*basis))
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        basis = V.basis
        self.assertEqual(len(basis), 2)
        self.assertTrue(V.is_basis(*basis))
        
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        basis = V.basis
        self.assertEqual(len(basis), 0)
        
        M_space = matrix_space("M", R, (2, 2))
        basis = M_space.basis
        self.assertEqual(len(basis), 4)

    def test_dim_property(self):
        """Test dim property."""
        V = fn("V", R, 3)
        self.assertEqual(V.dim, 3)
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        self.assertEqual(V.dim, 2)
        
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        self.assertEqual(V.dim, 0)
        
        V = fn("V", R, 0)
        self.assertEqual(V.dim, 0)
        
        M_space = matrix_space("M", R, (2, 2))
        self.assertEqual(M_space.dim, 4)
        
        P = poly_space("P", R, 3)
        self.assertEqual(P.dim, 4)


class TestVectorSpaceSpecialMethods(unittest.TestCase):
    """Test VectorSpace special methods."""

    def test_repr(self):
        """Test __repr__ method."""
        V = fn("V", R, 3)
        repr_str = repr(V)
        self.assertIn("V", repr_str)
        self.assertIn("basis", repr_str)
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        repr_str = repr(V)
        self.assertIn("V", repr_str)

    def test_str(self):
        """Test __str__ method."""
        V = fn("V", R, 3)
        self.assertEqual(str(V), "V")
        
        V = fn("MySpace", R, 2)
        self.assertEqual(str(V), "MySpace")

    def test_eq_same_space(self):
        """Test __eq__ with same space."""
        V = fn("V", R, 3)
        self.assertEqual(V, V)
        
        V1 = fn("V1", R, 3)
        V2 = fn("V2", R, 3)
        self.assertEqual(V1, V2)

    def test_eq_different_dimensions(self):
        """Test __eq__ with different dimensions."""
        V1 = fn("V1", R, 2)
        V2 = fn("V2", R, 3)
        self.assertNotEqual(V1, V2)

    def test_eq_subspaces(self):
        """Test __eq__ with subspaces."""
        V = fn("V", R, 3)
        U = fn("U", R, 3, constraints=["v0 + v1 == 0"])
        W = fn("W", R, 3, constraints=["v0 + v1 == 0"])
        self.assertEqual(U, W)
        self.assertNotEqual(V, U)

    def test_contains(self):
        """Test __contains__ method."""
        V = fn("V", R, 3)
        self.assertIn([1, 2, 3], V)
        self.assertIn([0, 0, 0], V)
        self.assertNotIn([1, 2], V)
        self.assertNotIn([1, 2, 3, 4], V)
        self.assertNotIn("not a vector", V)
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        self.assertIn([1, -1, 0], V)
        self.assertIn([0, 0, 5], V)
        self.assertNotIn([1, 1, 0], V)
        
        M_space = matrix_space("M", R, (2, 2))
        self.assertIn([[1, 2], [3, 4]], M_space)
        self.assertNotIn([1, 2, 3, 4], M_space)

    def test_pos(self):
        """Test __pos__ method."""
        V = fn("V", R, 3)
        self.assertIs(+V, V)

    def test_neg(self):
        """Test __neg__ method."""
        V = fn("V", R, 3)
        self.assertIs(-V, V)

    def test_add_vector_space(self):
        """Test __add__ with vector space."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W = U + V
        self.assertEqual(W.dim, 3)
        self.assertIsInstance(W, VectorSpace)
        
        # Test with same space
        V = fn("V", R, 3)
        result = V + V
        self.assertEqual(result, V)

    def test_add_vector(self):
        """Test __add__ with vector (creates coset)."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coset = V + v
        self.assertIsInstance(coset, AffineSpace)
        self.assertEqual(coset.vectorspace, V)
        self.assertEqual(coset.representative, v)

    def test_radd(self):
        """Test __radd__ method."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coset = v + V
        self.assertIsInstance(coset, AffineSpace)

    def test_sub_vector_space(self):
        """Test __sub__ with vector space."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W = U - V
        self.assertEqual(W.dim, 3)
        self.assertIsInstance(W, VectorSpace)

    def test_sub_vector(self):
        """Test __sub__ with vector (creates coset)."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coset = V - v
        self.assertIsInstance(coset, AffineSpace)
        self.assertEqual(coset.vectorspace, V)
        self.assertEqual(M(coset.representative), M([-1, -2, -3]))

    def test_rsub(self):
        """Test __rsub__ method."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coset = v - V
        self.assertIsInstance(coset, AffineSpace)


    def test_and(self):
        """Test __and__ method."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        intersection = U & V
        self.assertIsInstance(intersection, VectorSpace)
        self.assertEqual(intersection.dim, 1)


class TestVectorSpaceVectorMethods(unittest.TestCase):
    """Test VectorSpace vector-related methods."""

    def test_vector_random(self):
        """Test vector method with random generation."""
        V = fn("V", R, 3)
        v = V.vector()
        self.assertIn(v, V)
        
        v = V.vector(std=10)
        self.assertIn(v, V)

    def test_vector_arbitrary(self):
        """Test vector method with arbitrary=True."""
        V = fn("V", R, 3)
        v = V.vector(arbitrary=True)
        self.assertIn(v, V)
        # Should contain symbols
        self.assertTrue(any(isinstance(coord, sp.Symbol) for coord in v))
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        v = V.vector(arbitrary=True)
        self.assertIn(v, V)

    def test_vector_zero_dimension(self):
        """Test vector method with zero dimension."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        v = V.vector()
        self.assertEqual(M(v), M([0, 0, 0]))
        
        v = V.vector(arbitrary=True)
        self.assertEqual(M(v), M([0, 0, 0]))

    def test_to_coordinate_default_basis(self):
        """Test to_coordinate with default basis."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coord = V.to_coordinate(v)
        self.assertEqual(coord.shape, (3, 1))
        # Should be identity for standard basis
        self.assertEqual(coord, M([1, 2, 3]))

    def test_to_coordinate_custom_basis(self):
        """Test to_coordinate with custom basis."""
        V = fn("V", R, 3)
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        v = [2, 3, 4]
        coord = V.to_coordinate(v, basis=basis)
        self.assertEqual(coord, M([2, 3, 4]))
        
        basis = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
        coord = V.to_coordinate([1, 2, 3], basis=basis)
        self.assertIsNotNone(coord)

    def test_to_coordinate_invalid_vector(self):
        """Test to_coordinate with invalid vector."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.to_coordinate([1, 2])  # Wrong dimension

    def test_to_coordinate_invalid_basis(self):
        """Test to_coordinate with invalid basis."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        with self.assertRaises(ValueError):
            V.to_coordinate(v, basis=[[1, 0, 0], [2, 0, 0]])  # Not a basis

    def test_from_coordinate_default_basis(self):
        """Test from_coordinate with default basis."""
        V = fn("V", R, 3)
        coord = M([1, 2, 3])
        v = V.from_coordinate(coord)
        self.assertEqual(M(v), M([1, 2, 3]))

    def test_from_coordinate_custom_basis(self):
        """Test from_coordinate with custom basis."""
        V = fn("V", R, 3)
        basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        coord = M([2, 3, 4])
        v = V.from_coordinate(coord, basis=basis)
        self.assertEqual(M(v), M([2, 3, 4]))

    def test_from_coordinate_invalid_shape(self):
        """Test from_coordinate with invalid coordinate shape."""
        V = fn("V", R, 3)
        with self.assertRaises(ValueError):
            V.from_coordinate(M([1, 2]))  # Wrong dimension

    def test_to_from_coordinate_roundtrip(self):
        """Test roundtrip conversion."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coord = V.to_coordinate(v)
        v2 = V.from_coordinate(coord)
        self.assertEqual(M(v), M(v2))
        
        basis = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
        coord = V.to_coordinate(v, basis=basis)
        v2 = V.from_coordinate(coord, basis=basis)
        self.assertEqual(M(v), M(v2))

    def test_is_independent(self):
        """Test is_independent method."""
        V = fn("V", R, 3)
        self.assertTrue(V.is_independent([1, 0, 0], [0, 1, 0]))
        self.assertTrue(V.is_independent([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        self.assertFalse(V.is_independent([1, 2, 3], [2, 4, 6]))
        self.assertFalse(V.is_independent([0, 0, 0]))
        self.assertTrue(V.is_independent())  # Empty set is independent

    def test_is_independent_invalid_vector(self):
        """Test is_independent with invalid vector."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_independent([1, 2])  # Wrong dimension

    def test_is_basis(self):
        """Test is_basis method."""
        V = fn("V", R, 3)
        self.assertTrue(V.is_basis([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        self.assertFalse(V.is_basis([1, 0, 0], [0, 1, 0]))  # Not enough vectors
        self.assertFalse(V.is_basis([1, 2, 3], [2, 4, 6], [0, 0, 1]))  # Dependent
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        basis = V.basis
        self.assertTrue(V.is_basis(*basis))

    def test_is_basis_invalid_vector(self):
        """Test is_basis with invalid vector."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_basis([1, 2])  # Wrong dimension

    def test_change_of_basis(self):
        """Test change_of_basis method."""
        V = fn("V", R, 3)
        new_basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        P = V.change_of_basis(new_basis)
        self.assertEqual(P, M.eye(3))
        
        new_basis = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
        P = V.change_of_basis(new_basis)
        self.assertEqual(P.shape, (3, 3))
        self.assertTrue(is_invertible(P))

    def test_change_of_basis_invalid(self):
        """Test change_of_basis with invalid basis."""
        V = fn("V", R, 3)
        with self.assertRaises(ValueError):
            V.change_of_basis([[1, 0, 0], [2, 0, 0]])  # Not a basis


class TestVectorSpaceSpaceMethods(unittest.TestCase):
    """Test VectorSpace space-related methods."""

    def test_ambient_space(self):
        """Test ambient_space method."""
        V = fn("V", R, 3)
        ambient = V.ambient_space()
        self.assertEqual(ambient.dim, 3)
        self.assertEqual(ambient.field, R)
        
        U = fn("U", R, 3, constraints=["v0 + v1 == 0"])
        ambient = U.ambient_space()
        self.assertEqual(ambient.dim, 3)

    def test_sum(self):
        """Test sum method."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W = U.sum(V)
        self.assertEqual(W.dim, 3)
        self.assertTrue(W.is_subspace(U))  # W contains U, so U is subspace of W
        self.assertTrue(W.is_subspace(V))  # W contains V, so V is subspace of W
        
        # Sum with itself
        V = fn("V", R, 3)
        result = V.sum(V)
        self.assertEqual(result, V)

    def test_sum_different_ambient(self):
        """Test sum with different ambient space."""
        U = fn("U", R, 2)
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            U.sum(V)

    def test_intersection(self):
        """Test intersection method."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W = U.intersection(V)
        self.assertEqual(W.dim, 1)
        self.assertTrue(U.is_subspace(W))  # W is contained in U
        self.assertTrue(V.is_subspace(W))  # W is contained in V
        
        # Intersection with itself
        V = fn("V", R, 3)
        result = V.intersection(V)
        self.assertEqual(result, V)

    def test_intersection_different_ambient(self):
        """Test intersection with different ambient space."""
        U = fn("U", R, 2)
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            U.intersection(V)

    def test_span_vectors(self):
        """Test span method with vectors."""
        V = fn("V", R, 3)
        W = V.span("W", [1, 2, 3], [4, 5, 6])
        self.assertIsInstance(W, VectorSpace)
        self.assertTrue(V.is_subspace(W))  # W is contained in V
        
        # Span of empty set
        W = V.span("W")
        self.assertEqual(W.dim, 0)

    def test_span_basis(self):
        """Test span method with basis parameter."""
        V = fn("V", R, 3)
        W = V.span("W", basis=[[1, 0, 0], [0, 1, 0]])
        self.assertEqual(W.dim, 2)
        self.assertEqual(len(W.basis), 2)

    def test_span_invalid_vector(self):
        """Test span with invalid vector."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.span("W", [1, 2])  # Wrong dimension

    def test_is_subspace(self):
        """Test is_subspace method."""
        V = fn("V", R, 3)
        U = fn("U", R, 3, constraints=["v0 + v1 == 0"])
        self.assertTrue(V.is_subspace(U))  # U is a subspace of V
        self.assertFalse(U.is_subspace(V))  # V is not a subspace of U
        self.assertTrue(V.is_subspace(V))  # V is a subspace of itself
        
        W = fn("W", R, 3, constraints=["v0 == 0", "v1 == 0"])
        self.assertTrue(U.is_subspace(W))  # W is a subspace of U
        self.assertTrue(V.is_subspace(W))  # W is a subspace of V

    def test_is_subspace_different_ambient(self):
        """Test is_subspace with different ambient space."""
        U = fn("U", R, 2)
        V = fn("V", R, 3)
        self.assertFalse(U.is_subspace(V))

    def test_coset(self):
        """Test coset method."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        v = [1, 2, 3]
        coset = V.coset(v)
        self.assertIsInstance(coset, AffineSpace)
        self.assertEqual(coset.vectorspace, V)
        self.assertEqual(M(coset.representative), M(v))



class TestVectorSpaceInfo(unittest.TestCase):
    """Test VectorSpace info method."""

    def test_info(self):
        """Test info method."""
        V = fn("V", R, 3)
        info_str = V.info()
        self.assertIn("V", info_str)
        self.assertIn("Field", info_str)
        self.assertIn("Dimension", info_str)
        self.assertIn("Basis", info_str)
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        info_str = V.info()
        self.assertIn("V", info_str)
        self.assertIn(str(V.dim), info_str)


class TestVectorSpaceWithFactoryFunctions(unittest.TestCase):
    """Test VectorSpace with different factory functions."""

    def test_fn_factory(self):
        """Test fn factory function."""
        V = fn("V", R, 3)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.dim, 3)
        self.assertEqual(V.field, R)
        
        V = fn("V", C, 2)
        self.assertEqual(V.field, C)
        
        V = fn("V", R, 1)
        self.assertEqual(V.dim, 1)

    def test_matrix_space_factory(self):
        """Test matrix_space factory function."""
        M_space = matrix_space("M", R, (2, 2))
        self.assertIsInstance(M_space, VectorSpace)
        self.assertEqual(M_space.dim, 4)
        
        mat = [[1, 2], [3, 4]]
        self.assertIn(mat, M_space)
        
        M_space = matrix_space("M", R, (3, 2))
        self.assertEqual(M_space.dim, 6)

    def test_poly_space_factory(self):
        """Test poly_space factory function."""
        P = poly_space("P", R, 3)
        self.assertIsInstance(P, VectorSpace)
        self.assertEqual(P.dim, 4)
        
        poly = sp.Poly([1, 2, 3, 4], sp.Symbol('x'))
        self.assertIn(poly.as_expr(), P)

    def test_hom_factory(self):
        """Test hom factory function."""
        V1 = fn("V1", R, 2)
        V2 = fn("V2", R, 3)
        H = hom(V1, V2)
        self.assertIsInstance(H, VectorSpace)
        self.assertEqual(H.dim, 6)  # 3 * 2
        
        V1 = fn("V1", R, 3)
        V2 = fn("V2", R, 2)
        H = hom(V1, V2)
        self.assertEqual(H.dim, 6)  # 2 * 3

    def test_hom_different_fields(self):
        """Test hom with different fields."""
        V1 = fn("V1", R, 2)
        V2 = fn("V2", C, 3)
        with self.assertRaises(TypeError):
            hom(V1, V2)


class TestVectorSpaceEdgeCases(unittest.TestCase):
    """Test VectorSpace edge cases and comprehensive scenarios."""

    def test_add_edge_cases(self):
        """Test add property with edge cases."""
        V = fn("V", R, 3)
        # Zero vector
        zero = V.additive_id
        v = [1, 2, 3]
        self.assertEqual(M(V.add(zero, v)), M(v))
        self.assertEqual(M(V.add(v, zero)), M(v))
        self.assertEqual(M(V.add(zero, zero)), M([0, 0, 0]))
        
        # Negative vectors
        v1 = [1, 2, 3]
        v2 = [-1, -2, -3]
        self.assertEqual(M(V.add(v1, v2)), M([0, 0, 0]))
        
        # Complex field
        V = fn("V", C, 2)
        v1 = [1+2j, 3+4j]
        v2 = [5+6j, 7+8j]
        result = V.add(v1, v2)
        self.assertEqual(M(result), M([6+8j, 10+12j]))

    def test_mul_edge_cases(self):
        """Test mul property with edge cases."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        
        # Zero scalar
        self.assertEqual(M(V.mul(0, v)), M([0, 0, 0]))
        
        # One scalar
        self.assertEqual(M(V.mul(1, v)), M(v))
        
        # Negative scalar
        self.assertEqual(M(V.mul(-1, v)), M([-1, -2, -3]))
        
        # Fractional scalar
        result = M(V.mul(0.5, v))
        expected = M([0.5, 1, 1.5])
        self.assertEqual(result.shape, expected.shape)
        for i in range(result.shape[0]):
            self.assertAlmostEqual(float(result[i]), float(expected[i]))
        
        # Zero vector
        zero = V.additive_id
        self.assertEqual(M(V.mul(5, zero)), M([0, 0, 0]))

    def test_additive_inv_edge_cases(self):
        """Test additive_inv property with edge cases."""
        V = fn("V", R, 3)
        
        # Zero vector
        zero = V.additive_id
        self.assertEqual(M(V.additive_inv(zero)), M([0, 0, 0]))
        
        # Regular vector
        v = [1, 2, 3]
        self.assertEqual(M(V.additive_inv(v)), M([-1, -2, -3]))
        
        # Double negation
        self.assertEqual(M(V.additive_inv(V.additive_inv(v))), M(v))

    def test_contains_edge_cases(self):
        """Test __contains__ with edge cases."""
        V = fn("V", R, 3)
        
        # Zero vector
        self.assertIn([0, 0, 0], V)
        
        # Large numbers
        self.assertIn([1000, -2000, 3000], V)
        
        # Decimal numbers
        self.assertIn([1.5, 2.7, 3.9], V)
        
        # With constraints
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        self.assertIn([1, -1, 5], V)
        self.assertNotIn([1, 1, 5], V)
        self.assertIn([0, 0, 0], V)
        
        # Invalid types
        self.assertNotIn("not a vector", V)
        self.assertNotIn(123, V)
        self.assertNotIn([1, 2], V)  # Wrong dimension
        self.assertNotIn([1, 2, 3, 4], V)  # Wrong dimension

    def test_eq_edge_cases(self):
        """Test __eq__ with edge cases."""
        # Same space
        V1 = fn("V1", R, 3)
        V2 = fn("V2", R, 3)
        self.assertEqual(V1, V2)
        
        # Zero dimension
        V1 = fn("V1", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        V2 = fn("V2", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        self.assertEqual(V1, V2)
        
        # Different dimensions
        V1 = fn("V1", R, 2)
        V2 = fn("V2", R, 3)
        self.assertNotEqual(V1, V2)
        
        # Different fields
        V1 = fn("V1", R, 3)
        V2 = fn("V2", C, 3)
        self.assertNotEqual(V1, V2)
        
        # Self equality
        V = fn("V", R, 3)
        self.assertEqual(V, V)

    def test_vector_edge_cases(self):
        """Test vector method with edge cases."""
        # Zero dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        v = V.vector()
        self.assertEqual(M(v), M([0, 0, 0]))
        v = V.vector(arbitrary=True)
        self.assertEqual(M(v), M([0, 0, 0]))
        
        # One dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        v = V.vector()
        self.assertIn(v, V)
        v = V.vector(arbitrary=True)
        self.assertIn(v, V)
        self.assertEqual(len(v), 3)
        
        # Different std values
        V = fn("V", R, 3)
        v1 = V.vector(std=0.1)
        v2 = V.vector(std=100)
        self.assertIn(v1, V)
        self.assertIn(v2, V)

    def test_to_coordinate_edge_cases(self):
        """Test to_coordinate with edge cases."""
        # Zero dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        coord = V.to_coordinate([0, 0, 0])
        self.assertEqual(coord.shape[0], 0)  # Zero rows
        
        # One dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        v = [0, 0, 1]
        coord = V.to_coordinate(v)
        self.assertEqual(coord.shape, (1, 1))
        
        # Custom basis with zero dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        # Empty basis is valid for zero dimension
        coord = V.to_coordinate([0, 0, 0], basis=[])
        self.assertEqual(coord.shape[0], 0)

    def test_from_coordinate_edge_cases(self):
        """Test from_coordinate with edge cases."""
        # Zero dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        coord = M.zeros(0, 1)
        v = V.from_coordinate(coord)
        self.assertEqual(M(v), M([0, 0, 0]))
        
        # One dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        coord = M([1])
        v = V.from_coordinate(coord)
        self.assertIn(v, V)

    def test_is_independent_edge_cases(self):
        """Test is_independent with edge cases."""
        V = fn("V", R, 3)
        
        # Empty set
        self.assertTrue(V.is_independent())
        
        # Single zero vector
        self.assertFalse(V.is_independent([0, 0, 0]))
        
        # Single non-zero vector
        self.assertTrue(V.is_independent([1, 0, 0]))
        
        # Multiple zero vectors
        self.assertFalse(V.is_independent([0, 0, 0], [0, 0, 0]))
        
        # Zero dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        self.assertTrue(V.is_independent())  # Empty set is independent

    def test_is_basis_edge_cases(self):
        """Test is_basis with edge cases."""
        V = fn("V", R, 3)
        
        # Empty set for zero dimension
        V_zero = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        self.assertTrue(V_zero.is_basis())
        
        # Too few vectors
        self.assertFalse(V.is_basis([1, 0, 0]))
        self.assertFalse(V.is_basis([1, 0, 0], [0, 1, 0]))
        
        # Too many vectors
        self.assertFalse(V.is_basis([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]))
        
        # One dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        self.assertTrue(V.is_basis([0, 0, 1]))

    def test_change_of_basis_edge_cases(self):
        """Test change_of_basis with edge cases."""
        # Zero dimension - empty basis is valid
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        # For zero dimension, empty basis is valid, but change_of_basis might not be applicable
        # Let's test with a non-zero dimension instead
        
        # One dimension
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        new_basis = [[0, 0, 2]]
        P = V.change_of_basis(new_basis)
        self.assertEqual(P.shape, (1, 1))
        # Change of basis matrix transforms from old to new basis
        # [0, 0, 2] = 2 * [0, 0, 1], so coordinate in old basis is [2]
        # P transforms from old to new, so P = [1/2]
        self.assertEqual(P, M([[sp.Rational(1, 2)]]))

    def test_span_edge_cases(self):
        """Test span with edge cases."""
        V = fn("V", R, 3)
        
        # Span of zero vector
        W = V.span("W", [0, 0, 0])
        self.assertEqual(W.dim, 0)
        
        # Span of single vector
        W = V.span("W", [1, 0, 0])
        self.assertEqual(W.dim, 1)
        self.assertTrue(V.is_subspace(W))
        
        # Span with basis parameter (empty)
        W = V.span("W", basis=[])
        self.assertEqual(W.dim, 0)
        
        # Span with basis parameter (single vector)
        W = V.span("W", basis=[[1, 0, 0]])
        self.assertEqual(W.dim, 1)

    def test_sum_edge_cases(self):
        """Test sum with edge cases."""
        V = fn("V", R, 3)
        
        # Sum with zero subspace
        V_zero = fn("V_zero", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        W = V.sum(V_zero)
        self.assertEqual(W, V)
        
        # Sum of zero subspaces
        V_zero1 = fn("V_zero1", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        V_zero2 = fn("V_zero2", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        W = V_zero1.sum(V_zero2)
        self.assertEqual(W.dim, 0)

    def test_intersection_edge_cases(self):
        """Test intersection with edge cases."""
        V = fn("V", R, 3)
        
        # Intersection with zero subspace
        V_zero = fn("V_zero", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        W = V.intersection(V_zero)
        self.assertEqual(W.dim, 0)
        
        # Intersection of zero subspaces
        V_zero1 = fn("V_zero1", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        V_zero2 = fn("V_zero2", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        W = V_zero1.intersection(V_zero2)
        self.assertEqual(W.dim, 0)
        
        # Intersection of disjoint subspaces (only zero vector in common)
        U = fn("U", R, 3, constraints=["v0 == 0"])
        V = fn("V", R, 3, constraints=["v1 == 0"])
        W = U.intersection(V)
        self.assertEqual(W.dim, 1)  # Only v2 is free

    def test_ambient_space_edge_cases(self):
        """Test ambient_space with edge cases."""
        # Zero dimension subspace
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        ambient = V.ambient_space()
        self.assertEqual(ambient.dim, 3)
        
        # Full space
        V = fn("V", R, 3)
        ambient = V.ambient_space()
        self.assertEqual(ambient, V)

    def test_coset_edge_cases(self):
        """Test coset with edge cases."""
        # AffineSpace is already imported via from ablina import *
        V = fn("V", R, 3)
        
        # Coset with zero vector
        coset = V.coset([0, 0, 0])
        self.assertIsInstance(coset, AffineSpace)
        self.assertEqual(coset.vectorspace, V)
        
        # Coset with zero dimension space
        V_zero = fn("V_zero", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        coset = V_zero.coset([0, 0, 0])
        self.assertEqual(coset.dim, 0)

    def test_matrix_space_edge_cases(self):
        """Test matrix_space factory with edge cases."""
        # 1x1 matrix
        M_space = matrix_space("M", R, (1, 1))
        self.assertEqual(M_space.dim, 1)
        self.assertIn([[5]], M_space)
        
        # Large matrix
        M_space = matrix_space("M", R, (10, 10))
        self.assertEqual(M_space.dim, 100)
        
        # Non-square matrix
        M_space = matrix_space("M", R, (2, 3))
        self.assertEqual(M_space.dim, 6)
        self.assertIn([[1, 2, 3], [4, 5, 6]], M_space)

    def test_poly_space_edge_cases(self):
        """Test poly_space factory with edge cases."""
        # Degree 0 (constants)
        P = poly_space("P", R, 0)
        self.assertEqual(P.dim, 1)
        self.assertIn(5, P)
        self.assertIn(sp.Symbol('x') * 0 + 3, P)
        
        # Degree 1
        P = poly_space("P", R, 1)
        self.assertEqual(P.dim, 2)
        self.assertIn(sp.Symbol('x') + 1, P)
        
        # With constraints
        P = poly_space("P", R, 2, constraints=["v0 == 0"])
        self.assertEqual(P.dim, 2)  # No constant term

    def test_hom_edge_cases(self):
        """Test hom factory with edge cases."""
        # Zero dimension domain
        V1 = fn("V1", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        V2 = fn("V2", R, 2)
        H = hom(V1, V2)
        self.assertEqual(H.dim, 0)
        
        # Zero dimension codomain
        V1 = fn("V1", R, 2)
        V2 = fn("V2", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        H = hom(V1, V2)
        self.assertEqual(H.dim, 0)
        
        # Both zero dimension
        V1 = fn("V1", R, 2, constraints=["v0 == 0", "v1 == 0"])
        V2 = fn("V2", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        H = hom(V1, V2)
        self.assertEqual(H.dim, 0)

    # Additional comprehensive edge case tests

    def test_eq_with_none(self):
        """Test __eq__ with None."""
        V = fn("V", R, 3)
        self.assertNotEqual(V, None)
        self.assertFalse(V == None)
        self.assertFalse(None == V)

    def test_eq_with_non_vectorspace(self):
        """Test __eq__ with non-VectorSpace objects."""
        V = fn("V", R, 3)
        self.assertNotEqual(V, "not a vector space")
        self.assertNotEqual(V, 123)
        self.assertNotEqual(V, [1, 2, 3])

    def test_contains_with_none(self):
        """Test __contains__ with None."""
        V = fn("V", R, 3)
        self.assertNotIn(None, V)

    def test_add_with_none(self):
        """Test __add__ with None."""
        V = fn("V", R, 3)
        # Should create a coset, but None is not in the ambient space
        # This should raise TypeError when creating coset
        with self.assertRaises(TypeError):
            V + None

    def test_add_with_invalid_type(self):
        """Test __add__ with invalid types."""
        V = fn("V", R, 3)
        # Invalid types should raise TypeError when creating coset
        with self.assertRaises(TypeError):
            V + "not a vector"
        with self.assertRaises(TypeError):
            V + 123

    def test_sub_with_none(self):
        """Test __sub__ with None."""
        V = fn("V", R, 3)
        # None is not in ambient space, should raise TypeError
        with self.assertRaises(TypeError):
            V - None

    def test_sub_with_vector_not_in_ambient(self):
        """Test __sub__ with vector not in ambient space."""
        V = fn("V", R, 3)
        # Vector with wrong dimension
        with self.assertRaises(TypeError):
            V - [1, 2]  # Wrong dimension
        with self.assertRaises(TypeError):
            V - [1, 2, 3, 4]  # Wrong dimension

    def test_radd_with_invalid_type(self):
        """Test __radd__ with invalid types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            "not a vector" + V
        with self.assertRaises(TypeError):
            123 + V

    def test_rsub_with_invalid_type(self):
        """Test __rsub__ with invalid types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            "not a vector" - V
        with self.assertRaises(TypeError):
            123 - V

    def test_and_with_none(self):
        """Test __and__ with None."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V & None

    def test_and_with_non_vectorspace(self):
        """Test __and__ with non-VectorSpace."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V & "not a vector space"

    def test_vector_with_zero_std(self):
        """Test vector method with std=0."""
        V = fn("V", R, 3)
        # std=0 should still work, just generates vectors with very small weights
        v = V.vector(std=0)
        self.assertIn(v, V)

    def test_vector_with_negative_std(self):
        """Test vector method with negative std."""
        V = fn("V", R, 3)
        # Negative std should still work (gauss can handle it)
        v = V.vector(std=-1)
        self.assertIn(v, V)

    def test_to_coordinate_with_none(self):
        """Test to_coordinate with None."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.to_coordinate(None)

    def test_to_coordinate_vector_not_in_space(self):
        """Test to_coordinate with vector not in space."""
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        # Vector that doesn't satisfy constraint
        with self.assertRaises(TypeError):
            V.to_coordinate([1, 1, 0])  # Doesn't satisfy v0 + v1 == 0

    def test_to_coordinate_with_invalid_basis_type(self):
        """Test to_coordinate with invalid basis type."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        with self.assertRaises(TypeError):
            V.to_coordinate(v, basis="not a list")
        with self.assertRaises(TypeError):
            V.to_coordinate(v, basis=123)

    def test_from_coordinate_with_none(self):
        """Test from_coordinate with None."""
        V = fn("V", R, 3)
        with self.assertRaises((TypeError, ValueError)):
            V.from_coordinate(None)

    def test_from_coordinate_with_invalid_shape_list(self):
        """Test from_coordinate with invalid shape (list)."""
        V = fn("V", R, 3)
        # Wrong dimension
        with self.assertRaises(ValueError):
            V.from_coordinate([1, 2])  # Should be length 3
        with self.assertRaises(ValueError):
            V.from_coordinate([1, 2, 3, 4])  # Should be length 3

    def test_from_coordinate_with_invalid_field_elements(self):
        """Test from_coordinate with invalid field elements."""
        V = fn("V", R, 2)
        # Complex number in real field
        with self.assertRaises(ValueError):
            V.from_coordinate([1+2j, 3])

    def test_from_coordinate_with_invalid_basis_type(self):
        """Test from_coordinate with invalid basis type."""
        V = fn("V", R, 3)
        coord = M([1, 2, 3])
        with self.assertRaises(TypeError):
            V.from_coordinate(coord, basis="not a list")
        with self.assertRaises(TypeError):
            V.from_coordinate(coord, basis=123)

    def test_is_independent_with_none(self):
        """Test is_independent with None in vectors."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_independent(None, [1, 0, 0])
        with self.assertRaises(TypeError):
            V.is_independent([1, 0, 0], None)

    def test_is_independent_with_invalid_vector(self):
        """Test is_independent with invalid vector types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_independent("not a vector")
        with self.assertRaises(TypeError):
            V.is_independent(123)

    def test_is_basis_with_none(self):
        """Test is_basis with None in vectors."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_basis(None, [1, 0, 0], [0, 1, 0])
        with self.assertRaises(TypeError):
            V.is_basis([1, 0, 0], None, [0, 1, 0])

    def test_is_basis_with_invalid_vector(self):
        """Test is_basis with invalid vector types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.is_basis("not a vector")
        with self.assertRaises(TypeError):
            V.is_basis(123)

    def test_change_of_basis_with_none(self):
        """Test change_of_basis with None."""
        V = fn("V", R, 3)
        with self.assertRaises((TypeError, ValueError)):
            V.change_of_basis(None)

    def test_change_of_basis_with_invalid_type(self):
        """Test change_of_basis with invalid type."""
        V = fn("V", R, 3)
        with self.assertRaises((TypeError, ValueError)):
            V.change_of_basis("not a list")
        with self.assertRaises((TypeError, ValueError)):
            V.change_of_basis(123)

    def test_span_with_none_in_vectors(self):
        """Test span with None in vectors."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.span("W", None, [1, 0, 0])
        with self.assertRaises(TypeError):
            V.span("W", [1, 0, 0], None)

    def test_span_with_invalid_vector_type(self):
        """Test span with invalid vector types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.span("W", "not a vector")
        with self.assertRaises(TypeError):
            V.span("W", 123)

    def test_span_with_invalid_basis_type(self):
        """Test span with invalid basis type."""
        V = fn("V", R, 3)
        with self.assertRaises((TypeError, ValueError)):
            V.span("W", basis="not a list")
        with self.assertRaises((TypeError, ValueError)):
            V.span("W", basis=123)

    def test_sum_with_none(self):
        """Test sum with None."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.sum(None)

    def test_sum_with_non_vectorspace(self):
        """Test sum with non-VectorSpace."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.sum("not a vector space")

    def test_intersection_with_none(self):
        """Test intersection with None."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.intersection(None)

    def test_intersection_with_non_vectorspace(self):
        """Test intersection with non-VectorSpace."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.intersection("not a vector space")

    def test_is_subspace_with_none(self):
        """Test is_subspace with None."""
        V = fn("V", R, 3)
        self.assertFalse(V.is_subspace(None))

    def test_is_subspace_with_non_vectorspace(self):
        """Test is_subspace with non-VectorSpace."""
        V = fn("V", R, 3)
        self.assertFalse(V.is_subspace("not a vector space"))
        self.assertFalse(V.is_subspace(123))

    def test_coset_with_none(self):
        """Test coset with None."""
        V = fn("V", R, 3)
        # None is not in ambient space, should raise TypeError
        with self.assertRaises(TypeError):
            V.coset(None)

    def test_coset_with_invalid_type(self):
        """Test coset with invalid types."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            V.coset("not a vector")
        with self.assertRaises(TypeError):
            V.coset(123)

    def test_ambient_space_with_different_factory_functions(self):
        """Test ambient_space with different factory functions."""
        # fn factory
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        ambient = V.ambient_space()
        self.assertEqual(ambient.dim, 3)
        self.assertEqual(ambient.field, R)
        
        # matrix_space factory
        M_space = matrix_space("M", R, (2, 2), constraints=["v0 == 0"])
        ambient = M_space.ambient_space()
        self.assertEqual(ambient.dim, 4)
        
        # poly_space factory
        P = poly_space("P", R, 2, constraints=["v0 == 0"])
        ambient = P.ambient_space()
        self.assertEqual(ambient.dim, 3)

    def test_additive_id_with_different_factory_functions(self):
        """Test additive_id with different factory functions."""
        # fn factory
        V = fn("V", R, 3)
        zero = V.additive_id
        self.assertEqual(M(zero), M([0, 0, 0]))
        
        # matrix_space factory
        M_space = matrix_space("M", R, (2, 2))
        zero = M_space.additive_id
        self.assertEqual(M(zero), M.zeros(2, 2))
        
        # poly_space factory
        P = poly_space("P", R, 2)
        zero = P.additive_id
        # Should be zero polynomial
        self.assertEqual(zero, 0)

    def test_basis_with_different_factory_functions(self):
        """Test basis property with different factory functions."""
        # fn factory
        V = fn("V", R, 3)
        basis = V.basis
        self.assertEqual(len(basis), 3)
        
        # matrix_space factory
        M_space = matrix_space("M", R, (2, 2))
        basis = M_space.basis
        self.assertEqual(len(basis), 4)
        
        # poly_space factory
        P = poly_space("P", R, 2)
        basis = P.basis
        self.assertEqual(len(basis), 3)

    def test_info_with_zero_dimension(self):
        """Test info method with zero dimension."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        info_str = V.info()
        self.assertIn("V", info_str)
        self.assertIn("Dimension", info_str)
        self.assertIn("0", info_str)

    def test_info_with_one_dimension(self):
        """Test info method with one dimension."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        info_str = V.info()
        self.assertIn("V", info_str)
        self.assertIn("Dimension", info_str)
        self.assertIn("1", info_str)

    def test_repr_with_different_spaces(self):
        """Test __repr__ with different spaces."""
        V = fn("V", R, 3)
        repr_str = repr(V)
        self.assertIn("V", repr_str)
        self.assertIn("basis", repr_str)
        
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        repr_str = repr(V)
        self.assertIn("V", repr_str)

    def test_str_with_special_characters(self):
        """Test __str__ with special characters in name."""
        V = fn("V with spaces", R, 3)
        self.assertEqual(str(V), "V with spaces")
        
        V = fn("V-123_test", R, 3)
        self.assertEqual(str(V), "V-123_test")

    def test_to_coordinate_roundtrip_with_constraints(self):
        """Test to_coordinate/from_coordinate roundtrip with constraints."""
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        v = [1, -1, 5]
        coord = V.to_coordinate(v)
        v2 = V.from_coordinate(coord)
        self.assertEqual(M(v), M(v2))

    def test_change_of_basis_roundtrip(self):
        """Test change_of_basis preserves vector representation."""
        V = fn("V", R, 3)
        new_basis = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
        P = V.change_of_basis(new_basis)
        
        # Test that P transforms coordinates correctly
        v = [1, 2, 3]
        old_coord = V.to_coordinate(v)
        new_coord = V.to_coordinate(v, basis=new_basis)
        # P transforms from old to new: new_coord = P @ old_coord
        expected_new = P @ old_coord
        self.assertEqual(M(new_coord), M(expected_new))

    def test_span_with_zero_vectors(self):
        """Test span with multiple zero vectors."""
        V = fn("V", R, 3)
        # Span of multiple zero vectors should be zero dimension
        W = V.span("W", [0, 0, 0], [0, 0, 0])
        self.assertEqual(W.dim, 0)

    def test_span_with_duplicate_vectors(self):
        """Test span with duplicate vectors."""
        V = fn("V", R, 3)
        # Duplicate vectors should be handled (span removes duplicates)
        W = V.span("W", [1, 0, 0], [1, 0, 0])
        self.assertEqual(W.dim, 1)

    def test_sum_commutative(self):
        """Test that sum is commutative."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W1 = U.sum(V)
        W2 = V.sum(U)
        self.assertEqual(W1, W2)

    def test_intersection_commutative(self):
        """Test that intersection is commutative."""
        U = fn("U", R, 3, constraints=["v0 == v1"])
        V = fn("V", R, 3, constraints=["v1 == v2"])
        W1 = U.intersection(V)
        W2 = V.intersection(U)
        self.assertEqual(W1, W2)

    def test_additive_id_is_identity(self):
        """Test that additive_id is actually the identity."""
        V = fn("V", R, 3)
        zero = V.additive_id
        v = [1, 2, 3]
        # v + zero = v
        self.assertEqual(M(V.add(v, zero)), M(v))
        self.assertEqual(M(V.add(zero, v)), M(v))
        # zero + zero = zero
        self.assertEqual(M(V.add(zero, zero)), M(zero))

    def test_additive_inv_cancels(self):
        """Test that additive_inv cancels with addition."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        neg_v = V.additive_inv(v)
        # v + (-v) = 0
        self.assertEqual(M(V.add(v, neg_v)), M(V.additive_id))
        self.assertEqual(M(V.add(neg_v, v)), M(V.additive_id))

    def test_mul_distributive(self):
        """Test that scalar multiplication is distributive."""
        V = fn("V", R, 3)
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        scalar = 2
        # scalar * (v1 + v2) = scalar*v1 + scalar*v2
        left = V.mul(scalar, V.add(v1, v2))
        right = V.add(V.mul(scalar, v1), V.mul(scalar, v2))
        self.assertEqual(M(left), M(right))

    def test_mul_associative(self):
        """Test that scalar multiplication is associative."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        a, b = 2, 3
        # (a * b) * v = a * (b * v)
        left = V.mul(a * b, v)
        right = V.mul(a, V.mul(b, v))
        self.assertEqual(M(left), M(right))

    def test_init_with_constraints_and_basis_valid(self):
        """Test __init__ with both constraints and valid basis."""
        # Basis vectors that satisfy constraints
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"], 
               basis=[[1, -1, 0], [0, 0, 1]])
        self.assertEqual(V.dim, 2)
        # All basis vectors should be in the space
        for vec in V.basis:
            self.assertIn(vec, V)

    def test_init_with_constraints_and_basis_invalid_not_in_space(self):
        """Test __init__ with constraints and basis not in constrained space."""
        # Basis vector that doesn't satisfy constraints
        with self.assertRaises(TypeError):
            fn("V", R, 3, constraints=["v0 + v1 == 0"], 
               basis=[[1, 1, 0], [0, 0, 1]])  # [1, 1, 0] doesn't satisfy constraint

    def test_init_with_constraints_and_basis_invalid_dependent(self):
        """Test __init__ with constraints and linearly dependent basis."""
        # Basis vectors that satisfy constraints but are dependent
        with self.assertRaises(ValueError):
            fn("V", R, 3, constraints=["v0 + v1 == 0"], 
               basis=[[1, -1, 0], [2, -2, 0]])  # Linearly dependent

    def test_init_basis_after_constraints_creates_subspace(self):
        """Test that basis after constraints creates correct subspace."""
        # When both constraints and basis are provided, the final space
        # should be the span of the basis vectors (which must satisfy constraints)
        V1 = fn("V1", R, 3, constraints=["v0 + v1 == 0"])
        V2 = fn("V2", R, 3, constraints=["v0 + v1 == 0"], 
                basis=[[1, -1, 0], [0, 0, 1]])
        # V2 should be a subspace of V1
        self.assertTrue(V1.is_subspace(V2))
        # And they should be equal since the basis spans the constrained space
        self.assertEqual(V1, V2)


class TestAffineSpaceInit(unittest.TestCase):
    """Test AffineSpace.__init__ method."""

    def test_init_basic(self):
        """Test AffineSpace initialization with basic parameters."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.vectorspace, V)
        self.assertEqual(M(aspace.representative), M(v))
        self.assertEqual(aspace.name, f"{V} + {v}")

    def test_init_zero_vector(self):
        """Test AffineSpace initialization with zero vector."""
        V = fn("V", R, 3)
        zero = [0, 0, 0]
        aspace = AffineSpace(V, zero)
        self.assertEqual(aspace.vectorspace, V)
        self.assertEqual(M(aspace.representative), M(zero))

    def test_init_with_subspace(self):
        """Test AffineSpace initialization with subspace."""
        V = fn("V", R, 3)
        U = fn("U", R, 3, constraints=["v0 + v1 == 0"])
        v = [1, -1, 5]
        aspace = AffineSpace(U, v)
        self.assertEqual(aspace.vectorspace, U)
        self.assertEqual(M(aspace.representative), M(v))

    def test_init_invalid_vectorspace_type(self):
        """Test AffineSpace initialization with invalid vectorspace type."""
        v = [1, 2, 3]
        with self.assertRaises(TypeError):
            AffineSpace("not a vector space", v)

    def test_init_representative_not_in_ambient(self):
        """Test AffineSpace initialization with representative not in ambient space."""
        V = fn("V", R, 3)
        # Wrong dimension
        with self.assertRaises(TypeError):
            AffineSpace(V, [1, 2])  # Wrong dimension
        with self.assertRaises(TypeError):
            AffineSpace(V, [1, 2, 3, 4])  # Wrong dimension

    def test_init_complex_field(self):
        """Test AffineSpace initialization with complex field."""
        V = fn("V", C, 2)
        v = [1+2j, 3+4j]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.vectorspace, V)
        self.assertEqual(M(aspace.representative), M(v))


class TestAffineSpaceProperties(unittest.TestCase):
    """Test AffineSpace properties."""

    def test_vectorspace_property(self):
        """Test vectorspace property."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.vectorspace, V)

    def test_representative_property(self):
        """Test representative property."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(M(aspace.representative), M(v))

    def test_dim_property(self):
        """Test dim property."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.dim, 3)
        
        U = fn("U", R, 3, constraints=["v0 + v1 == 0"])
        aspace = AffineSpace(U, [1, -1, 5])
        self.assertEqual(aspace.dim, 2)

    def test_set_property(self):
        """Test set property."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertIsNotNone(aspace.set)
        self.assertEqual(aspace.set.name, aspace.name)


class TestAffineSpaceSpecialMethods(unittest.TestCase):
    """Test AffineSpace special methods."""

    def test_repr(self):
        """Test __repr__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        repr_str = repr(aspace)
        self.assertIn("AffineSpace", repr_str)
        self.assertIn("vectorspace", repr_str)
        self.assertIn("representative", repr_str)

    def test_str(self):
        """Test __str__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(str(aspace), f"{V} + {v}")

    def test_eq_same_affine_space(self):
        """Test __eq__ with same affine space."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace1 = AffineSpace(V, v)
        aspace2 = AffineSpace(V, v)
        # Two affine spaces with same vector space and representative are equal
        self.assertEqual(aspace1, aspace2)

    def test_eq_different_representatives(self):
        """Test __eq__ with different representatives in same coset."""
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        # Two representatives that differ by a vector in V should be equal
        v1 = [1, -1, 5]
        v2 = [2, -2, 5]  # v2 - v1 = [1, -1, 0] is in V
        aspace1 = AffineSpace(V, v1)
        aspace2 = AffineSpace(V, v2)
        self.assertEqual(aspace1, aspace2)

    def test_eq_different_affine_spaces(self):
        """Test __eq__ with different affine spaces."""
        # Use a proper subspace, not the full space
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        v1 = [1, -1, 5]
        v2 = [1, 0, 5]  # Different coset (v2 - v1 = [0, 1, 0], and 0 + 1 != 0, so not in V)
        aspace1 = AffineSpace(V, v1)
        aspace2 = AffineSpace(V, v2)
        # Different representatives in different cosets are different
        self.assertNotEqual(aspace1, aspace2)

    def test_eq_with_non_affinespace(self):
        """Test __eq__ with non-AffineSpace objects."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertNotEqual(aspace, "not an affine space")
        self.assertNotEqual(aspace, 123)
        self.assertNotEqual(aspace, None)

    def test_contains(self):
        """Test __contains__ method."""
        # Use a proper subspace, not the full space
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        v = [1, -1, 5]
        aspace = AffineSpace(V, v)
        
        # Points in the affine space
        point1 = [2, -2, 5]  # v + [1, -1, 0] where [1, -1, 0] is in V
        self.assertIn(point1, aspace)
        
        # Representative itself
        self.assertIn(v, aspace)
        
        # Points not in the affine space
        point2 = [2, -1, 5]  # Doesn't satisfy v0 + v1 == 0
        self.assertNotIn(point2, aspace)
        
        # Invalid types
        self.assertNotIn("not a point", aspace)
        self.assertNotIn([1, 2], aspace)  # Wrong dimension

    def test_contains_with_subspace(self):
        """Test __contains__ with subspace."""
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        v = [1, -1, 5]
        aspace = AffineSpace(V, v)
        
        # Points in the affine space
        point1 = [2, -2, 5]  # v + [1, -1, 0] where [1, -1, 0] is in V
        self.assertIn(point1, aspace)
        
        # Points not in the affine space
        point2 = [2, -1, 5]  # Doesn't satisfy v0 + v1 == 0
        self.assertNotIn(point2, aspace)

    def test_pos(self):
        """Test __pos__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertIs(+aspace, aspace)

    def test_neg(self):
        """Test __neg__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        neg_aspace = -aspace
        self.assertIsInstance(neg_aspace, AffineSpace)
        self.assertEqual(neg_aspace.vectorspace, V)
        self.assertEqual(M(neg_aspace.representative), M([-1, -2, -3]))

    def test_add_with_affinespace(self):
        """Test __add__ with AffineSpace."""
        V = fn("V", R, 3)
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        aspace1 = AffineSpace(V, v1)
        aspace2 = AffineSpace(V, v2)
        result = aspace1 + aspace2
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        self.assertEqual(M(result.representative), M([5, 7, 9]))

    def test_add_with_vector(self):
        """Test __add__ with vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        u = [4, 5, 6]
        aspace = AffineSpace(V, v)
        result = aspace + u
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        self.assertEqual(M(result.representative), M([5, 7, 9]))

    def test_add_with_invalid_type(self):
        """Test __add__ with invalid type."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        with self.assertRaises(TypeError):
            aspace + "not a vector"
        with self.assertRaises(TypeError):
            aspace + [1, 2]  # Wrong dimension

    def test_radd(self):
        """Test __radd__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        u = [4, 5, 6]
        aspace = AffineSpace(V, v)
        result = u + aspace
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(M(result.representative), M([5, 7, 9]))

    def test_sub_with_affinespace(self):
        """Test __sub__ with AffineSpace."""
        V = fn("V", R, 3)
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        aspace1 = AffineSpace(V, v1)
        aspace2 = AffineSpace(V, v2)
        result = aspace1 - aspace2
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        # Should be v1 + (-v2) = v1 - v2
        self.assertEqual(M(result.representative), M([-3, -3, -3]))

    def test_sub_with_vector(self):
        """Test __sub__ with vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        u = [4, 5, 6]
        aspace = AffineSpace(V, v)
        result = aspace - u
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        self.assertEqual(M(result.representative), M([-3, -3, -3]))

    def test_sub_with_invalid_type(self):
        """Test __sub__ with invalid type."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        with self.assertRaises(TypeError):
            aspace - "not a vector"
        with self.assertRaises(TypeError):
            aspace - [1, 2]  # Wrong dimension

    def test_rsub(self):
        """Test __rsub__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        u = [4, 5, 6]
        aspace = AffineSpace(V, v)
        result = u - aspace
        self.assertIsInstance(result, AffineSpace)
        # Should be u + (-v) = u - v
        self.assertEqual(M(result.representative), M([3, 3, 3]))

    def test_mul(self):
        """Test __mul__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace * 2
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        self.assertEqual(M(result.representative), M([2, 4, 6]))
        
        result = aspace * 0
        self.assertEqual(M(result.representative), M([0, 0, 0]))
        
        result = aspace * -1
        self.assertEqual(M(result.representative), M([-1, -2, -3]))

    def test_mul_with_invalid_scalar(self):
        """Test __mul__ with invalid scalar."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        # Complex number in real field
        with self.assertRaises(TypeError):
            aspace * (1+2j)

    def test_rmul(self):
        """Test __rmul__ method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = 2 * aspace
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(M(result.representative), M([2, 4, 6]))


class TestAffineSpaceMethods(unittest.TestCase):
    """Test AffineSpace methods."""

    def test_info(self):
        """Test info method."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        info_str = aspace.info()
        self.assertIn(str(V), info_str)
        self.assertIn("Vector Space", info_str)
        self.assertIn("Representative", info_str)
        self.assertIn("Dimension", info_str)
        self.assertIn("Point", info_str)

    def test_point_random(self):
        """Test point method with random generation."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        point = aspace.point()
        self.assertIn(point, aspace)
        
        point = aspace.point(std=10)
        self.assertIn(point, aspace)

    def test_point_arbitrary(self):
        """Test point method with arbitrary=True."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        point = aspace.point(arbitrary=True)
        self.assertIn(point, aspace)
        # Point should be a valid point in the affine space
        # (For arbitrary=True, it should contain symbolic expressions)
        # Just verify it's in the space, which we already do above

    def test_point_zero_dimension(self):
        """Test point method with zero dimension."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        v = [0, 0, 0]
        aspace = AffineSpace(V, v)
        point = aspace.point()
        self.assertEqual(M(point), M([0, 0, 0]))
        
        point = aspace.point(arbitrary=True)
        self.assertEqual(M(point), M([0, 0, 0]))

    def test_sum(self):
        """Test sum method."""
        V = fn("V", R, 3)
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        aspace1 = AffineSpace(V, v1)
        aspace2 = AffineSpace(V, v2)
        result = aspace1.sum(aspace2)
        self.assertIsInstance(result, AffineSpace)
        self.assertEqual(result.vectorspace, V)
        self.assertEqual(M(result.representative), M([5, 7, 9]))

    def test_sum_different_vectorspaces(self):
        """Test sum with different vector spaces."""
        V1 = fn("V1", R, 3)
        V2 = fn("V2", R, 3, constraints=["v0 + v1 == 0"])
        aspace1 = AffineSpace(V1, [1, 2, 3])
        aspace2 = AffineSpace(V2, [1, -1, 5])
        with self.assertRaises(TypeError):
            aspace1.sum(aspace2)

    def test_sum_with_non_affinespace(self):
        """Test sum with non-AffineSpace."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            aspace.sum("not an affine space")

    def test_intersection(self):
        """Test intersection method."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(NotImplementedError):
            aspace.intersection(aspace)


class TestAffineSpaceEdgeCases(unittest.TestCase):
    """Test AffineSpace edge cases."""

    def test_eq_with_none(self):
        """Test __eq__ with None."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        self.assertNotEqual(aspace, None)

    def test_eq_with_self(self):
        """Test __eq__ with self."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        self.assertEqual(aspace, aspace)

    def test_contains_with_none(self):
        """Test __contains__ with None."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        self.assertNotIn(None, aspace)

    def test_contains_zero_dimension(self):
        """Test __contains__ with zero-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        v = [0, 0, 0]
        aspace = AffineSpace(V, v)
        self.assertIn([0, 0, 0], aspace)
        self.assertNotIn([1, 0, 0], aspace)

    def test_contains_one_dimension(self):
        """Test __contains__ with one-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        v = [0, 0, 1]
        aspace = AffineSpace(V, v)
        self.assertIn([0, 0, 1], aspace)
        self.assertIn([0, 0, 2], aspace)
        self.assertNotIn([1, 0, 1], aspace)

    def test_contains_with_matrix_object(self):
        """Test __contains__ with Matrix object."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        point = M([2, 3, 4])
        self.assertIn(point, aspace)

    def test_contains_with_wrong_field(self):
        """Test __contains__ with wrong field elements."""
        V = fn("V", R, 2)
        v = [1, 2]
        aspace = AffineSpace(V, v)
        # Complex number in real field - should fail type check
        self.assertNotIn([1+2j, 3], aspace)

    def test_init_with_none_representative(self):
        """Test __init__ with None as representative."""
        V = fn("V", R, 3)
        with self.assertRaises(TypeError):
            AffineSpace(V, None)

    def test_init_with_matrix_representative(self):
        """Test __init__ with Matrix object as representative."""
        V = fn("V", R, 3)
        v = M([1, 2, 3])
        aspace = AffineSpace(V, v)
        self.assertEqual(M(aspace.representative), M([1, 2, 3]))

    def test_init_zero_dimension_space(self):
        """Test __init__ with zero-dimension vector space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        v = [0, 0, 0]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.dim, 0)

    def test_init_one_dimension_space(self):
        """Test __init__ with one-dimension vector space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        v = [0, 0, 1]
        aspace = AffineSpace(V, v)
        self.assertEqual(aspace.dim, 1)

    def test_dim_zero_dimension(self):
        """Test dim property with zero-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        aspace = AffineSpace(V, [0, 0, 0])
        self.assertEqual(aspace.dim, 0)

    def test_dim_one_dimension(self):
        """Test dim property with one-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        aspace = AffineSpace(V, [0, 0, 1])
        self.assertEqual(aspace.dim, 1)

    def test_mul_with_zero(self):
        """Test __mul__ with zero scalar."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        result = aspace * 0
        self.assertEqual(M(result.representative), M([0, 0, 0]))

    def test_mul_with_one(self):
        """Test __mul__ with one scalar."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace * 1
        self.assertEqual(M(result.representative), M(v))

    def test_mul_with_fractional_scalar(self):
        """Test __mul__ with fractional scalar."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace * 0.5
        expected = M([0.5, 1, 1.5])
        self.assertEqual(result.vectorspace, V)
        result_mat = M(result.representative)
        for i in range(3):
            self.assertAlmostEqual(float(result_mat[i]), float(expected[i]))

    def test_mul_with_negative_scalar(self):
        """Test __mul__ with negative scalar."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace * -2
        self.assertEqual(M(result.representative), M([-2, -4, -6]))

    def test_mul_complex_field(self):
        """Test __mul__ with complex field."""
        V = fn("V", C, 2)
        v = [1+2j, 3+4j]
        aspace = AffineSpace(V, v)
        result = aspace * (2+3j)
        self.assertEqual(result.vectorspace, V)
        # Verify result is correct
        expected = V.mul(2+3j, v)
        self.assertEqual(M(result.representative), M(expected))

    def test_rmul_with_invalid_scalar(self):
        """Test __rmul__ with invalid scalar."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            (1+2j) * aspace

    def test_add_with_zero_vector(self):
        """Test __add__ with zero vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace + [0, 0, 0]
        self.assertEqual(result, aspace)

    def test_add_with_negative_vector(self):
        """Test __add__ with negative vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace + [-1, -2, -3]
        self.assertEqual(M(result.representative), M([0, 0, 0]))

    def test_sub_with_zero_vector(self):
        """Test __sub__ with zero vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace - [0, 0, 0]
        self.assertEqual(result, aspace)

    def test_sub_with_negative_vector(self):
        """Test __sub__ with negative vector."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        result = aspace - [-1, -2, -3]
        self.assertEqual(M(result.representative), M([2, 4, 6]))

    def test_add_with_none(self):
        """Test __add__ with None."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            aspace + None

    def test_sub_with_none(self):
        """Test __sub__ with None."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            aspace - None

    def test_radd_with_invalid_type(self):
        """Test __radd__ with invalid type."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            "not a vector" + aspace
        with self.assertRaises(TypeError):
            None + aspace

    def test_rsub_with_invalid_type(self):
        """Test __rsub__ with invalid type."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        with self.assertRaises(TypeError):
            "not a vector" - aspace
        with self.assertRaises(TypeError):
            None - aspace

    def test_neg_twice(self):
        """Test double negation."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertEqual(-(-aspace), aspace)

    def test_add_sub_cancellation(self):
        """Test that addition and subtraction cancel."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        u = [4, 5, 6]
        aspace = AffineSpace(V, v)
        result = (aspace + u) - u
        self.assertEqual(result, aspace)

    def test_mul_associative(self):
        """Test that scalar multiplication is associative."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        # (a * b) * aspace = a * (b * aspace)
        result1 = (2 * 3) * aspace
        result2 = 2 * (3 * aspace)
        self.assertEqual(result1, result2)

    def test_point_one_dimension(self):
        """Test point method with one-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0"])
        v = [0, 0, 1]
        aspace = AffineSpace(V, v)
        point = aspace.point()
        self.assertIn(point, aspace)
        point = aspace.point(arbitrary=True)
        self.assertIn(point, aspace)

    def test_point_with_std_zero(self):
        """Test point method with std=0."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        point = aspace.point(std=0)
        self.assertIn(point, aspace)

    def test_point_with_negative_std(self):
        """Test point method with negative std."""
        V = fn("V", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        point = aspace.point(std=-1)
        self.assertIn(point, aspace)

    def test_sum_with_zero_representative(self):
        """Test sum with zero vector representatives."""
        V = fn("V", R, 3)
        aspace1 = AffineSpace(V, [0, 0, 0])
        aspace2 = AffineSpace(V, [1, 2, 3])
        result = aspace1.sum(aspace2)
        self.assertEqual(M(result.representative), M([1, 2, 3]))

    def test_sum_commutative(self):
        """Test that sum is commutative."""
        V = fn("V", R, 3)
        aspace1 = AffineSpace(V, [1, 2, 3])
        aspace2 = AffineSpace(V, [4, 5, 6])
        result1 = aspace1.sum(aspace2)
        result2 = aspace2.sum(aspace1)
        self.assertEqual(result1, result2)

    def test_sum_with_same_affinespace(self):
        """Test sum with same affine space."""
        V = fn("V", R, 3)
        aspace = AffineSpace(V, [1, 2, 3])
        result = aspace.sum(aspace)
        self.assertEqual(M(result.representative), M([2, 4, 6]))

    def test_info_with_subspace(self):
        """Test info method with subspace."""
        V = fn("V", R, 3, constraints=["v0 + v1 == 0"])
        aspace = AffineSpace(V, [1, -1, 5])
        info_str = aspace.info()
        self.assertIn("Vector Space", info_str)
        self.assertIn("Representative", info_str)

    def test_info_with_zero_dimension(self):
        """Test info method with zero-dimension space."""
        V = fn("V", R, 3, constraints=["v0 == 0", "v1 == 0", "v2 == 0"])
        aspace = AffineSpace(V, [0, 0, 0])
        info_str = aspace.info()
        self.assertIn("Dimension", info_str)
        self.assertIn("0", info_str)

    def test_str_with_special_characters(self):
        """Test __str__ with special characters in name."""
        V = fn("V with spaces", R, 3)
        v = [1, 2, 3]
        aspace = AffineSpace(V, v)
        self.assertIn("V with spaces", str(aspace))


class TestFactoryFunctions(unittest.TestCase):
    """Test remaining factory functions."""

    def test_is_vectorspace_valid(self):
        """Test is_vectorspace with valid constraints."""
        constraints = ["v0 + 2*v1 == 0"]
        self.assertTrue(is_vectorspace(2, constraints))
        
        constraints = ["v0 + v1 == 0", "v1 + v2 == 0"]
        self.assertTrue(is_vectorspace(3, constraints))
        
        constraints = []
        self.assertTrue(is_vectorspace(3, constraints))

    def test_is_vectorspace_invalid_nonlinear(self):
        """Test is_vectorspace with nonlinear constraints."""
        constraints = ["v0 * v1 == 0"]  # Non-linear
        self.assertFalse(is_vectorspace(2, constraints))
        
        constraints = ["v0**2 == 0"]  # Quadratic
        self.assertFalse(is_vectorspace(2, constraints))

    def test_is_vectorspace_invalid_nonzero_constant(self):
        """Test is_vectorspace with nonzero constant."""
        constraints = ["v0 + v1 == 1"]  # Nonzero constant
        self.assertFalse(is_vectorspace(2, constraints))
        
        constraints = ["v0 == 5"]  # Nonzero constant
        self.assertFalse(is_vectorspace(2, constraints))

    def test_is_vectorspace_valid_homogeneous(self):
        """Test is_vectorspace with valid homogeneous constraints."""
        constraints = ["v0 + 2*v1 == 0", "v1 - v2 == 0"]
        self.assertTrue(is_vectorspace(3, constraints))

    def test_rowspace_basic(self):
        """Test rowspace factory function."""
        matrix = [[1, 2], [3, 4]]
        V = rowspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.field, R)
        self.assertEqual(V.dim, 2)  # Full rank matrix
        
        # Check that rows are in the space
        self.assertIn([1, 2], V)
        self.assertIn([3, 4], V)

    def test_rowspace_rank_deficient(self):
        """Test rowspace with rank-deficient matrix."""
        matrix = [[1, 2], [2, 4]]  # Second row is 2 * first row
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_rowspace_zero_matrix(self):
        """Test rowspace with zero matrix."""
        matrix = [[0, 0], [0, 0]]
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 0)

    def test_rowspace_complex_field(self):
        """Test rowspace with complex field."""
        matrix = [[1+2j, 3+4j], [5+6j, 7+8j]]
        V = rowspace("V", matrix, field=C)
        self.assertEqual(V.field, C)
        self.assertIn([1+2j, 3+4j], V)

    def test_columnspace_basic(self):
        """Test columnspace factory function."""
        matrix = [[1, 2], [3, 4]]
        V = columnspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.field, R)
        self.assertEqual(V.dim, 2)  # Full rank matrix

    def test_columnspace_rank_deficient(self):
        """Test columnspace with rank-deficient matrix."""
        matrix = [[1, 2], [2, 4]]  # Second column is 2 * first column
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_columnspace_zero_matrix(self):
        """Test columnspace with zero matrix."""
        matrix = [[0, 0], [0, 0]]
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 0)

    def test_columnspace_equals_image(self):
        """Test that columnspace equals image."""
        matrix = [[1, 2], [3, 4]]
        V1 = columnspace("V1", matrix)
        V2 = image("V2", matrix)
        self.assertEqual(V1, V2)

    def test_nullspace_basic(self):
        """Test nullspace factory function."""
        matrix = [[1, 2], [3, 4]]
        V = nullspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.field, R)
        # Full rank 2x2 matrix has null space of dimension 0
        self.assertEqual(V.dim, 0)

    def test_nullspace_rank_deficient(self):
        """Test nullspace with rank-deficient matrix."""
        matrix = [[1, 2], [2, 4]]  # Rank 1, so null space has dimension 1
        V = nullspace("V", matrix)
        self.assertEqual(V.dim, 1)
        # Check that null space vectors are actually in the null space
        basis = V.basis
        if basis:
            null_vec = basis[0]
            # M @ null_vec should be zero
            result = M(matrix) @ M(null_vec)
            self.assertTrue(result.is_zero_matrix)

    def test_nullspace_equals_kernel(self):
        """Test that nullspace equals kernel."""
        matrix = [[1, 2], [2, 4]]
        V1 = nullspace("V1", matrix)
        V2 = kernel("V2", matrix)
        self.assertEqual(V1, V2)

    def test_left_nullspace_basic(self):
        """Test left_nullspace factory function."""
        matrix = [[1, 2], [3, 4]]
        V = left_nullspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.field, R)
        # Full rank 2x2 matrix has left null space of dimension 0
        self.assertEqual(V.dim, 0)

    def test_left_nullspace_rank_deficient(self):
        """Test left_nullspace with rank-deficient matrix."""
        matrix = [[1, 2], [2, 4]]  # Rank 1, so left null space has dimension 1
        V = left_nullspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_left_nullspace_equals_nullspace_of_transpose(self):
        """Test that left_nullspace equals nullspace of transpose."""
        matrix = [[1, 2], [2, 4]]
        V1 = left_nullspace("V1", matrix)
        V2 = nullspace("V2", M(matrix).T)
        self.assertEqual(V1, V2)

    def test_rowspace_columnspace_relationship(self):
        """Test relationship between rowspace and columnspace."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        row_V = rowspace("Row", matrix)
        col_V = columnspace("Col", matrix)
        # Rank of row space equals rank of column space
        self.assertEqual(row_V.dim, col_V.dim)

    def test_nullspace_rowspace_orthogonal_complement(self):
        """Test that nullspace is orthogonal complement of rowspace."""
        matrix = [[1, 2], [2, 4]]
        null_V = nullspace("Null", matrix)
        row_V = rowspace("Row", matrix)
        # Dimension of null space + dimension of row space = number of columns
        self.assertEqual(null_V.dim + row_V.dim, 2)

    def test_columnspace_left_nullspace_orthogonal_complement(self):
        """Test that left nullspace is orthogonal complement of columnspace."""
        matrix = [[1, 2], [2, 4]]
        col_V = columnspace("Col", matrix)
        left_null_V = left_nullspace("LeftNull", matrix)
        # Dimension of column space + dimension of left null space = number of rows
        self.assertEqual(col_V.dim + left_null_V.dim, 2)

    def test_rowspace_with_matrix_object(self):
        """Test rowspace with Matrix object."""
        matrix = M([[1, 2], [3, 4]])
        V = rowspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.dim, 2)

    def test_columnspace_with_matrix_object(self):
        """Test columnspace with Matrix object."""
        matrix = M([[1, 2], [3, 4]])
        V = columnspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.dim, 2)

    def test_nullspace_with_matrix_object(self):
        """Test nullspace with Matrix object."""
        matrix = M([[1, 2], [2, 4]])
        V = nullspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.dim, 1)

    def test_left_nullspace_with_matrix_object(self):
        """Test left_nullspace with Matrix object."""
        matrix = M([[1, 2], [2, 4]])
        V = left_nullspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertEqual(V.dim, 1)

    def test_is_vectorspace_with_none(self):
        """Test is_vectorspace with None constraints."""
        # None should raise TypeError (constraints must be iterable)
        with self.assertRaises(TypeError):
            is_vectorspace(3, None)

    def test_is_vectorspace_zero_dimension(self):
        """Test is_vectorspace with zero dimension."""
        constraints = ["v0 == 0"]
        self.assertTrue(is_vectorspace(1, constraints))

    def test_is_vectorspace_one_dimension(self):
        """Test is_vectorspace with one dimension."""
        constraints = []
        self.assertTrue(is_vectorspace(1, constraints))

    def test_is_vectorspace_mixed_valid_invalid(self):
        """Test is_vectorspace with mix of valid and invalid constraints."""
        # If any constraint is invalid, should return False
        constraints = ["v0 + v1 == 0", "v0 * v1 == 0"]  # Second is nonlinear
        self.assertFalse(is_vectorspace(2, constraints))

    def test_rowspace_one_row(self):
        """Test rowspace with single row matrix."""
        matrix = [[1, 2, 3]]
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 1)
        self.assertIn([1, 2, 3], V)

    def test_rowspace_one_column(self):
        """Test rowspace with single column matrix."""
        matrix = [[1], [2], [3]]
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_rowspace_1x1_matrix(self):
        """Test rowspace with 1x1 matrix."""
        matrix = [[5]]
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 1)
        self.assertIn([5], V)

    def test_rowspace_large_matrix(self):
        """Test rowspace with large matrix."""
        matrix = [[i + j for j in range(10)] for i in range(5)]
        V = rowspace("V", matrix)
        self.assertIsInstance(V, VectorSpace)
        self.assertLessEqual(V.dim, 5)

    def test_columnspace_one_column(self):
        """Test columnspace with single column matrix."""
        matrix = [[1], [2], [3]]
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_columnspace_one_row(self):
        """Test columnspace with single row matrix."""
        matrix = [[1, 2, 3]]
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_columnspace_1x1_matrix(self):
        """Test columnspace with 1x1 matrix."""
        matrix = [[5]]
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_nullspace_one_row(self):
        """Test nullspace with single row matrix."""
        matrix = [[1, 2, 3]]
        V = nullspace("V", matrix)
        # Rank 1, so null space has dimension 2
        self.assertEqual(V.dim, 2)

    def test_nullspace_one_column(self):
        """Test nullspace with single column matrix."""
        matrix = [[1], [2], [3]]
        V = nullspace("V", matrix)
        # Full rank, so null space has dimension 0
        self.assertEqual(V.dim, 0)

    def test_nullspace_1x1_zero(self):
        """Test nullspace with 1x1 zero matrix."""
        matrix = [[0]]
        V = nullspace("V", matrix)
        self.assertEqual(V.dim, 1)

    def test_left_nullspace_one_row(self):
        """Test left_nullspace with single row matrix."""
        matrix = [[1, 2, 3]]
        V = left_nullspace("V", matrix)
        # Rank 1, so left null space has dimension 0
        self.assertEqual(V.dim, 0)

    def test_left_nullspace_one_column(self):
        """Test left_nullspace with single column matrix."""
        matrix = [[1], [2], [3]]
        V = left_nullspace("V", matrix)
        # Rank 1, so left null space has dimension 2
        self.assertEqual(V.dim, 2)

    def test_rowspace_with_invalid_matrix(self):
        """Test rowspace with invalid matrix input."""
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((TypeError, ValueError)):
            rowspace("V", "not a matrix")

    def test_columnspace_with_invalid_matrix(self):
        """Test columnspace with invalid matrix input."""
        with self.assertRaises((TypeError, ValueError)):
            columnspace("V", "not a matrix")

    def test_nullspace_with_invalid_matrix(self):
        """Test nullspace with invalid matrix input."""
        with self.assertRaises((TypeError, ValueError)):
            nullspace("V", "not a matrix")

    def test_left_nullspace_with_invalid_matrix(self):
        """Test left_nullspace with invalid matrix input."""
        with self.assertRaises((TypeError, ValueError)):
            left_nullspace("V", "not a matrix")

    def test_rowspace_empty_matrix(self):
        """Test rowspace with empty matrix."""
        matrix = []
        V = rowspace("V", matrix)
        self.assertEqual(V.dim, 0)

    def test_columnspace_empty_matrix(self):
        """Test columnspace with empty matrix."""
        matrix = [[]]
        V = columnspace("V", matrix)
        self.assertEqual(V.dim, 0)

    def test_fundamental_theorem_rank_nullity(self):
        """Test rank-nullity theorem: rank + nullity = number of columns."""
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        row_V = rowspace("Row", matrix)
        null_V = nullspace("Null", matrix)
        self.assertEqual(row_V.dim + null_V.dim, 4)  # 4 columns

    def test_fundamental_theorem_row_column_rank(self):
        """Test that row rank equals column rank."""
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        row_V = rowspace("Row", matrix)
        col_V = columnspace("Col", matrix)
        self.assertEqual(row_V.dim, col_V.dim)


if __name__ == '__main__':
    unittest.main()

