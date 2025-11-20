"""
Unit tests for the ablina.linearmap module.
"""

import unittest
from ablina import *


class TestLinearMapError(unittest.TestCase):
    """Test LinearMapError exception."""

    def test_linear_map_error_init_no_message(self):
        """Test LinearMapError initialization with no message."""
        error = LinearMapError()
        self.assertEqual(str(error), "")

    def test_linear_map_error_init_with_message(self):
        """Test LinearMapError initialization with message."""
        error = LinearMapError("Custom error message")
        self.assertEqual(str(error), "Custom error message")

    def test_linear_map_error_is_exception(self):
        """Test LinearMapError is an Exception."""
        self.assertIsInstance(LinearMapError(), Exception)


class TestLinearMapInit(unittest.TestCase):
    """Test LinearMap.__init__ method."""

    def test_init_with_matrix(self):
        """Test LinearMap initialization with matrix."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.zeros(3, 2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.name, "T")
        self.assertEqual(lm.domain, domain)
        self.assertEqual(lm.codomain, codomain)

    def test_init_with_mapping(self):
        """Test LinearMap initialization with mapping."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        self.assertEqual(lm.name, "T")
        self.assertEqual(lm.domain, domain)
        self.assertEqual(lm.codomain, codomain)

    def test_init_with_both_matrix_and_mapping(self):
        """Test LinearMap initialization with both matrix and mapping."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        def mapping(vec):
            return vec
        lm = LinearMap("T", domain, codomain, mapping=mapping, matrix=matrix)
        self.assertEqual(lm.name, "T")

    def test_init_without_matrix_or_mapping(self):
        """Test LinearMap initialization without matrix or mapping."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        with self.assertRaises(LinearMapError):
            LinearMap("T", domain, codomain)

    def test_init_invalid_domain_type(self):
        """Test LinearMap initialization with invalid domain type."""
        codomain = fn("W", R, 2)
        with self.assertRaises(TypeError):
            LinearMap("T", "not a VectorSpace", codomain, matrix=M.eye(2))

    def test_init_invalid_codomain_type(self):
        """Test LinearMap initialization with invalid codomain type."""
        domain = fn("V", R, 2)
        with self.assertRaises(TypeError):
            LinearMap("T", domain, "not a VectorSpace", matrix=M.eye(2))

    def test_init_different_fields_raises_error(self):
        """Test LinearMap initialization with different fields raises error."""
        domain = fn("V", R, 2)
        codomain = fn("W", C, 2)
        with self.assertRaises(LinearMapError):
            LinearMap("T", domain, codomain, matrix=M.eye(2))

    def test_init_invalid_matrix_shape(self):
        """Test LinearMap initialization with invalid matrix shape."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.eye(2)  # Wrong shape
        with self.assertRaises(ValueError):
            LinearMap("T", domain, codomain, matrix=matrix)

    def test_init_mapping_wrong_arity(self):
        """Test LinearMap initialization with mapping of wrong arity."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def wrong_mapping(u, v):
            return u
        with self.assertRaises(TypeError):
            LinearMap("T", domain, codomain, mapping=wrong_mapping)


class TestLinearMapProperties(unittest.TestCase):
    """Test LinearMap properties."""

    def test_field_property(self):
        """Test field property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        self.assertEqual(lm.field, R)

    def test_domain_property(self):
        """Test domain property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        self.assertEqual(lm.domain, domain)

    def test_codomain_property(self):
        """Test codomain property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        self.assertEqual(lm.codomain, codomain)

    def test_mapping_property(self):
        """Test mapping property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return vec
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        self.assertIsNotNone(lm.mapping)
        self.assertTrue(callable(lm.mapping))

    def test_matrix_property(self):
        """Test matrix property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.matrix, matrix)

    def test_rank_property(self):
        """Test rank property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.rank, 2)

    def test_rank_zero_map(self):
        """Test rank of zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.zeros(3, 2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.rank, 0)

    def test_nullity_property(self):
        """Test nullity property."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.nullity, 0)

    def test_nullity_zero_map(self):
        """Test nullity of zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.zeros(3, 2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertEqual(lm.nullity, domain.dim)


class TestLinearMapRepr(unittest.TestCase):
    """Test LinearMap.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        repr_str = repr(lm)
        self.assertIn("LinearMap", repr_str)
        self.assertIn("'T'", repr_str)


class TestLinearMapStr(unittest.TestCase):
    """Test LinearMap.__str__ method."""

    def test_str_returns_name(self):
        """Test __str__ returns the map name."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        self.assertEqual(str(lm), "T")


class TestLinearMapEq(unittest.TestCase):
    """Test LinearMap.__eq__ method."""

    def test_eq_same_map(self):
        """Test equality with same map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        lm1 = LinearMap("T1", domain, codomain, matrix=matrix)
        lm2 = LinearMap("T2", domain, codomain, matrix=matrix)
        self.assertEqual(lm1, lm2)

    def test_eq_different_matrix(self):
        """Test equality with different matrix."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm1 = LinearMap("T1", domain, codomain, matrix=M.eye(2))
        lm2 = LinearMap("T2", domain, codomain, matrix=2 * M.eye(2))
        self.assertNotEqual(lm1, lm2)

    def test_eq_different_domain(self):
        """Test equality with different domain."""
        domain1 = fn("V", R, 2)
        domain2 = fn("U", R, 3)
        codomain = fn("W", R, 2)
        lm1 = LinearMap("T1", domain1, codomain, matrix=M.zeros(2, 2))
        lm2 = LinearMap("T2", domain2, codomain, matrix=M.zeros(2, 3))
        self.assertNotEqual(lm1, lm2)

    def test_eq_not_linear_map(self):
        """Test equality with non-LinearMap object."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        self.assertNotEqual(lm, "not a LinearMap")


class TestLinearMapAdd(unittest.TestCase):
    """Test LinearMap.__add__ method."""

    def test_add_compatible_maps(self):
        """Test addition of compatible linear maps."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping1(vec):
            return [vec[0] * 2, vec[1] * 2]
        def mapping2(vec):
            return [vec[0] * 3, vec[1] * 3]
        lm1 = LinearMap("T1", domain, codomain, mapping=mapping1)
        lm2 = LinearMap("T2", domain, codomain, mapping=mapping2)
        lm3 = lm1 + lm2
        self.assertIsInstance(lm3, LinearMap)
        result = lm3([1, 1])
        # Should be [5, 5] = [2, 2] + [3, 3]
        self.assertEqual(len(result), 2)

    def test_add_with_zero_map(self):
        """Test addition with zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm1 = LinearMap("T1", domain, codomain, mapping=mapping)
        zero_map = LinearMap("T0", domain, codomain, matrix=M.zeros(2, 2))
        lm2 = lm1 + zero_map
        # Adding zero map should return same map
        result1 = lm1([1, 1])
        result2 = lm2([1, 1])
        self.assertEqual(list(result1), list(result2))

    def test_add_incompatible_domains_raises_error(self):
        """Test addition with incompatible domains raises error."""
        domain1 = fn("V", R, 2)
        domain2 = fn("U", R, 3)
        codomain = fn("W", R, 2)
        lm1 = LinearMap("T1", domain1, codomain, matrix=M.zeros(2, 2))
        lm2 = LinearMap("T2", domain2, codomain, matrix=M.zeros(2, 3))
        with self.assertRaises(LinearMapError):
            lm1 + lm2


class TestLinearMapMul(unittest.TestCase):
    """Test LinearMap.__mul__ method."""

    def test_mul_with_scalar(self):
        """Test multiplication with scalar."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm1 = LinearMap("T", domain, codomain, mapping=mapping)
        lm2 = 3 * lm1
        self.assertIsInstance(lm2, LinearMap)
        result = lm2([1, 1])
        # Should be [6, 6] = 3 * [2, 2]
        self.assertEqual(len(result), 2)

    def test_mul_with_zero_scalar(self):
        """Test multiplication with zero scalar."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm1 = LinearMap("T", domain, codomain, mapping=mapping)
        zero_map = 0 * lm1
        result = zero_map([1, 1])
        # Should be zero vector
        result_list = list(result)
        self.assertEqual(result_list[0], 0)
        self.assertEqual(result_list[1], 0)

    def test_mul_with_negative_scalar(self):
        """Test multiplication with negative scalar."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm1 = LinearMap("T", domain, codomain, mapping=mapping)
        neg_map = -1 * lm1
        result = neg_map([1, 1])
        # Should be [-2, -2]
        result_list = list(result)
        self.assertEqual(result_list[0], -2)
        self.assertEqual(result_list[1], -2)

    def test_mul_invalid_scalar_raises_error(self):
        """Test multiplication with invalid scalar raises TypeError."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        with self.assertRaises(TypeError):
            lm * "not a scalar"

    def test_rmul(self):
        """Test __rmul__ method (right multiplication)."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm1 = LinearMap("T", domain, codomain, mapping=mapping)
        # Test right multiplication: 3 * lm1
        lm2 = 3 * lm1
        self.assertIsInstance(lm2, LinearMap)
        result = lm2([1, 1])
        # Should be [6, 6] = 3 * [2, 2]
        self.assertEqual(len(result), 2)


class TestLinearMapCall(unittest.TestCase):
    """Test LinearMap.__call__ method."""

    def test_call_with_vector(self):
        """Test __call__ with vector."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        result = lm([1, 2])
        self.assertEqual(len(result), 2)
        # Should be [2, 4]
        result_list = list(result)
        self.assertEqual(result_list[0], 2)
        self.assertEqual(result_list[1], 4)

    def test_call_with_subspace(self):
        """Test __call__ with subspace."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        result = lm(subspace)
        self.assertIsNotNone(result)

    def test_call_with_zero_vector(self):
        """Test __call__ with zero vector."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        zero_vec = [0, 0]
        result = lm(zero_vec)
        # Should map to zero
        result_list = list(result)
        self.assertEqual(result_list[0], 0)
        self.assertEqual(result_list[1], 0)


class TestLinearMapMethods(unittest.TestCase):
    """Test LinearMap methods."""

    def test_info(self):
        """Test info method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        info_str = lm.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("T", info_str)
        self.assertIn("Rank", info_str)
        self.assertIn("Nullity", info_str)

    def test_change_of_basis(self):
        """Test change_of_basis method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        new_basis = [[1, 1], [1, -1]]
        mat, dom_change, cod_change = lm.change_of_basis(domain_basis=new_basis)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(dom_change)
        self.assertIsNotNone(cod_change)

    def test_change_of_basis_no_change(self):
        """Test change_of_basis with no change (None bases)."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        mat, dom_change, cod_change = lm.change_of_basis()
        # Should return identity matrices
        self.assertIsNotNone(mat)
        self.assertIsNotNone(dom_change)
        self.assertIsNotNone(cod_change)

    def test_restriction(self):
        """Test restriction method."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        restricted = lm.restriction(subspace)
        self.assertIsInstance(restricted, LinearMap)

    def test_restriction_to_full_domain(self):
        """Test restriction to full domain."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        # Restricting to full domain should return equivalent map
        restricted = lm.restriction(domain)
        self.assertIsInstance(restricted, LinearMap)

    def test_restriction_invalid_subspace_raises_error(self):
        """Test restriction with invalid subspace raises TypeError."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        # Not a subspace of domain
        subspace = fn("U", R, 3)
        with self.assertRaises(TypeError):
            lm.restriction(subspace)

    def test_composition(self):
        """Test composition method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping1(vec):
            return [vec[0] * 2, vec[1] * 2]
        def mapping2(vec):
            return [vec[0] * 3, vec[1] * 3]
        lm1 = LinearMap("T1", codomain, codomain, mapping=mapping1)
        lm2 = LinearMap("T2", domain, codomain, mapping=mapping2)
        comp = lm1.composition(lm2)
        self.assertIsInstance(comp, LinearMap)

    def test_composition_with_identity(self):
        """Test composition with identity map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        id_map = IdentityMap(codomain)
        comp = id_map.composition(lm)
        # Composing with identity should return same map
        result1 = lm([1, 1])
        result2 = comp([1, 1])
        self.assertEqual(list(result1), list(result2))

    def test_composition_with_zero_map(self):
        """Test composition with zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        lm = LinearMap("T", domain, codomain, mapping=mapping)
        zero_map = LinearMap("T0", codomain, codomain, matrix=M.zeros(2, 2))
        comp = zero_map.composition(lm)
        # Composing with zero map should return zero map
        result = comp([1, 1])
        result_list = list(result)
        self.assertEqual(result_list[0], 0)
        self.assertEqual(result_list[1], 0)

    def test_composition_incompatible_raises_error(self):
        """Test composition with incompatible maps raises error."""
        domain = fn("V", R, 2)
        codomain1 = fn("W1", R, 2)
        codomain2 = fn("W2", R, 3)
        lm1 = LinearMap("T1", codomain2, codomain2, matrix=M.eye(3))
        lm2 = LinearMap("T2", domain, codomain1, matrix=M.eye(2))
        with self.assertRaises(LinearMapError):
            lm1.composition(lm2)

    def test_image(self):
        """Test image method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        img = lm.image()
        self.assertIsNotNone(img)
        self.assertEqual(img.dim, 2)

    def test_image_zero_map(self):
        """Test image of zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.zeros(3, 2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        img = lm.image()
        self.assertEqual(img.dim, 0)

    def test_image_surjective_map(self):
        """Test image of surjective map."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        img = lm.image()
        self.assertEqual(img.dim, codomain.dim)

    def test_kernel(self):
        """Test kernel method."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        ker = lm.kernel()
        self.assertIsNotNone(ker)
        self.assertEqual(ker.dim, 1)

    def test_kernel_injective_map(self):
        """Test kernel of injective map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        ker = lm.kernel()
        self.assertEqual(ker.dim, 0)

    def test_kernel_zero_map(self):
        """Test kernel of zero map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M.zeros(3, 2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        ker = lm.kernel()
        self.assertEqual(ker.dim, domain.dim)

    def test_range_alias(self):
        """Test range alias for image."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        # range should be same as image
        self.assertEqual(lm.range().dim, lm.image().dim)

    def test_nullspace_alias(self):
        """Test nullspace alias for kernel."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        # nullspace should be same as kernel
        self.assertEqual(lm.nullspace().dim, lm.kernel().dim)

    def test_is_injective(self):
        """Test is_injective method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertTrue(lm.is_injective())

    def test_is_surjective(self):
        """Test is_surjective method."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertTrue(lm.is_surjective())

    def test_is_bijective(self):
        """Test is_bijective method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        matrix = M.eye(2)
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertTrue(lm.is_bijective())

    def test_is_injective_not_injective(self):
        """Test is_injective with non-injective map."""
        domain = fn("V", R, 3)
        codomain = fn("W", R, 2)
        matrix = M([[1, 0, 0], [0, 1, 0]])  # Rank 2, nullity 1
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertFalse(lm.is_injective())

    def test_is_surjective_not_surjective(self):
        """Test is_surjective with non-surjective map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])  # Rank 2, rows 3
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertFalse(lm.is_surjective())

    def test_is_bijective_not_bijective(self):
        """Test is_bijective with non-bijective map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        self.assertFalse(lm.is_bijective())

    def test_adjoint_not_implemented(self):
        """Test adjoint raises NotImplementedError."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        with self.assertRaises(NotImplementedError):
            lm.adjoint()

    def test_pseudoinverse_not_implemented(self):
        """Test pseudoinverse raises NotImplementedError."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        with self.assertRaises(NotImplementedError):
            lm.pseudoinverse()

    def test_inverse(self):
        """Test inverse method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        lm = LinearMap("T", domain, codomain, matrix=M.eye(2))
        inv = lm.inverse()
        self.assertIsInstance(inv, LinearMap)
        self.assertEqual(inv.domain, codomain)
        self.assertEqual(inv.codomain, domain)

    def test_inverse_non_bijective_raises_error(self):
        """Test inverse of non-bijective map raises LinearMapError."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])  # Not bijective
        lm = LinearMap("T", domain, codomain, matrix=matrix)
        with self.assertRaises(LinearMapError):
            lm.inverse()


class TestLinearOperator(unittest.TestCase):
    """Test LinearOperator class."""

    def test_init(self):
        """Test LinearOperator initialization."""
        vs = fn("V", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        op = LinearOperator("T", vs, mapping=mapping)
        self.assertEqual(op.domain, vs)
        self.assertEqual(op.codomain, vs)

    def test_repr(self):
        """Test LinearOperator.__repr__ method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        repr_str = repr(op)
        self.assertIn("LinearOperator", repr_str)

    def test_pow(self):
        """Test LinearOperator.__pow__ method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        op2 = op ** 2
        self.assertIsInstance(op2, LinearOperator)
        self.assertEqual(op2.domain, vs)

    def test_pow_zero(self):
        """Test LinearOperator.__pow__ with zero exponent."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        op0 = op ** 0
        # Should be identity
        self.assertIsInstance(op0, LinearOperator)
        v = [1, 2]
        result = op0(v)
        result_list = list(result)
        self.assertEqual(result_list[0], 1)
        self.assertEqual(result_list[1], 2)

    def test_pow_one(self):
        """Test LinearOperator.__pow__ with exponent 1."""
        vs = fn("V", R, 2)
        def mapping(vec):
            return [vec[0] * 2, vec[1] * 2]
        op = LinearOperator("T", vs, mapping=mapping)
        op1 = op ** 1
        # Should be same as original
        v = [1, 2]
        result1 = op(v)
        result2 = op1(v)
        self.assertEqual(list(result1), list(result2))

    def test_pow_negative(self):
        """Test LinearOperator.__pow__ with negative exponent."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        op_neg = op ** -1
        # Should be inverse
        self.assertIsInstance(op_neg, LinearOperator)

    def test_change_of_basis(self):
        """Test LinearOperator.change_of_basis method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        new_basis = [[1, 1], [1, -1]]
        mat, change = op.change_of_basis(new_basis)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(change)

    def test_change_of_basis_same_basis(self):
        """Test LinearOperator.change_of_basis with same basis."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        same_basis = vs.basis
        mat, change = op.change_of_basis(same_basis)
        # Should return same matrix (approximately)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(change)

    def test_inverse(self):
        """Test LinearOperator.inverse method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        inv = op.inverse()
        self.assertIsInstance(inv, LinearOperator)

    def test_inverse_composition(self):
        """Test that inverse composed with original gives identity."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        inv = op.inverse()
        comp = op.composition(inv)
        # Should be identity
        v = [1, 2]
        result = comp(v)
        result_list = list(result)
        self.assertEqual(result_list[0], 1)
        self.assertEqual(result_list[1], 2)

    def test_inverse_non_invertible_raises_error(self):
        """Test inverse of non-invertible operator raises error."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, 0]])  # Not invertible
        op = LinearOperator("T", vs, matrix=matrix)
        with self.assertRaises(LinearMapError):
            op.inverse()

    def test_is_invariant_subspace(self):
        """Test is_invariant_subspace method."""
        vs = fn("V", R, 3)
        matrix = M([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        op = LinearOperator("T", vs, matrix=matrix)
        subspace = fn("U", R, 3, basis=[[1, 0, 0], [0, 1, 0]])
        self.assertTrue(op.is_invariant_subspace(subspace))

    def test_is_invariant_subspace_not_invariant(self):
        """Test is_invariant_subspace with non-invariant subspace."""
        vs = fn("V", R, 2)
        matrix = M([[0, 1], [1, 0]])  # Swap operator
        op = LinearOperator("T", vs, matrix=matrix)
        subspace = fn("U", R, 2, basis=[[1, 0]])
        self.assertFalse(op.is_invariant_subspace(subspace))

    def test_is_invariant_subspace_zero_subspace(self):
        """Test is_invariant_subspace with zero subspace."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        # Zero subspace is always invariant
        zero_subspace = fn("U", R, 2, constraints=["v0 == 0", "v1 == 0"])
        self.assertTrue(op.is_invariant_subspace(zero_subspace))

    def test_is_invariant_subspace_full_space(self):
        """Test is_invariant_subspace with full space."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        # Full space is always invariant
        self.assertTrue(op.is_invariant_subspace(vs))

    def test_is_invariant_subspace_invalid_raises_error(self):
        """Test is_invariant_subspace with invalid subspace raises TypeError."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        # Not a subspace of vs
        subspace = fn("U", R, 3)
        with self.assertRaises(TypeError):
            op.is_invariant_subspace(subspace)

    def test_is_symmetric(self):
        """Test is_symmetric method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        result = op.is_symmetric(ip)
        self.assertTrue(result)

    def test_is_symmetric_complex_raises_error(self):
        """Test is_symmetric with complex field raises error."""
        vs = fn("V", C, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        with self.assertRaises(LinearMapError):
            op.is_symmetric(ip)

    def test_is_hermitian(self):
        """Test is_hermitian method."""
        vs = fn("V", C, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        result = op.is_hermitian(ip)
        self.assertTrue(result)

    def test_is_orthogonal(self):
        """Test is_orthogonal method."""
        vs = fn("V", R, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        result = op.is_orthogonal(ip)
        self.assertTrue(result)

    def test_is_orthogonal_complex_raises_error(self):
        """Test is_orthogonal with complex field raises error."""
        vs = fn("V", C, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        with self.assertRaises(LinearMapError):
            op.is_orthogonal(ip)

    def test_is_unitary(self):
        """Test is_unitary method."""
        vs = fn("V", C, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        result = op.is_unitary(ip)
        self.assertTrue(result)

    def test_is_normal(self):
        """Test is_normal method."""
        vs = fn("V", C, 2)
        op = LinearOperator("T", vs, matrix=M.eye(2))
        ip = InnerProduct("ip", vs, matrix=M.eye(2))
        result = op.is_normal(ip)
        self.assertTrue(result)


class TestLinearFunctional(unittest.TestCase):
    """Test LinearFunctional class."""

    def test_init_with_mapping(self):
        """Test LinearFunctional initialization with mapping."""
        vs = fn("V", R, 2)
        def mapping(vec):
            return vec[0] + vec[1]
        lf = LinearFunctional("f", vs, mapping=mapping)
        self.assertEqual(lf.domain, vs)
        self.assertEqual(lf.codomain.dim, 1)

    def test_init_with_matrix(self):
        """Test LinearFunctional initialization with matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 1]])
        lf = LinearFunctional("f", vs, matrix=matrix)
        self.assertEqual(lf.domain, vs)
        self.assertEqual(lf.codomain.dim, 1)

    def test_repr(self):
        """Test LinearFunctional.__repr__ method."""
        vs = fn("V", R, 2)
        lf = LinearFunctional("f", vs, matrix=M([[1, 1]]))
        repr_str = repr(lf)
        self.assertIn("LinearFunctional", repr_str)

    def test_restriction(self):
        """Test LinearFunctional.restriction method."""
        vs = fn("V", R, 3)
        def mapping(vec):
            return vec[0] + vec[1] + vec[2]
        lf = LinearFunctional("f", vs, mapping=mapping)
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        restricted = lf.restriction(subspace)
        self.assertIsInstance(restricted, LinearFunctional)

    def test_restriction_to_full_domain(self):
        """Test LinearFunctional.restriction to full domain."""
        vs = fn("V", R, 2)
        def mapping(vec):
            return vec[0] + vec[1]
        lf = LinearFunctional("f", vs, mapping=mapping)
        restricted = lf.restriction(vs)
        self.assertIsInstance(restricted, LinearFunctional)

    def test_restriction_invalid_subspace_raises_error(self):
        """Test restriction with invalid subspace raises TypeError."""
        vs = fn("V", R, 2)
        lf = LinearFunctional("f", vs, matrix=M([[1, 1]]))
        # Not a subspace of vs
        subspace = fn("U", R, 3)
        with self.assertRaises(TypeError):
            lf.restriction(subspace)


class TestIsomorphism(unittest.TestCase):
    """Test Isomorphism class."""

    def test_init_bijective(self):
        """Test Isomorphism initialization with bijective map."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        iso = Isomorphism("T", domain, codomain, matrix=M.eye(2))
        self.assertEqual(iso.domain, domain)
        self.assertEqual(iso.codomain, codomain)

    def test_init_non_bijective_raises_error(self):
        """Test Isomorphism initialization with non-bijective map raises error."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 3)
        matrix = M([[1, 0], [0, 1], [0, 0]])  # Not bijective
        with self.assertRaises(LinearMapError):
            Isomorphism("T", domain, codomain, matrix=matrix)

    def test_repr(self):
        """Test Isomorphism.__repr__ method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        iso = Isomorphism("T", domain, codomain, matrix=M.eye(2))
        repr_str = repr(iso)
        self.assertIn("Isomorphism", repr_str)

    def test_info(self):
        """Test Isomorphism.info method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        iso = Isomorphism("T", domain, codomain, matrix=M.eye(2))
        info_str = iso.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("T", info_str)

    def test_inverse(self):
        """Test Isomorphism.inverse method."""
        domain = fn("V", R, 2)
        codomain = fn("W", R, 2)
        iso = Isomorphism("T", domain, codomain, matrix=M.eye(2))
        inv = iso.inverse()
        self.assertIsInstance(inv, Isomorphism)
        self.assertEqual(inv.domain, codomain)
        self.assertEqual(inv.codomain, domain)


class TestIdentityMap(unittest.TestCase):
    """Test IdentityMap class."""

    def test_init(self):
        """Test IdentityMap initialization."""
        vs = fn("V", R, 2)
        id_map = IdentityMap(vs)
        self.assertEqual(id_map.domain, vs)
        self.assertEqual(id_map.codomain, vs)
        self.assertEqual(id_map.name, "Id")

    def test_repr(self):
        """Test IdentityMap.__repr__ method."""
        vs = fn("V", R, 2)
        id_map = IdentityMap(vs)
        repr_str = repr(id_map)
        self.assertIn("IdentityMap", repr_str)

    def test_info(self):
        """Test IdentityMap.info method."""
        vs = fn("V", R, 2)
        id_map = IdentityMap(vs)
        info_str = id_map.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("Id", info_str)

    def test_inverse(self):
        """Test IdentityMap.inverse method."""
        vs = fn("V", R, 2)
        id_map = IdentityMap(vs)
        inv = id_map.inverse()
        self.assertIsInstance(inv, IdentityMap)
        self.assertEqual(inv.domain, vs)
        # Identity's inverse should be itself
        self.assertIs(inv, id_map)

    def test_call(self):
        """Test IdentityMap.__call__ method."""
        vs = fn("V", R, 2)
        id_map = IdentityMap(vs)
        v = [1, 2]
        result = id_map(v)
        # Should return the same vector
        result_list = list(result)
        self.assertEqual(result_list[0], 1)
        self.assertEqual(result_list[1], 2)


if __name__ == '__main__':
    unittest.main()

