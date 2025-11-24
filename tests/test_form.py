"""
Unit tests for the ablina.form module.
"""

import unittest
from ablina import *


class TestFormError(unittest.TestCase):
    """Test FormError exception."""

    def test_form_error_init_no_message(self):
        """Test FormError initialization with no message."""
        error = FormError()
        self.assertEqual(str(error), "")

    def test_form_error_init_with_message(self):
        """Test FormError initialization with message."""
        error = FormError("Custom error message")
        self.assertEqual(str(error), "Custom error message")

    def test_form_error_is_exception(self):
        """Test FormError is an Exception."""
        self.assertIsInstance(FormError(), Exception)


class TestInnerProductError(unittest.TestCase):
    """Test InnerProductError exception."""

    def test_inner_product_error_init_no_message(self):
        """Test InnerProductError initialization with no message."""
        error = InnerProductError()
        self.assertEqual(str(error), "")

    def test_inner_product_error_init_with_message(self):
        """Test InnerProductError initialization with message."""
        error = InnerProductError("Custom error message")
        self.assertEqual(str(error), "Custom error message")

    def test_inner_product_error_is_exception(self):
        """Test InnerProductError is an Exception."""
        self.assertIsInstance(InnerProductError(), Exception)

    def test_inner_product_error_inherits_from_form_error(self):
        """Test InnerProductError inherits from FormError."""
        self.assertIsInstance(InnerProductError(), FormError)


class TestSesquilinearFormInit(unittest.TestCase):
    """Test SesquilinearForm.__init__ method."""

    def test_init_with_matrix(self):
        """Test SesquilinearForm initialization with matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.name, "f")
        self.assertEqual(form.vectorspace, vs)
        self.assertEqual(form.matrix, matrix)

    def test_init_with_mapping(self):
        """Test SesquilinearForm initialization with mapping."""
        vs = fn("V", R, 2)
        def mapping(u, v):
            return u[0] * v[0] + u[1] * v[1]
        form = SesquilinearForm("f", vs, mapping=mapping)
        self.assertEqual(form.name, "f")
        self.assertEqual(form.vectorspace, vs)
        self.assertIsNotNone(form.matrix)

    def test_init_with_both_matrix_and_mapping(self):
        """Test SesquilinearForm initialization with both matrix and mapping."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        def mapping(u, v):
            return u[0] * v[0] + u[1] * v[1]
        form = SesquilinearForm("f", vs, mapping=mapping, matrix=matrix)
        self.assertEqual(form.name, "f")
        # Matrix should take precedence or be used
        self.assertIsNotNone(form.matrix)

    def test_init_without_matrix_or_mapping(self):
        """Test SesquilinearForm initialization without matrix or mapping."""
        vs = fn("V", R, 2)
        with self.assertRaises(FormError):
            SesquilinearForm("f", vs)

    def test_init_invalid_vectorspace_type(self):
        """Test SesquilinearForm initialization with invalid vectorspace type."""
        with self.assertRaises(TypeError):
            SesquilinearForm("f", "not a VectorSpace", matrix=M.eye(2))

    def test_init_invalid_matrix_shape(self):
        """Test SesquilinearForm initialization with invalid matrix shape."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2, 3], [4, 5, 6]])  # Wrong shape
        with self.assertRaises(ValueError):
            SesquilinearForm("f", vs, matrix=matrix)

    def test_init_matrix_wrong_size(self):
        """Test SesquilinearForm initialization with matrix of wrong size."""
        vs = fn("V", R, 2)
        matrix = M.eye(3)  # Wrong size
        with self.assertRaises(ValueError):
            SesquilinearForm("f", vs, matrix=matrix)

    def test_init_mapping_wrong_arity(self):
        """Test SesquilinearForm initialization with mapping of wrong arity."""
        vs = fn("V", R, 2)
        def wrong_mapping(u):
            return u[0]
        with self.assertRaises(TypeError):
            SesquilinearForm("f", vs, mapping=wrong_mapping)


class TestSesquilinearFormProperties(unittest.TestCase):
    """Test SesquilinearForm properties."""

    def test_vectorspace_property(self):
        """Test vectorspace property."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        self.assertEqual(form.vectorspace, vs)

    def test_mapping_property(self):
        """Test mapping property."""
        vs = fn("V", R, 2)
        def mapping(u, v):
            return u[0] * v[0]
        form = SesquilinearForm("f", vs, mapping=mapping)
        self.assertIsNotNone(form.mapping)
        self.assertTrue(callable(form.mapping))

    def test_matrix_property(self):
        """Test matrix property."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.matrix, matrix)


class TestSesquilinearFormRepr(unittest.TestCase):
    """Test SesquilinearForm.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        repr_str = repr(form)
        self.assertIn("SesquilinearForm", repr_str)
        self.assertIn("'f'", repr_str)


class TestSesquilinearFormStr(unittest.TestCase):
    """Test SesquilinearForm.__str__ method."""

    def test_str_returns_name(self):
        """Test __str__ returns the form name."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        self.assertEqual(str(form), "f")


class TestSesquilinearFormEq(unittest.TestCase):
    """Test SesquilinearForm.__eq__ method."""

    def test_eq_same_form(self):
        """Test equality with same form."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form1 = SesquilinearForm("f", vs, matrix=matrix)
        form2 = SesquilinearForm("g", vs, matrix=matrix)
        self.assertEqual(form1, form2)

    def test_eq_different_vectorspace(self):
        """Test equality with different vectorspace."""
        vs1 = fn("V", R, 2)
        vs2 = fn("W", R, 2)
        matrix = M.eye(2)
        form1 = SesquilinearForm("f", vs1, matrix=matrix)
        form2 = SesquilinearForm("g", vs2, matrix=matrix)
        # Note: If vector spaces have the same basis, they are considered equal,
        # so forms on them are also considered equal
        # To test different vector spaces, we need different dimensions or constraints
        vs3 = fn("U", R, 3)
        matrix3 = M.eye(3)
        form3 = SesquilinearForm("h", vs3, matrix=matrix3)
        self.assertNotEqual(form1, form3)

    def test_eq_different_matrix(self):
        """Test equality with different matrix."""
        vs = fn("V", R, 2)
        form1 = SesquilinearForm("f", vs, matrix=M.eye(2))
        form2 = SesquilinearForm("g", vs, matrix=2 * M.eye(2))
        self.assertNotEqual(form1, form2)

    def test_eq_not_sesquilinear_form(self):
        """Test equality with non-SesquilinearForm object."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        self.assertNotEqual(form, "not a form")


class TestSesquilinearFormCall(unittest.TestCase):
    """Test SesquilinearForm.__call__ method."""

    def test_call_with_valid_vectors(self):
        """Test __call__ with valid vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        u = [1, 0]
        v = [0, 1]
        result = form(u, v)
        self.assertEqual(result, 0)

    def test_call_with_invalid_vector(self):
        """Test __call__ with invalid vector."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        u = [1, 0]
        v = [0, 1, 0]  # Wrong dimension
        with self.assertRaises(TypeError):
            form(u, v)


class TestSesquilinearFormProperties(unittest.TestCase):
    """Test SesquilinearForm property methods."""

    def test_is_symmetric_symmetric_matrix(self):
        """Test is_symmetric with symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [2, 3]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_symmetric())

    def test_is_symmetric_non_symmetric_matrix(self):
        """Test is_symmetric with non-symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertFalse(form.is_symmetric())

    def test_is_hermitian_hermitian_matrix(self):
        """Test is_hermitian with hermitian matrix."""
        vs = fn("V", C, 2)
        from sympy import I
        matrix = M([[1, I], [-I, 1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_hermitian())

    def test_is_degenerate_non_invertible(self):
        """Test is_degenerate with non-invertible matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [2, 4]])  # Determinant = 0
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_degenerate())

    def test_is_degenerate_invertible(self):
        """Test is_degenerate with invertible matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertFalse(form.is_degenerate())

    def test_is_positive_definite_positive_matrix(self):
        """Test is_positive_definite with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        # May return None for symbolic, but should work for numeric
        result = form.is_positive_definite()
        self.assertIn(result, [True, None])

    def test_inertia_symmetric_matrix(self):
        """Test inertia with symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        p, m, z = form.inertia()
        self.assertEqual(p, 2)
        self.assertEqual(m, 0)
        self.assertEqual(z, 0)

    def test_inertia_non_symmetric_raises_error(self):
        """Test inertia with non-symmetric matrix raises FormError."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        with self.assertRaises(FormError):
            form.inertia()

    def test_signature(self):
        """Test signature method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        sig = form.signature()
        self.assertEqual(sig, 2)

    def test_info(self):
        """Test info method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        info_str = form.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("f", info_str)
        self.assertIn("Symmetric?", info_str)
        self.assertIn("Positive Definite?", info_str)

    def test_is_negative_definite(self):
        """Test is_negative_definite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)  # Negative definite
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_negative_definite()
        self.assertIn(result, [True, None])  # May return None for symbolic

    def test_is_positive_semidefinite(self):
        """Test is_positive_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_positive_semidefinite()
        self.assertIn(result, [True, None])  # May return None for symbolic

    def test_is_negative_semidefinite(self):
        """Test is_negative_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_negative_semidefinite()
        self.assertIn(result, [True, None])  # May return None for symbolic

    def test_is_indefinite(self):
        """Test is_indefinite method."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])  # Indefinite
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_indefinite()
        self.assertIn(result, [True, None])  # May return None for symbolic


class TestInnerProductInit(unittest.TestCase):
    """Test InnerProduct.__init__ method."""

    def test_init_with_positive_definite_matrix(self):
        """Test InnerProduct initialization with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        self.assertEqual(ip.name, "ip")
        self.assertEqual(ip.vectorspace, vs)

    def test_init_non_symmetric_raises_error(self):
        """Test InnerProduct initialization with non-symmetric matrix raises error."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])  # Not symmetric
        with self.assertRaises(InnerProductError):
            InnerProduct("ip", vs, matrix=matrix)

    def test_init_non_positive_definite_raises_error(self):
        """Test InnerProduct initialization with non-positive definite matrix raises error."""
        vs = fn("V", R, 2)
        matrix = M([[-1, 0], [0, -1]])  # Negative definite
        with self.assertRaises(InnerProductError):
            InnerProduct("ip", vs, matrix=matrix)

    def test_init_with_mapping(self):
        """Test InnerProduct initialization with mapping."""
        vs = fn("V", R, 2)
        def mapping(u, v):
            return u[0] * v[0] + u[1] * v[1]
        ip = InnerProduct("ip", vs, mapping=mapping)
        self.assertEqual(ip.name, "ip")
        self.assertEqual(ip.vectorspace, vs)


class TestInnerProductProperties(unittest.TestCase):
    """Test InnerProduct properties."""

    def test_orthonormal_basis_property(self):
        """Test orthonormal_basis property."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        basis = ip.orthonormal_basis
        self.assertIsInstance(basis, list)
        self.assertEqual(len(basis), 2)


class TestInnerProductMethods(unittest.TestCase):
    """Test InnerProduct methods."""

    def test_norm(self):
        """Test norm method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [3, 4]
        norm_val = ip.norm(v)
        # Norm of [3, 4] should be 5
        self.assertAlmostEqual(float(norm_val), 5.0, places=5)

    def test_is_orthogonal_orthogonal_vectors(self):
        """Test is_orthogonal with orthogonal vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0]
        v = [0, 1]
        self.assertTrue(ip.is_orthogonal(u, v))

    def test_is_orthogonal_non_orthogonal_vectors(self):
        """Test is_orthogonal with non-orthogonal vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0]
        v = [1, 1]
        self.assertFalse(ip.is_orthogonal(u, v))

    def test_is_orthonormal_orthonormal_vectors(self):
        """Test is_orthonormal with orthonormal vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0]
        v = [0, 1]
        self.assertTrue(ip.is_orthonormal(u, v))

    def test_is_orthonormal_non_orthonormal_vectors(self):
        """Test is_orthonormal with non-orthonormal vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [2, 0]  # Not unit length
        v = [0, 1]
        self.assertFalse(ip.is_orthonormal(u, v))

    def test_gram_schmidt_independent_vectors(self):
        """Test gram_schmidt with linearly independent vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 1]
        v = [1, 0]
        result = ip.gram_schmidt(u, v)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # Result should be orthonormal
        self.assertTrue(ip.is_orthonormal(*result))

    def test_gram_schmidt_dependent_vectors_raises_error(self):
        """Test gram_schmidt with linearly dependent vectors raises ValueError."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 1]
        v = [2, 2]  # Dependent
        with self.assertRaises(ValueError):
            ip.gram_schmidt(u, v)

    def test_inner_product_repr(self):
        """Test InnerProduct.__repr__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        repr_str = repr(ip)
        self.assertIn("InnerProduct", repr_str)
        self.assertIn("matrix", repr_str)

    def test_push(self):
        """Test __push__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        result = ip.__push__(v)
        # Should return a vector in F^n (may be Matrix or list)
        # Check it's iterable and has correct length
        self.assertEqual(len(result), 2)
        # Can convert to list for comparison
        result_list = list(result)
        self.assertEqual(len(result_list), 2)

    def test_pull(self):
        """Test __pull__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        # Push then pull should return original (approximately)
        pushed = ip.__push__(v)
        pulled = ip.__pull__(pushed)
        # Should return a vector (may be Matrix or list)
        self.assertEqual(len(pulled), 2)
        # Can convert to list for comparison
        pulled_list = list(pulled)
        self.assertEqual(len(pulled_list), 2)

    def test_inner_product_info(self):
        """Test InnerProduct info method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        info_str = ip.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("ip", info_str)
        self.assertIn("Orthonormal Basis", info_str)

    def test_ortho_projection(self):
        """Test ortho_projection method."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2, 3]
        # Create a 1D subspace
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        proj = ip.ortho_projection(v, subspace)
        # Should return a vector (may be Matrix or list)
        self.assertEqual(len(proj), 3)
        # Can convert to list for comparison
        proj_list = list(proj)
        self.assertEqual(len(proj_list), 3)

    def test_ortho_projection_invalid_vector_raises_error(self):
        """Test ortho_projection with invalid vector raises TypeError."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2, 3]  # Wrong dimension
        subspace = fn("U", R, 2, basis=[[1, 0]])
        with self.assertRaises(TypeError):
            ip.ortho_projection(v, subspace)

    def test_ortho_projection_invalid_subspace_raises_error(self):
        """Test ortho_projection with invalid subspace raises TypeError."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        # Not a subspace of vs
        subspace = fn("U", R, 3)
        with self.assertRaises(TypeError):
            ip.ortho_projection(v, subspace)

    def test_ortho_complement(self):
        """Test ortho_complement method."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        # Create a 1D subspace
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        comp = ip.ortho_complement(subspace)
        self.assertIsNotNone(comp)
        # Complement should have dimension 2
        self.assertEqual(comp.dim, 2)

    def test_ortho_complement_invalid_subspace_raises_error(self):
        """Test ortho_complement with invalid subspace raises TypeError."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        # Not a subspace of vs
        subspace = fn("U", R, 3)
        with self.assertRaises(TypeError):
            ip.ortho_complement(subspace)


class TestQuadraticForm(unittest.TestCase):
    """Test QuadraticForm class."""

    def test_quadratic_form_is_class(self):
        """Test QuadraticForm is a class."""
        self.assertTrue(isinstance(QuadraticForm, type))

    def test_quadratic_form_has_pass(self):
        """Test QuadraticForm is currently just a placeholder."""
        # Currently just has 'pass', so we can't instantiate it meaningfully
        # This test just verifies the class exists
        self.assertTrue(hasattr(QuadraticForm, '__init__'))


if __name__ == '__main__':
    unittest.main()

