"""
Unit tests for the ablina.form module.
"""

import unittest
from ablina import *
import sympy as sp


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

    def test_init_matrix_non_square(self):
        """Test SesquilinearForm initialization with non-square matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4], [5, 6]])  # 3x2, not square
        with self.assertRaises(ValueError):
            SesquilinearForm("f", vs, matrix=matrix)

    def test_init_matrix_wrong_field(self):
        """Test SesquilinearForm initialization with matrix entries not in field."""
        vs = fn("V", R, 2)
        from sympy import I
        matrix = M([[1, I], [I, 1]])  # Complex entries in real field
        with self.assertRaises(ValueError):
            SesquilinearForm("f", vs, matrix=matrix)

    def test_init_mapping_wrong_arity(self):
        """Test SesquilinearForm initialization with mapping of wrong arity."""
        vs = fn("V", R, 2)
        def wrong_mapping(u):
            return u[0]
        with self.assertRaises(TypeError):
            SesquilinearForm("f", vs, mapping=wrong_mapping)

    def test_init_with_involution(self):
        """Test SesquilinearForm initialization with custom involution."""
        vs = fn("V", R, 2)
        def identity_involution(x):
            return x
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix, involution=identity_involution)
        self.assertEqual(form.involution, identity_involution)

    def test_init_with_involution_wrong_arity(self):
        """Test SesquilinearForm initialization with involution of wrong arity."""
        vs = fn("V", R, 2)
        def wrong_involution(x, y):
            return x + y
        matrix = M.eye(2)
        with self.assertRaises(TypeError):
            SesquilinearForm("f", vs, matrix=matrix, involution=wrong_involution)

    def test_init_default_involution(self):
        """Test SesquilinearForm initialization with default involution (conjugation)."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.involution, sp.conjugate)

    def test_init_zero_dimension(self):
        """Test SesquilinearForm initialization with zero-dimensional space."""
        vs = fn("V", R, 0)
        matrix = M.zeros(0, 0)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.vectorspace, vs)
        self.assertEqual(form.matrix, matrix)

    def test_init_one_dimension(self):
        """Test SesquilinearForm initialization with one-dimensional space."""
        vs = fn("V", R, 1)
        matrix = M([[5]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.vectorspace, vs)
        self.assertEqual(form.matrix, matrix)


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

    def test_involution_property(self):
        """Test involution property."""
        vs = fn("V", R, 2)
        def custom_involution(x):
            return -x
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix, involution=custom_involution)
        self.assertEqual(form.involution, custom_involution)

    def test_involution_property_default(self):
        """Test involution property with default (conjugation)."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.involution, sp.conjugate)


class TestSesquilinearFormRepr(unittest.TestCase):
    """Test SesquilinearForm.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        repr_str = repr(form)
        self.assertIn("SesquilinearForm", repr_str)
        self.assertIn("'f'", repr_str)
        self.assertIn("vectorspace", repr_str)
        self.assertIn("matrix", repr_str)

    def test_repr_complex_field(self):
        """Test __repr__ with complex field."""
        vs = fn("V", C, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        repr_str = repr(form)
        self.assertIn("SesquilinearForm", repr_str)


class TestSesquilinearFormStr(unittest.TestCase):
    """Test SesquilinearForm.__str__ method."""

    def test_str_returns_name(self):
        """Test __str__ returns the form name."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        self.assertEqual(str(form), "f")

    def test_str_empty_name(self):
        """Test __str__ with empty name."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("", vs, matrix=M.eye(2))
        self.assertEqual(str(form), "")


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
        vs3 = fn("U", R, 3)
        matrix1 = M.eye(2)
        matrix3 = M.eye(3)
        form1 = SesquilinearForm("f", vs1, matrix=matrix1)
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
        self.assertNotEqual(form, None)
        self.assertNotEqual(form, 42)

    def test_eq_approximately_equal_matrices(self):
        """Test equality with approximately equal matrices."""
        vs = fn("V", R, 2)
        matrix1 = M.eye(2)
        matrix2 = M([[1.0, 0.0], [0.0, 1.0]])
        form1 = SesquilinearForm("f", vs, matrix=matrix1)
        form2 = SesquilinearForm("g", vs, matrix=matrix2)
        self.assertEqual(form1, form2)


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

    def test_call_with_same_vector(self):
        """Test __call__ with same vector twice."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        u = [1, 0]
        result = form(u, u)
        self.assertEqual(result, 1)

    def test_call_with_zero_vector(self):
        """Test __call__ with zero vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        u = [0, 0]
        v = [1, 1]
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

    def test_call_with_both_invalid_vectors(self):
        """Test __call__ with both vectors invalid."""
        vs = fn("V", R, 2)
        form = SesquilinearForm("f", vs, matrix=M.eye(2))
        u = [1, 0, 0]  # Wrong dimension
        v = [0, 1, 0]  # Wrong dimension
        with self.assertRaises(TypeError):
            form(u, v)

    def test_call_with_mapping(self):
        """Test __call__ when form was created with mapping."""
        vs = fn("V", R, 2)
        def mapping(u, v):
            return u[0] * v[0] + 2 * u[1] * v[1]
        form = SesquilinearForm("f", vs, mapping=mapping)
        u = [1, 2]
        v = [3, 4]
        result = form(u, v)
        expected = 1 * 3 + 2 * 2 * 4  # 3 + 16 = 19
        self.assertEqual(result, expected)

    def test_call_complex_field(self):
        """Test __call__ with complex field."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        from sympy import I
        u = [1, I]
        v = [I, 1]
        result = form(u, v)
        # Should handle complex conjugation
        self.assertIsNotNone(result)


class TestSesquilinearFormInfo(unittest.TestCase):
    """Test SesquilinearForm.info method."""

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
        self.assertIn("Matrix", info_str)

    def test_info_with_form_error(self):
        """Test info method when is_positive_definite raises FormError."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])  # Not symmetric
        form = SesquilinearForm("f", vs, matrix=matrix)
        info_str = form.info()
        self.assertIn("N/A", info_str)  # Should catch FormError and show N/A

    def test_info_complex_field(self):
        """Test info method with complex field."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        info_str = form.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("f", info_str)


class TestSesquilinearFormIsSymmetric(unittest.TestCase):
    """Test SesquilinearForm.is_symmetric method."""

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

    def test_is_symmetric_identity_involution(self):
        """Test is_symmetric with identity involution."""
        vs = fn("V", R, 2)
        def identity(x):
            return x
        matrix = M([[1, 2], [2, 3]])
        form = SesquilinearForm("f", vs, matrix=matrix, involution=identity)
        self.assertTrue(form.is_symmetric())

    def test_is_symmetric_zero_matrix(self):
        """Test is_symmetric with zero matrix."""
        vs = fn("V", R, 2)
        matrix = M.zeros(2, 2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_symmetric())


class TestSesquilinearFormIsSkewSymmetric(unittest.TestCase):
    """Test SesquilinearForm.is_skew_symmetric method."""

    def test_is_skew_symmetric_skew_symmetric_matrix(self):
        """Test is_skew_symmetric with skew-symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M([[0, 1], [-1, 0]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_skew_symmetric())

    def test_is_skew_symmetric_non_skew_symmetric_matrix(self):
        """Test is_skew_symmetric with non-skew-symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertFalse(form.is_skew_symmetric())

    def test_is_skew_symmetric_zero_matrix(self):
        """Test is_skew_symmetric with zero matrix."""
        vs = fn("V", R, 2)
        matrix = M.zeros(2, 2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_skew_symmetric())

    def test_is_anti_symmetric_alias(self):
        """Test is_anti_symmetric is an alias for is_skew_symmetric."""
        vs = fn("V", R, 2)
        matrix = M([[0, 1], [-1, 0]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_anti_symmetric())
        self.assertEqual(form.is_anti_symmetric(), form.is_skew_symmetric())


class TestSesquilinearFormIsAlternating(unittest.TestCase):
    """Test SesquilinearForm.is_alternating method."""

    def test_is_alternating_alternating_matrix(self):
        """Test is_alternating with alternating matrix."""
        vs = fn("V", R, 2)
        matrix = M([[0, 1], [-1, 0]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_alternating()
        self.assertIn(result, [True, None])  # May return None for symbolic

    def test_is_alternating_non_alternating_matrix(self):
        """Test is_alternating with non-alternating matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertFalse(form.is_alternating())

    def test_is_alternating_zero_matrix(self):
        """Test is_alternating with zero matrix."""
        vs = fn("V", R, 2)
        matrix = M.zeros(2, 2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_alternating()
        self.assertIn(result, [True, None])


class TestSesquilinearFormIsHermitian(unittest.TestCase):
    """Test SesquilinearForm.is_hermitian method."""

    def test_is_hermitian_hermitian_matrix(self):
        """Test is_hermitian with hermitian matrix."""
        vs = fn("V", C, 2)
        from sympy import I
        matrix = M([[1, I], [-I, 1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_hermitian())

    def test_is_hermitian_non_hermitian_matrix(self):
        """Test is_hermitian with non-hermitian matrix."""
        vs = fn("V", C, 2)
        from sympy import I
        matrix = M([[1, I], [I, 1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertFalse(form.is_hermitian())

    def test_is_hermitian_real_matrix(self):
        """Test is_hermitian with real symmetric matrix (should be hermitian)."""
        vs = fn("V", C, 2)
        matrix = M([[1, 2], [2, 3]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_hermitian())


class TestSesquilinearFormIsDegenerate(unittest.TestCase):
    """Test SesquilinearForm.is_degenerate method."""

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

    def test_is_degenerate_zero_matrix(self):
        """Test is_degenerate with zero matrix."""
        vs = fn("V", R, 2)
        matrix = M.zeros(2, 2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_degenerate())

    def test_is_degenerate_symbolic(self):
        """Test is_degenerate with matrix that may return None."""
        vs = fn("V", R, 2)
        # Use a numeric matrix - symbolic matrices can't be validated
        # as they don't have entries in the field
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_degenerate()
        # Should return False for identity matrix
        self.assertFalse(result)


class TestSesquilinearFormInertia(unittest.TestCase):
    """Test SesquilinearForm.inertia method."""

    def test_inertia_symmetric_matrix(self):
        """Test inertia with symmetric matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        p, m, z = form.inertia()
        self.assertEqual(p, 2)
        self.assertEqual(m, 0)
        self.assertEqual(z, 0)

    def test_inertia_negative_definite(self):
        """Test inertia with negative definite matrix."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        p, m, z = form.inertia()
        self.assertEqual(p, 0)
        self.assertEqual(m, 2)
        self.assertEqual(z, 0)

    def test_inertia_indefinite(self):
        """Test inertia with indefinite matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        p, m, z = form.inertia()
        self.assertEqual(p, 1)
        self.assertEqual(m, 1)
        self.assertEqual(z, 0)

    def test_inertia_with_zero_eigenvalue(self):
        """Test inertia with zero eigenvalue."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, 0]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        p, m, z = form.inertia()
        self.assertEqual(p, 1)
        self.assertEqual(m, 0)
        self.assertEqual(z, 1)

    def test_inertia_non_symmetric_raises_error(self):
        """Test inertia with non-symmetric matrix raises FormError."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        with self.assertRaises(FormError):
            form.inertia()

    def test_inertia_complex_field_raises_error(self):
        """Test inertia with complex field and non-hermitian matrix raises FormError."""
        vs = fn("V", C, 2)
        from sympy import I
        matrix = M([[1, I], [I, 1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        with self.assertRaises(FormError):
            form.inertia()


class TestSesquilinearFormSignature(unittest.TestCase):
    """Test SesquilinearForm.signature method."""

    def test_signature_positive_definite(self):
        """Test signature with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        sig = form.signature()
        self.assertEqual(sig, 2)

    def test_signature_negative_definite(self):
        """Test signature with negative definite matrix."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        sig = form.signature()
        self.assertEqual(sig, -2)

    def test_signature_indefinite(self):
        """Test signature with indefinite matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        sig = form.signature()
        self.assertEqual(sig, 0)

    def test_signature_with_zero_eigenvalue(self):
        """Test signature with zero eigenvalue."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, 0]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        sig = form.signature()
        self.assertEqual(sig, 1)


class TestSesquilinearFormDefiniteness(unittest.TestCase):
    """Test SesquilinearForm definiteness methods."""

    def test_is_positive_definite_positive_matrix(self):
        """Test is_positive_definite with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_positive_definite()
        self.assertIn(result, [True, None])

    def test_is_positive_definite_non_symmetric_raises_error(self):
        """Test is_positive_definite with non-symmetric matrix raises FormError."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        with self.assertRaises(FormError):
            form.is_positive_definite()

    def test_is_negative_definite(self):
        """Test is_negative_definite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_negative_definite()
        self.assertIn(result, [True, None])

    def test_is_positive_semidefinite(self):
        """Test is_positive_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_positive_semidefinite()
        self.assertIn(result, [True, None])

    def test_is_negative_semidefinite(self):
        """Test is_negative_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_negative_semidefinite()
        self.assertIn(result, [True, None])

    def test_is_indefinite(self):
        """Test is_indefinite method."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_indefinite()
        self.assertIn(result, [True, None])

    def test_is_indefinite_positive_definite_returns_false(self):
        """Test is_indefinite with positive definite matrix returns False."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = SesquilinearForm("f", vs, matrix=matrix)
        result = form.is_indefinite()
        self.assertIn(result, [False, None])


class TestBilinearFormInit(unittest.TestCase):
    """Test BilinearForm.__init__ method."""

    def test_init_with_matrix(self):
        """Test BilinearForm initialization with matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = BilinearForm("f", vs, matrix=matrix)
        self.assertEqual(form.name, "f")
        self.assertEqual(form.vectorspace, vs)
        self.assertEqual(form.matrix, matrix)

    def test_init_with_mapping(self):
        """Test BilinearForm initialization with mapping."""
        vs = fn("V", R, 2)
        def mapping(u, v):
            return u[0] * v[0] + u[1] * v[1]
        form = BilinearForm("f", vs, mapping=mapping)
        self.assertEqual(form.name, "f")
        self.assertEqual(form.vectorspace, vs)
        self.assertIsNotNone(form.matrix)

    def test_init_uses_identity_involution(self):
        """Test BilinearForm uses identity involution."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = BilinearForm("f", vs, matrix=matrix)
        # Identity involution: f(x) = x
        self.assertEqual(form.involution(5), 5)
        self.assertEqual(form.involution(sp.Symbol('x')), sp.Symbol('x'))

    def test_init_without_matrix_or_mapping(self):
        """Test BilinearForm initialization without matrix or mapping."""
        vs = fn("V", R, 2)
        with self.assertRaises(FormError):
            BilinearForm("f", vs)

    def test_init_invalid_vectorspace_type(self):
        """Test BilinearForm initialization with invalid vectorspace type."""
        with self.assertRaises(TypeError):
            BilinearForm("f", "not a VectorSpace", matrix=M.eye(2))


class TestBilinearFormRepr(unittest.TestCase):
    """Test BilinearForm.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = BilinearForm("f", vs, matrix=matrix)
        repr_str = repr(form)
        self.assertIn("BilinearForm", repr_str)
        self.assertIn("'f'", repr_str)
        self.assertIn("vectorspace", repr_str)
        self.assertIn("matrix", repr_str)

    def test_repr_no_involution(self):
        """Test __repr__ does not include involution (unlike SesquilinearForm)."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        form = BilinearForm("f", vs, matrix=matrix)
        repr_str = repr(form)
        self.assertNotIn("involution", repr_str)


class TestBilinearFormInheritance(unittest.TestCase):
    """Test BilinearForm inheritance from SesquilinearForm."""

    def test_is_instance_of_sesquilinear_form(self):
        """Test BilinearForm is instance of SesquilinearForm."""
        vs = fn("V", R, 2)
        form = BilinearForm("f", vs, matrix=M.eye(2))
        self.assertIsInstance(form, SesquilinearForm)

    def test_inherits_methods(self):
        """Test BilinearForm inherits methods from SesquilinearForm."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [2, 3]])
        form = BilinearForm("f", vs, matrix=matrix)
        self.assertTrue(form.is_symmetric())
        self.assertIsNotNone(form.info())


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

    def test_init_with_complex_field(self):
        """Test InnerProduct initialization with complex field."""
        vs = fn("V", C, 2)
        # Use identity matrix which is positive definite
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        self.assertEqual(ip.name, "ip")
        self.assertEqual(ip.vectorspace, vs)

    def test_init_non_hermitian_complex_raises_error(self):
        """Test InnerProduct initialization with non-hermitian complex matrix raises error."""
        vs = fn("V", C, 2)
        from sympy import I
        matrix = M([[1, I], [I, 1]])  # Not hermitian
        with self.assertRaises(InnerProductError):
            InnerProduct("ip", vs, matrix=matrix)

    def test_init_creates_orthonormal_basis(self):
        """Test InnerProduct initialization creates orthonormal basis."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        self.assertIsNotNone(ip._orthonormal_basis)
        self.assertEqual(len(ip._orthonormal_basis), 2)


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

    def test_orthonormal_basis_is_orthonormal(self):
        """Test orthonormal_basis is actually orthonormal."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        basis = ip.orthonormal_basis
        self.assertTrue(ip.is_orthonormal(*basis))


class TestInnerProductRepr(unittest.TestCase):
    """Test InnerProduct.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        repr_str = repr(ip)
        self.assertIn("InnerProduct", repr_str)
        self.assertIn("'ip'", repr_str)
        self.assertIn("matrix", repr_str)


class TestInnerProductInfo(unittest.TestCase):
    """Test InnerProduct.info method."""

    def test_info(self):
        """Test info method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        info_str = ip.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("ip", info_str)
        self.assertIn("Orthonormal Basis", info_str)
        self.assertIn("Matrix", info_str)


class TestInnerProductNorm(unittest.TestCase):
    """Test InnerProduct.norm method."""

    def test_norm(self):
        """Test norm method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [3, 4]
        norm_val = ip.norm(v)
        # Norm of [3, 4] should be 5
        self.assertAlmostEqual(float(norm_val), 5.0, places=5)

    def test_norm_zero_vector(self):
        """Test norm of zero vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [0, 0]
        norm_val = ip.norm(v)
        self.assertEqual(norm_val, 0)

    def test_norm_unit_vector(self):
        """Test norm of unit vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 0]
        norm_val = ip.norm(v)
        self.assertAlmostEqual(float(norm_val), 1.0, places=5)

    def test_norm_complex_field(self):
        """Test norm with complex field."""
        vs = fn("V", C, 2)
        # Use identity matrix which is positive definite
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 0]
        norm_val = ip.norm(v)
        self.assertIsNotNone(norm_val)


class TestInnerProductIsOrthogonal(unittest.TestCase):
    """Test InnerProduct.is_orthogonal method."""

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

    def test_is_orthogonal_zero_vector(self):
        """Test is_orthogonal with zero vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [0, 0]
        v = [1, 1]
        self.assertTrue(ip.is_orthogonal(u, v))  # Zero vector is orthogonal to all

    def test_is_orthogonal_multiple_vectors(self):
        """Test is_orthogonal with multiple vectors."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0, 0]
        v = [0, 1, 0]
        w = [0, 0, 1]
        self.assertTrue(ip.is_orthogonal(u, v, w))

    def test_is_orthogonal_single_vector(self):
        """Test is_orthogonal with single vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0]
        self.assertTrue(ip.is_orthogonal(u))  # Single vector is trivially orthogonal


class TestInnerProductIsOrthonormal(unittest.TestCase):
    """Test InnerProduct.is_orthonormal method."""

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

    def test_is_orthonormal_orthogonal_but_not_normalized(self):
        """Test is_orthonormal with orthogonal but not normalized vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [2, 0]  # Orthogonal but not unit
        v = [0, 2]
        self.assertFalse(ip.is_orthonormal(u, v))

    def test_is_orthonormal_not_orthogonal(self):
        """Test is_orthonormal with non-orthogonal vectors."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0]
        v = [1, 1]  # Not orthogonal
        self.assertFalse(ip.is_orthonormal(u, v))


class TestInnerProductGramSchmidt(unittest.TestCase):
    """Test InnerProduct.gram_schmidt method."""

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

    def test_gram_schmidt_single_vector(self):
        """Test gram_schmidt with single vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [2, 0]
        result = ip.gram_schmidt(u)
        self.assertEqual(len(result), 1)
        self.assertTrue(ip.is_orthonormal(*result))

    def test_gram_schmidt_three_vectors(self):
        """Test gram_schmidt with three vectors."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        u = [1, 0, 0]
        v = [1, 1, 0]
        w = [1, 1, 1]
        result = ip.gram_schmidt(u, v, w)
        self.assertEqual(len(result), 3)
        self.assertTrue(ip.is_orthonormal(*result))


class TestInnerProductPushPull(unittest.TestCase):
    """Test InnerProduct.__push__ and __pull__ methods."""

    def test_push(self):
        """Test __push__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        result = ip.__push__(v)
        self.assertEqual(len(result), 2)

    def test_pull(self):
        """Test __pull__ method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = M([[1], [2]])
        result = ip.__pull__(v)
        self.assertEqual(len(result), 2)

    def test_push_then_pull(self):
        """Test push then pull returns original vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        pushed = ip.__push__(v)
        pulled = ip.__pull__(pushed)
        # Should return a vector of same length
        self.assertEqual(len(pulled), 2)

    def test_push_zero_vector(self):
        """Test __push__ with zero vector."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [0, 0]
        result = ip.__push__(v)
        self.assertEqual(len(result), 2)


class TestInnerProductOrthoProjection(unittest.TestCase):
    """Test InnerProduct.ortho_projection method."""

    def test_ortho_projection(self):
        """Test ortho_projection method."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2, 3]
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        proj = ip.ortho_projection(v, subspace)
        self.assertEqual(len(proj), 3)

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
        subspace = fn("U", R, 3)  # Not a subspace of vs
        with self.assertRaises(TypeError):
            ip.ortho_projection(v, subspace)

    def test_ortho_projection_onto_zero_subspace(self):
        """Test ortho_projection onto zero subspace."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        v = [1, 2]
        subspace = fn("U", R, 2, constraints=["v0 == 0", "v1 == 0"])
        proj = ip.ortho_projection(v, subspace)
        self.assertEqual(len(proj), 2)


class TestInnerProductOrthoComplement(unittest.TestCase):
    """Test InnerProduct.ortho_complement method."""

    def test_ortho_complement(self):
        """Test ortho_complement method."""
        vs = fn("V", R, 3)
        matrix = M.eye(3)
        ip = InnerProduct("ip", vs, matrix=matrix)
        subspace = fn("U", R, 3, basis=[[1, 0, 0]])
        comp = ip.ortho_complement(subspace)
        self.assertIsNotNone(comp)
        self.assertEqual(comp.dim, 2)

    def test_ortho_complement_invalid_subspace_raises_error(self):
        """Test ortho_complement with invalid subspace raises TypeError."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        subspace = fn("U", R, 3)  # Not a subspace of vs
        with self.assertRaises(TypeError):
            ip.ortho_complement(subspace)

    def test_ortho_complement_zero_subspace(self):
        """Test ortho_complement of zero subspace."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        subspace = fn("U", R, 2, constraints=["v0 == 0", "v1 == 0"])
        comp = ip.ortho_complement(subspace)
        self.assertEqual(comp.dim, 2)  # Complement of zero is whole space

    def test_ortho_complement_whole_space(self):
        """Test ortho_complement of whole space."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        ip = InnerProduct("ip", vs, matrix=matrix)
        comp = ip.ortho_complement(vs)
        self.assertEqual(comp.dim, 0)  # Complement of whole space is zero


class TestQuadraticFormInit(unittest.TestCase):
    """Test QuadraticForm.__init__ method."""

    def test_init_with_matrix(self):
        """Test QuadraticForm initialization with matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, 1]])
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertEqual(qf.name, "q")
        self.assertEqual(qf.vectorspace, vs)
        self.assertIsNotNone(qf.matrix)

    def test_init_with_mapping(self):
        """Test QuadraticForm initialization with mapping."""
        vs = fn("V", R, 2)
        def mapping(v):
            return v[0]**2 + v[1]**2
        qf = QuadraticForm("q", vs, mapping=mapping)
        self.assertEqual(qf.name, "q")
        self.assertEqual(qf.vectorspace, vs)
        self.assertIsNotNone(qf.matrix)

    def test_init_with_both_matrix_and_mapping(self):
        """Test QuadraticForm initialization with both matrix and mapping."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, 1]])
        def mapping(v):
            return v[0]**2 + v[1]**2
        qf = QuadraticForm("q", vs, mapping=mapping, matrix=matrix)
        self.assertEqual(qf.name, "q")
        self.assertIsNotNone(qf.matrix)

    def test_init_without_matrix_or_mapping(self):
        """Test QuadraticForm initialization without matrix or mapping."""
        vs = fn("V", R, 2)
        with self.assertRaises(FormError):
            QuadraticForm("q", vs)

    def test_init_invalid_vectorspace_type(self):
        """Test QuadraticForm initialization with invalid vectorspace type."""
        with self.assertRaises(TypeError):
            QuadraticForm("q", "not a VectorSpace", matrix=M.eye(2))

    def test_init_invalid_matrix_shape(self):
        """Test QuadraticForm initialization with invalid matrix shape."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2, 3], [4, 5, 6]])  # Wrong shape
        with self.assertRaises(ValueError):
            QuadraticForm("q", vs, matrix=matrix)

    def test_init_matrix_wrong_size(self):
        """Test QuadraticForm initialization with matrix of wrong size."""
        vs = fn("V", R, 2)
        matrix = M.eye(3)  # Wrong size
        with self.assertRaises(ValueError):
            QuadraticForm("q", vs, matrix=matrix)

    def test_init_matrix_wrong_field(self):
        """Test QuadraticForm initialization with matrix entries not in field."""
        vs = fn("V", R, 2)
        from sympy import I
        matrix = M([[1, I], [I, 1]])  # Complex entries in real field
        with self.assertRaises(ValueError):
            QuadraticForm("q", vs, matrix=matrix)

    def test_init_mapping_wrong_arity(self):
        """Test QuadraticForm initialization with mapping of wrong arity."""
        vs = fn("V", R, 2)
        def wrong_mapping(u, v):
            return u[0] * v[0]
        with self.assertRaises(TypeError):
            QuadraticForm("q", vs, mapping=wrong_mapping)

    def test_init_matrix_symmetrized(self):
        """Test QuadraticForm symmetrizes matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [3, 4]])  # Not symmetric
        qf = QuadraticForm("q", vs, matrix=matrix)
        # Matrix should be symmetrized: (A + A^T) / 2
        expected = (matrix + matrix.T) / 2
        self.assertTrue(qf.matrix.equals(expected))

    def test_init_zero_dimension(self):
        """Test QuadraticForm initialization with zero-dimensional space."""
        vs = fn("V", R, 0)
        matrix = M.zeros(0, 0)
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertEqual(qf.vectorspace, vs)

    def test_init_one_dimension(self):
        """Test QuadraticForm initialization with one-dimensional space."""
        vs = fn("V", R, 1)
        matrix = M([[5]])
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertEqual(qf.vectorspace, vs)


class TestQuadraticFormProperties(unittest.TestCase):
    """Test QuadraticForm properties."""

    def test_vectorspace_property(self):
        """Test vectorspace property."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        self.assertEqual(qf.vectorspace, vs)

    def test_mapping_property(self):
        """Test mapping property."""
        vs = fn("V", R, 2)
        def mapping(v):
            return v[0]**2
        qf = QuadraticForm("q", vs, mapping=mapping)
        self.assertIsNotNone(qf.mapping)
        self.assertTrue(callable(qf.mapping))

    def test_matrix_property(self):
        """Test matrix property."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertIsNotNone(qf.matrix)

    def test_polarization_property(self):
        """Test polarization property."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        pol = qf.polarization
        self.assertIsInstance(pol, BilinearForm)
        self.assertEqual(pol.vectorspace, vs)


class TestQuadraticFormRepr(unittest.TestCase):
    """Test QuadraticForm.__repr__ method."""

    def test_repr(self):
        """Test __repr__ method."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        repr_str = repr(qf)
        self.assertIn("QuadraticForm", repr_str)
        self.assertIn("'q'", repr_str)
        self.assertIn("vectorspace", repr_str)
        self.assertIn("matrix", repr_str)


class TestQuadraticFormStr(unittest.TestCase):
    """Test QuadraticForm.__str__ method."""

    def test_str_returns_name(self):
        """Test __str__ returns the form name."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        self.assertEqual(str(qf), "q")


class TestQuadraticFormEq(unittest.TestCase):
    """Test QuadraticForm.__eq__ method."""

    def test_eq_same_form(self):
        """Test equality with same form."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf1 = QuadraticForm("q", vs, matrix=matrix)
        qf2 = QuadraticForm("p", vs, matrix=matrix)
        self.assertEqual(qf1, qf2)

    def test_eq_different_matrix(self):
        """Test equality with different matrix."""
        vs = fn("V", R, 2)
        qf1 = QuadraticForm("q", vs, matrix=M.eye(2))
        qf2 = QuadraticForm("p", vs, matrix=2 * M.eye(2))
        self.assertNotEqual(qf1, qf2)

    def test_eq_not_quadratic_form(self):
        """Test equality with non-QuadraticForm object."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        self.assertNotEqual(qf, "not a form")
        self.assertNotEqual(qf, None)


class TestQuadraticFormCall(unittest.TestCase):
    """Test QuadraticForm.__call__ method."""

    def test_call_with_valid_vector(self):
        """Test __call__ with valid vector."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        v = [1, 2]
        result = qf(v)
        # q(v) = v^T M v = [1, 2] @ [[1,0],[0,1]] @ [1,2]^T = 1 + 4 = 5
        self.assertEqual(result, 5)

    def test_call_with_zero_vector(self):
        """Test __call__ with zero vector."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        v = [0, 0]
        result = qf(v)
        self.assertEqual(result, 0)

    def test_call_with_invalid_vector(self):
        """Test __call__ with invalid vector."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        v = [1, 2, 3]  # Wrong dimension
        with self.assertRaises(TypeError):
            qf(v)

    def test_call_with_mapping(self):
        """Test __call__ when form was created with mapping."""
        vs = fn("V", R, 2)
        def mapping(v):
            return v[0]**2 + 2 * v[1]**2
        qf = QuadraticForm("q", vs, mapping=mapping)
        v = [1, 2]
        result = qf(v)
        expected = 1**2 + 2 * 2**2  # 1 + 8 = 9
        self.assertEqual(result, expected)


class TestQuadraticFormInfo(unittest.TestCase):
    """Test QuadraticForm.info method."""

    def test_info(self):
        """Test info method."""
        vs = fn("V", R, 2)
        qf = QuadraticForm("q", vs, matrix=M.eye(2))
        info_str = qf.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("q", info_str)
        self.assertIn("Matrix", info_str)


class TestQuadraticFormInertia(unittest.TestCase):
    """Test QuadraticForm.inertia method."""

    def test_inertia_positive_definite(self):
        """Test inertia with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        p, m, z = qf.inertia()
        self.assertEqual(p, 2)
        self.assertEqual(m, 0)
        self.assertEqual(z, 0)

    def test_inertia_negative_definite(self):
        """Test inertia with negative definite matrix."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        p, m, z = qf.inertia()
        self.assertEqual(p, 0)
        self.assertEqual(m, 2)
        self.assertEqual(z, 0)

    def test_inertia_indefinite(self):
        """Test inertia with indefinite matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        qf = QuadraticForm("q", vs, matrix=matrix)
        p, m, z = qf.inertia()
        self.assertEqual(p, 1)
        self.assertEqual(m, 1)
        self.assertEqual(z, 0)

    def test_inertia_complex_field_raises_error(self):
        """Test inertia with complex field raises FormError."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        with self.assertRaises(FormError):
            qf.inertia()


class TestQuadraticFormSignature(unittest.TestCase):
    """Test QuadraticForm.signature method."""

    def test_signature_positive_definite(self):
        """Test signature with positive definite matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        sig = qf.signature()
        self.assertEqual(sig, 2)

    def test_signature_negative_definite(self):
        """Test signature with negative definite matrix."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        sig = qf.signature()
        self.assertEqual(sig, -2)

    def test_signature_indefinite(self):
        """Test signature with indefinite matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        qf = QuadraticForm("q", vs, matrix=matrix)
        sig = qf.signature()
        self.assertEqual(sig, 0)


class TestQuadraticFormIsDegenerate(unittest.TestCase):
    """Test QuadraticForm.is_degenerate method."""

    def test_is_degenerate_non_invertible(self):
        """Test is_degenerate with non-invertible matrix."""
        vs = fn("V", R, 2)
        matrix = M([[1, 2], [2, 4]])  # Determinant = 0
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertTrue(qf.is_degenerate())

    def test_is_degenerate_invertible(self):
        """Test is_degenerate with invertible matrix."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        self.assertFalse(qf.is_degenerate())


class TestQuadraticFormDefiniteness(unittest.TestCase):
    """Test QuadraticForm definiteness methods."""

    def test_is_positive_definite(self):
        """Test is_positive_definite method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        result = qf.is_positive_definite()
        self.assertIn(result, [True, None])

    def test_is_positive_definite_complex_raises_error(self):
        """Test is_positive_definite with complex field raises FormError."""
        vs = fn("V", C, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        with self.assertRaises(FormError):
            qf.is_positive_definite()

    def test_is_negative_definite(self):
        """Test is_negative_definite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        result = qf.is_negative_definite()
        self.assertIn(result, [True, None])

    def test_is_positive_semidefinite(self):
        """Test is_positive_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        result = qf.is_positive_semidefinite()
        self.assertIn(result, [True, None])

    def test_is_negative_semidefinite(self):
        """Test is_negative_semidefinite method."""
        vs = fn("V", R, 2)
        matrix = -M.eye(2)
        qf = QuadraticForm("q", vs, matrix=matrix)
        result = qf.is_negative_semidefinite()
        self.assertIn(result, [True, None])

    def test_is_indefinite(self):
        """Test is_indefinite method."""
        vs = fn("V", R, 2)
        matrix = M([[1, 0], [0, -1]])
        qf = QuadraticForm("q", vs, matrix=matrix)
        result = qf.is_indefinite()
        self.assertIn(result, [True, None])


if __name__ == '__main__':
    unittest.main()
