"""
Unit tests for the ablina.mathset module.
"""

import unittest
from ablina import *


class TestSetInit(unittest.TestCase):
    """Test Set.__init__ method."""

    def test_set_init_basic(self):
        """Test Set initialization with basic parameters."""
        s = Set("A", int, lambda x: x > 0)
        self.assertEqual(s.name, "A")
        self.assertEqual(s.cls, int)
        self.assertEqual(len(s.predicates), 1)

    def test_set_init_multiple_predicates(self):
        """Test Set initialization with multiple predicates."""
        s = Set("B", list, lambda x: len(x) > 0, lambda x: len(x) < 10)
        self.assertEqual(s.name, "B")
        self.assertEqual(s.cls, list)
        self.assertEqual(len(s.predicates), 2)

    def test_set_init_list_of_predicates(self):
        """Test Set initialization with list of predicates."""
        preds = [lambda x: x > 0, lambda x: x < 100]
        s = Set("C", int, preds)
        self.assertEqual(s.name, "C")
        self.assertEqual(s.cls, int)
        self.assertEqual(len(s.predicates), 2)

    def test_set_init_no_predicates(self):
        """Test Set initialization with no predicates."""
        s = Set("D", str)
        self.assertEqual(s.name, "D")
        self.assertEqual(s.cls, str)
        self.assertEqual(len(s.predicates), 0)

    def test_set_init_invalid_cls_type(self):
        """Test Set initialization with invalid cls type."""
        with self.assertRaises(TypeError):
            Set("E", "not a type")

    def test_set_init_non_callable_predicate(self):
        """Test Set initialization with non-callable predicate."""
        with self.assertRaises(TypeError):
            Set("F", int, "not callable")

    def test_set_init_predicate_wrong_arity(self):
        """Test Set initialization with predicate of wrong arity."""
        def two_args(x, y):
            return x + y
        with self.assertRaises(ValueError):
            Set("G", int, two_args)

    def test_set_init_duplicate_predicates_removed(self):
        """Test that duplicate predicates are removed."""
        pred = lambda x: x > 0
        s = Set("H", int, pred, pred, pred)
        self.assertEqual(len(s.predicates), 1)


class TestSetProperties(unittest.TestCase):
    """Test Set properties."""

    def test_cls_property(self):
        """Test cls property."""
        s = Set("A", float)
        self.assertEqual(s.cls, float)

    def test_predicates_property(self):
        """Test predicates property."""
        pred1 = lambda x: x > 0
        pred2 = lambda x: x < 100
        s = Set("B", int, pred1, pred2)
        self.assertEqual(len(s.predicates), 2)
        self.assertIn(pred1, s.predicates)
        self.assertIn(pred2, s.predicates)


class TestSetRepr(unittest.TestCase):
    """Test Set.__repr__ method."""

    def test_repr_basic(self):
        """Test __repr__ with basic set."""
        s = Set("A", int, lambda x: x > 0)
        repr_str = repr(s)
        self.assertIn("Set", repr_str)
        self.assertIn("'A'", repr_str)
        self.assertIn("int", repr_str)

    def test_repr_no_predicates(self):
        """Test __repr__ with no predicates."""
        s = Set("B", str)
        repr_str = repr(s)
        self.assertIn("Set", repr_str)
        self.assertIn("'B'", repr_str)
        self.assertIn("str", repr_str)


class TestSetStr(unittest.TestCase):
    """Test Set.__str__ method."""

    def test_str_returns_name(self):
        """Test __str__ returns the set name."""
        s = Set("MySet", int)
        self.assertEqual(str(s), "MySet")


class TestSetEq(unittest.TestCase):
    """Test Set.__eq__ method."""

    def test_eq_same_name(self):
        """Test equality with same name."""
        s1 = Set("A", int, lambda x: x > 0)
        s2 = Set("A", str, lambda x: len(x) > 0)
        self.assertEqual(s1, s2)

    def test_eq_different_name(self):
        """Test equality with different name."""
        s1 = Set("A", int)
        s2 = Set("B", int)
        self.assertNotEqual(s1, s2)

    def test_eq_not_set(self):
        """Test equality with non-Set object."""
        s = Set("A", int)
        self.assertNotEqual(s, "not a set")
        self.assertNotEqual(s, 42)


class TestSetContains(unittest.TestCase):
    """Test Set.__contains__ method."""

    def test_contains_correct_type_and_predicate(self):
        """Test __contains__ with correct type and satisfying predicate."""
        s = Set("A", int, lambda x: x > 0)
        self.assertIn(5, s)
        self.assertIn(100, s)

    def test_contains_correct_type_fails_predicate(self):
        """Test __contains__ with correct type but failing predicate."""
        s = Set("B", int, lambda x: x > 0)
        self.assertNotIn(0, s)
        self.assertNotIn(-5, s)

    def test_contains_wrong_type(self):
        """Test __contains__ with wrong type."""
        s = Set("C", int, lambda x: x > 0)
        self.assertNotIn("5", s)
        self.assertNotIn(5.0, s)
        self.assertNotIn([5], s)

    def test_contains_multiple_predicates(self):
        """Test __contains__ with multiple predicates."""
        s = Set("D", list, lambda x: len(x) > 0, lambda x: len(x) < 5)
        self.assertIn([1], s)
        self.assertIn([1, 2, 3], s)
        self.assertNotIn([], s)  # Fails first predicate
        self.assertNotIn([1, 2, 3, 4, 5], s)  # Fails second predicate

    def test_contains_no_predicates(self):
        """Test __contains__ with no predicates."""
        s = Set("E", int)
        self.assertIn(0, s)
        self.assertIn(5, s)
        self.assertIn(-10, s)
        self.assertNotIn("5", s)  # Wrong type


class TestSetPos(unittest.TestCase):
    """Test Set.__pos__ method."""

    def test_pos_returns_self(self):
        """Test __pos__ returns the same set."""
        s = Set("A", int)
        self.assertIs(+s, s)


class TestSetNeg(unittest.TestCase):
    """Test Set.__neg__ method."""

    def test_neg_returns_complement(self):
        """Test __neg__ returns complement."""
        s = Set("A", int, lambda x: x > 0)
        complement = -s
        self.assertIsInstance(complement, Set)
        self.assertNotIn(5, complement)
        self.assertIn(0, complement)
        self.assertIn(-5, complement)


class TestSetAnd(unittest.TestCase):
    """Test Set.__and__ method."""

    def test_and_returns_intersection(self):
        """Test __and__ returns intersection."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        intersection = s1 & s2
        self.assertIsInstance(intersection, Set)
        self.assertIn([1, 2, 3], intersection)
        self.assertNotIn([2, 3, 4], intersection)
        self.assertNotIn([1, 2], intersection)


class TestSetOr(unittest.TestCase):
    """Test Set.__or__ method."""

    def test_or_returns_union(self):
        """Test __or__ returns union."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        union = s1 | s2
        self.assertIsInstance(union, Set)
        self.assertIn([1, 2, 3], union)
        self.assertIn([2, 3, 4], union)  # In s1
        self.assertIn([1, 2], union)  # In s2


class TestSetSub(unittest.TestCase):
    """Test Set.__sub__ method."""

    def test_sub_returns_difference(self):
        """Test __sub__ returns difference."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        difference = s1 - s2
        self.assertIsInstance(difference, Set)
        self.assertIn([2, 3, 4], difference)
        self.assertNotIn([1, 2, 3], difference)
        self.assertNotIn([1, 2], difference)


class TestSetComplement(unittest.TestCase):
    """Test Set.complement method."""

    def test_complement_basic(self):
        """Test complement with basic set."""
        s = Set("A", list, lambda x: len(x) == 3)
        complement = s.complement()
        self.assertIsInstance(complement, Set)
        self.assertEqual(complement.name, "A^C")
        self.assertEqual(complement.cls, list)
        self.assertNotIn([1, 2, 3], complement)
        self.assertIn([1, 2], complement)
        self.assertIn([1, 2, 3, 4], complement)
        self.assertNotIn((1, 2, 3), complement)  # Wrong type

    def test_complement_no_predicates(self):
        """Test complement with no predicates."""
        s = Set("B", int)
        complement = s.complement()
        # Complement of universal set is empty set
        self.assertIsInstance(complement, Set)
        self.assertNotIn(0, complement)
        self.assertNotIn(5, complement)

    def test_complement_multiple_predicates(self):
        """Test complement with multiple predicates."""
        s = Set("C", int, lambda x: x > 0, lambda x: x < 10)
        complement = s.complement()
        self.assertNotIn(5, complement)  # Satisfies both
        self.assertIn(0, complement)  # Fails first
        self.assertIn(10, complement)  # Fails second
        self.assertIn(-5, complement)  # Fails first


class TestSetIntersection(unittest.TestCase):
    """Test Set.intersection method."""

    def test_intersection_basic(self):
        """Test intersection with basic sets."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        intersection = s1.intersection(s2)
        self.assertIsInstance(intersection, Set)
        self.assertIn("∩", intersection.name)
        self.assertEqual(intersection.cls, list)
        self.assertIn([1, 2, 3], intersection)
        self.assertNotIn([2, 3, 4], intersection)
        self.assertNotIn([1, 2], intersection)

    def test_intersection_different_cls(self):
        """Test intersection with different cls raises ValueError."""
        s1 = Set("A", int)
        s2 = Set("B", str)
        with self.assertRaises(ValueError):
            s1.intersection(s2)

    def test_intersection_not_set(self):
        """Test intersection with non-Set raises TypeError."""
        s = Set("A", int)
        with self.assertRaises(TypeError):
            s.intersection("not a set")

    def test_intersection_no_predicates(self):
        """Test intersection with sets having no predicates."""
        s1 = Set("A", int)
        s2 = Set("B", int, lambda x: x > 0)
        intersection = s1.intersection(s2)
        self.assertEqual(intersection.cls, int)
        self.assertIn(5, intersection)
        self.assertNotIn(0, intersection)


class TestSetUnion(unittest.TestCase):
    """Test Set.union method."""

    def test_union_basic(self):
        """Test union with basic sets."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        union = s1.union(s2)
        self.assertIsInstance(union, Set)
        self.assertIn("∪", union.name)
        self.assertEqual(union.cls, list)
        self.assertIn([1, 2, 3], union)  # In both
        self.assertIn([2, 3, 4], union)  # In s1
        self.assertIn([1, 2], union)  # In s2
        self.assertNotIn([5, 6], union)  # In neither

    def test_union_different_cls(self):
        """Test union with different cls raises ValueError."""
        s1 = Set("A", int)
        s2 = Set("B", str)
        with self.assertRaises(ValueError):
            s1.union(s2)

    def test_union_not_set(self):
        """Test union with non-Set raises TypeError."""
        s = Set("A", int)
        with self.assertRaises(TypeError):
            s.union("not a set")

    def test_union_no_predicates(self):
        """Test union with sets having no predicates."""
        s1 = Set("A", int)
        s2 = Set("B", int, lambda x: x > 0)
        union = s1.union(s2)
        # Union should contain all ints (since s1 has no predicates)
        self.assertIn(0, union)
        self.assertIn(5, union)
        self.assertIn(-5, union)


class TestSetDifference(unittest.TestCase):
    """Test Set.difference method."""

    def test_difference_basic(self):
        """Test difference with basic sets."""
        s1 = Set("A", list, lambda x: len(x) == 3)
        s2 = Set("B", list, lambda x: 1 in x)
        difference = s1.difference(s2)
        self.assertIsInstance(difference, Set)
        self.assertEqual(difference.cls, list)
        self.assertIn([2, 3, 4], difference)  # In s1, not in s2
        self.assertNotIn([1, 2, 3], difference)  # In both
        self.assertNotIn([1, 2], difference)  # Not in s1

    def test_difference_different_cls(self):
        """Test difference with different cls raises ValueError."""
        s1 = Set("A", int)
        s2 = Set("B", str)
        with self.assertRaises(ValueError):
            s1.difference(s2)

    def test_difference_not_set(self):
        """Test difference with non-Set raises TypeError."""
        s = Set("A", int)
        with self.assertRaises(TypeError):
            s.difference("not a set")


class TestSetIsSubset(unittest.TestCase):
    """Test Set.is_subset method."""

    def test_is_subset_true(self):
        """Test is_subset returns True when appropriate."""
        pred1 = lambda x: len(x) == 3
        pred2 = lambda x: 1 in x
        s1 = Set("A", list, pred1)
        s2 = Set("B", list, pred1, pred2)
        self.assertTrue(s1.is_subset(s2))

    def test_is_subset_false(self):
        """Test is_subset returns False when appropriate."""
        pred1 = lambda x: len(x) == 3
        pred2 = lambda x: 1 in x
        s1 = Set("A", list, pred1)
        s2 = Set("B", list, pred2)
        self.assertFalse(s1.is_subset(s2))

    def test_is_subset_same_predicates(self):
        """Test is_subset with same predicates."""
        pred = lambda x: x > 0
        s1 = Set("A", int, pred)
        s2 = Set("B", int, pred)
        self.assertTrue(s1.is_subset(s2))
        self.assertTrue(s2.is_subset(s1))

    def test_is_subset_different_cls(self):
        """Test is_subset with different cls raises ValueError."""
        s1 = Set("A", int)
        s2 = Set("B", str)
        with self.assertRaises(ValueError):
            s1.is_subset(s2)

    def test_is_subset_not_set(self):
        """Test is_subset with non-Set raises TypeError."""
        s = Set("A", int)
        with self.assertRaises(TypeError):
            s.is_subset("not a set")

    def test_is_subset_no_predicates(self):
        """Test is_subset with no predicates."""
        s1 = Set("A", int)
        s2 = Set("B", int, lambda x: x > 0)
        # s1 has no predicates, so all([]) is True
        # (empty set of predicates is considered subset)
        self.assertTrue(s1.is_subset(s2))


class TestRemoveDuplicates(unittest.TestCase):
    """Test remove_duplicates function."""

    def test_remove_duplicates_basic(self):
        """Test remove_duplicates with basic list."""
        result = remove_duplicates([1, 2, 2, 3, 4, 3])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_remove_duplicates_no_duplicates(self):
        """Test remove_duplicates with no duplicates."""
        result = remove_duplicates([1, 2, 3, 4])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_remove_duplicates_all_duplicates(self):
        """Test remove_duplicates with all duplicates."""
        result = remove_duplicates([1, 1, 1, 1])
        self.assertEqual(result, [1])

    def test_remove_duplicates_empty(self):
        """Test remove_duplicates with empty list."""
        result = remove_duplicates([])
        self.assertEqual(result, [])

    def test_remove_duplicates_preserves_order(self):
        """Test remove_duplicates preserves order."""
        result = remove_duplicates([3, 1, 2, 1, 3, 2])
        self.assertEqual(result, [3, 1, 2])

    def test_remove_duplicates_strings(self):
        """Test remove_duplicates with strings."""
        result = remove_duplicates(['a', 'b', 'a', 'c', 'b'])
        self.assertEqual(result, ['a', 'b', 'c'])

    def test_remove_duplicates_mixed_types(self):
        """Test remove_duplicates with mixed types."""
        result = remove_duplicates([1, 'a', 1, 'b', 'a'])
        self.assertEqual(result, [1, 'a', 'b'])


class TestNegate(unittest.TestCase):
    """Test negate function."""

    def test_negate_basic(self):
        """Test negate with basic predicate."""
        def pred(x):
            return len(x) == 3
        negated = negate(pred)
        self.assertTrue(pred([1, 2, 3]))
        self.assertFalse(negated([1, 2, 3]))
        self.assertFalse(pred([1, 2]))
        self.assertTrue(negated([1, 2]))

    def test_negate_lambda(self):
        """Test negate with lambda."""
        pred = lambda x: x > 0
        negated = negate(pred)
        self.assertTrue(pred(5))
        self.assertFalse(negated(5))
        self.assertFalse(pred(0))
        self.assertTrue(negated(0))

    def test_negate_function_name(self):
        """Test negate sets function name."""
        def my_pred(x):
            return x > 0
        negated = negate(my_pred)
        self.assertEqual(negated.__name__, "not_my_pred")

    def test_negate_always_true(self):
        """Test negate with always-true predicate."""
        def always_true(x):
            return True
        negated = negate(always_true)
        self.assertFalse(negated(42))
        self.assertFalse(negated("anything"))

    def test_negate_always_false(self):
        """Test negate with always-false predicate."""
        def always_false(x):
            return False
        negated = negate(always_false)
        self.assertTrue(negated(42))
        self.assertTrue(negated("anything"))


if __name__ == '__main__':
    unittest.main()

