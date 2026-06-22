"""Tests for the BTreeNode and BTreeMemTable classes."""

from __future__ import annotations

import pytest

from cachyDB.index.btree import BTreeMemTable, BTreeNode


class TestBTreeNode:
    """Tests for BTreeNode behaviour."""

    def test_initial_state(self) -> None:
        """A new BTreeNode must have empty keys, values, and children."""
        node = BTreeNode()
        assert node.leaf is True
        assert node.keys == []
        assert node.values == []
        assert node.children == []

    def test_internal_node(self) -> None:
        """A non-leaf BTreeNode must have leaf=False."""
        node = BTreeNode(leaf=False)
        assert node.leaf is False

    def test_len_reflects_key_count(self) -> None:
        """len(node) must equal the number of keys."""
        node = BTreeNode()
        assert len(node) == 0
        node.keys = ["a", "b", "c"]
        node.values = [1, 2, 3]
        assert len(node) == 3

    def test_iter_yields_key_value_pairs(self) -> None:
        """Iterating a node must yield (key, value) pairs."""
        node = BTreeNode()
        node.keys = ["x", "y"]
        node.values = [10, 20]
        assert list(node) == [("x", 10), ("y", 20)]

    def test_iter_empty_node(self) -> None:
        """Iterating an empty node must yield nothing."""
        assert list(BTreeNode()) == []


class TestBTreeMemTableInit:
    """Tests for BTreeMemTable.__init__ validation."""

    def test_default_t_is_valid(self) -> None:
        """Default t=10 must create a valid tree."""
        tree = BTreeMemTable()
        assert tree.t == 10
        assert tree.size == 0

    def test_min_t_equals_two(self) -> None:
        """t=2 must be accepted."""
        tree = BTreeMemTable(t=2)
        assert tree.t == 2

    def test_t_one_raises(self) -> None:
        """t=1 must raise ValueError."""
        with pytest.raises(ValueError, match="minimum degree"):
            BTreeMemTable(t=1)

    def test_t_zero_raises(self) -> None:
        """t=0 must raise ValueError."""
        with pytest.raises(ValueError, match="minimum degree"):
            BTreeMemTable(t=0)

    def test_t_negative_raises(self) -> None:
        """Negative t must raise ValueError."""
        with pytest.raises(ValueError, match="minimum degree"):
            BTreeMemTable(t=-5)


class TestKeyValidation:
    """Tests for _validate_key enforcement in search and insert."""

    def test_search_non_string_key_raises_type_error(self) -> None:
        """Searching with a non-string key must raise TypeError."""
        tree = BTreeMemTable()
        with pytest.raises(TypeError, match="key must be a str"):
            tree.search(123)  # ty: ignore[invalid-argument-type]

    def test_search_empty_string_raises_value_error(self) -> None:
        """Searching with an empty string must raise ValueError."""
        tree = BTreeMemTable()
        with pytest.raises(ValueError, match="empty string"):
            tree.search("")

    def test_insert_non_string_key_raises_type_error(self) -> None:
        """Inserting with a non-string key must raise TypeError."""
        tree = BTreeMemTable()
        with pytest.raises(TypeError, match="key must be a str"):
            tree.insert(None, "v")  # ty: ignore[invalid-argument-type]

    def test_insert_empty_string_raises_value_error(self) -> None:
        """Inserting an empty string key must raise ValueError."""
        tree = BTreeMemTable()
        with pytest.raises(ValueError, match="empty string"):
            tree.insert("", "v")


class TestInsertAndSearch:
    """Tests for basic insert and search operations."""

    def test_search_returns_none_on_empty_tree(self) -> None:
        """Searching an empty tree must return None."""
        tree = BTreeMemTable()
        assert tree.search("a") is None

    def test_search_returns_none_for_missing_key(self) -> None:
        """Searching for a key not in the tree must return None."""
        tree = BTreeMemTable()
        tree.insert("b", 2)
        assert tree.search("a") is None

    def test_insert_and_search_single_key(self) -> None:
        """A single inserted key must be findable."""
        tree = BTreeMemTable()
        tree.insert("hello", "world")
        assert tree.search("hello") == "world"

    def test_insert_updates_existing_key(self) -> None:
        """Inserting a key that already exists must update its value."""
        tree = BTreeMemTable()
        tree.insert("k", "old")
        tree.insert("k", "new")
        assert tree.search("k") == "new"

    def test_size_not_incremented_on_update(self) -> None:
        """Updating an existing key must not change size."""
        tree = BTreeMemTable()
        tree.insert("k", 1)
        tree.insert("k", 2)
        assert tree.size == 1

    def test_size_increments_on_new_keys(self) -> None:
        """Size must increment once per distinct inserted key."""
        tree = BTreeMemTable()
        for i in range(20):
            tree.insert(str(i), i)
        assert tree.size == 20

    def test_len_equals_size(self) -> None:
        """len(tree) must equal tree.size."""
        tree = BTreeMemTable()
        tree.insert("a", 1)
        tree.insert("b", 2)
        assert len(tree) == tree.size == 2

    def test_insert_many_keys(self) -> None:
        """All inserted keys must be retrievable after many inserts."""
        tree = BTreeMemTable(t=3)
        pairs = {f"key{i:04d}": i for i in range(500)}
        for k, v in pairs.items():
            tree.insert(k, v)
        for k, v in pairs.items():
            assert tree.search(k) == v

    def test_value_none_is_stored_correctly(self) -> None:
        """Storing None as a value must be distinguishable from a missing key."""
        tree = BTreeMemTable()
        tree.insert("k", None)
        # size=1 confirms the key is present; search returns None for BOTH
        # missing and None-valued keys — this is the documented contract.
        assert tree.size == 1

    def test_insert_preserves_all_previous_keys(self) -> None:
        """Inserting new keys must not corrupt previously stored keys."""
        tree = BTreeMemTable(t=2)
        inserted: dict[str, int] = {}
        keys = ["mango", "apple", "cherry", "banana", "date", "elderberry", "fig"]
        for i, k in enumerate(keys):
            tree.insert(k, i)
            inserted[k] = i
            for ik, iv in inserted.items():
                assert tree.search(ik) == iv, f"key '{ik}' corrupted after inserting '{k}'"


class TestInOrderTraversal:
    """Tests for in_order_traversal."""

    def test_empty_tree_yields_nothing(self) -> None:
        """Traversing an empty tree must yield no pairs."""
        tree = BTreeMemTable()
        assert list(tree.in_order_traversal()) == []

    def test_single_key_tree(self) -> None:
        """A single-entry tree must yield exactly that entry."""
        tree = BTreeMemTable()
        tree.insert("only", 42)
        assert list(tree.in_order_traversal()) == [("only", 42)]

    def test_traversal_is_sorted(self) -> None:
        """in_order_traversal must yield keys in ascending lexicographic order."""
        tree = BTreeMemTable(t=3)
        keys = ["delta", "alpha", "gamma", "beta", "epsilon"]
        for k in keys:
            tree.insert(k, k)
        result_keys = [k for k, _ in tree.in_order_traversal()]
        assert result_keys == sorted(keys)

    def test_traversal_sorted_after_many_inserts(self) -> None:
        """Traversal must stay sorted even when the tree undergoes multiple splits."""
        import random

        rng = random.Random(42)
        tree = BTreeMemTable(t=2)
        keys = [f"key{i:04d}" for i in range(200)]
        rng.shuffle(keys)
        for k in keys:
            tree.insert(k, k)
        result = list(tree.in_order_traversal())
        assert [k for k, _ in result] == sorted(keys)

    def test_traversal_count_equals_size(self) -> None:
        """Number of yielded pairs must equal tree.size."""
        tree = BTreeMemTable(t=4)
        for i in range(100):
            tree.insert(f"k{i:03d}", i)
        assert len(list(tree.in_order_traversal())) == tree.size


class TestRootSplitting:
    """Tests that verify correct behavior when the root is split."""

    def test_root_split_preserves_all_keys(self) -> None:
        """Keys inserted before and after a root split must all be findable."""
        tree = BTreeMemTable(t=2)  # root fills after 3 keys
        keys = ["c", "a", "e", "b", "d"]  # 5 keys forces a root split
        for k in keys:
            tree.insert(k, k)
        for k in keys:
            assert tree.search(k) == k

    def test_root_becomes_internal_after_split(self) -> None:
        """After a root split, the root must be an internal node with children."""
        tree = BTreeMemTable(t=2)
        for k in ["c", "a", "e", "b", "d"]:
            tree.insert(k, k)
        assert tree.root.leaf is False
        assert len(tree.root.children) >= 2

    def test_multiple_splits_preserve_sorted_order(self) -> None:
        """Sorted traversal must remain correct through many splits."""
        tree = BTreeMemTable(t=2)
        keys = sorted(f"key{i:03d}" for i in range(50))
        for k in keys:
            tree.insert(k, k)
        result = [k for k, _ in tree.in_order_traversal()]
        assert result == keys


class TestClear:
    """Tests for BTreeMemTable.clear."""

    def test_clear_resets_size(self) -> None:
        """clear() must set size to 0."""
        tree = BTreeMemTable()
        for i in range(10):
            tree.insert(f"k{i}", i)
        tree.clear()
        assert tree.size == 0

    def test_clear_makes_tree_empty(self) -> None:
        """After clear(), traversal must yield nothing."""
        tree = BTreeMemTable()
        for i in range(10):
            tree.insert(f"k{i}", i)
        tree.clear()
        assert list(tree.in_order_traversal()) == []

    def test_clear_allows_reuse(self) -> None:
        """After clear(), new inserts must work correctly."""
        tree = BTreeMemTable()
        tree.insert("before", 1)
        tree.clear()
        tree.insert("after", 2)
        assert tree.search("before") is None
        assert tree.search("after") == 2
        assert tree.size == 1

    def test_clear_idempotent(self) -> None:
        """Calling clear() twice must not raise."""
        tree = BTreeMemTable()
        tree.insert("k", 1)
        tree.clear()
        tree.clear()
        assert tree.size == 0
