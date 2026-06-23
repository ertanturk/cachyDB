"""Module for the B-tree index structure used in cachyDB.

This module provides an in-memory B-tree (``BTreeMemTable``) that supports
key/value insertion, lookup, and in-order traversal with O(log n) performance.
It is intended as the primary index structure for the storage engine's memtable.
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Iterator


class BTreeNode:
    """Represents a single node in the B-tree.

    A node holds an ordered list of string keys with associated values and,
    for internal nodes, a list of child pointers.

    Attributes:
        leaf (bool): ``True`` if the node is a leaf (has no children).
        keys (list[str]): Sorted list of keys stored in this node.
        values (list[Any]): Values corresponding to each key in ``keys``.
        children (list[BTreeNode]): Child node pointers for internal nodes.
            A node with ``n`` keys has exactly ``n + 1`` children.
    """

    __slots__ = ("leaf", "keys", "values", "children")

    def __init__(self, leaf: bool = True) -> None:
        """Initializes an empty B-tree node.

        Args:
            leaf: Whether this node is a leaf.  Defaults to ``True``.
        """
        self.leaf: bool = leaf
        self.keys: list[str] = []
        self.values: list[Any] = []
        self.children: list[BTreeNode] = []

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Yields ``(key, value)`` pairs stored directly in this node.

        Note:
            This iterates only over the keys/values in this node, not its
            subtree.  Use :meth:`~BTreeMemTable.in_order_traversal` for a
            full sorted traversal.

        Yields:
            Tuples of ``(key, value)`` in key order.
        """
        for key, value in zip(self.keys, self.values):
            yield key, value

    def __len__(self) -> int:
        """Returns the number of keys stored in this node.

        Returns:
            The count of keys in ``self.keys``.
        """
        return len(self.keys)


class BTreeMemTable:
    """In-memory B-tree used as a memtable index in cachyDB.

    Supports O(log n) insert and lookup operations.  Duplicate keys are
    handled by updating the value in place (last-write-wins semantics).
    The minimum degree ``t`` controls the branching factor: each non-root
    node holds between ``t - 1`` and ``2t - 1`` keys.

    Attributes:
        root (BTreeNode): Root node of the tree.
        t (int): Minimum degree.  Must be >= 2.
        size (int): Number of distinct keys currently stored.

    Example:
        >>> tree = BTreeMemTable(t=3)
        >>> tree.insert("b", 2)
        >>> tree.insert("a", 1)
        >>> list(tree.in_order_traversal())
        [('a', 1), ('b', 2)]
    """

    MIN_DEGREE: int = 2  # Absolute lower bound for t

    def __init__(self, t: int = 10) -> None:
        """Initializes an empty B-tree with the given minimum degree.

        Args:
            t: Minimum degree of the B-tree.  Every non-root node must
                have at least ``t - 1`` keys and at most ``2t - 1`` keys.
                Must be >= ``MIN_DEGREE`` (2).

        Raises:
            ValueError: If ``t`` is less than ``MIN_DEGREE``.
        """
        if t < self.MIN_DEGREE:
            raise ValueError(
                f"B-tree minimum degree t must be >= {self.MIN_DEGREE}, got {t}. "
                "With t=1 the split operation produces empty nodes and corrupts "
                "B-tree invariants."
            )
        self.root: BTreeNode = BTreeNode(leaf=True)
        self.t: int = t
        self.size: int = 0
        self._logger: logging.Logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """Returns the number of distinct keys stored in the tree.

        Returns:
            Current key count.
        """
        return self.size

    def search(self, key: str) -> Any | None:
        """Searches for a key and returns its associated value.

        Args:
            key: The key to search for.  Must be a non-empty string.

        Returns:
            The value associated with ``key``, or ``None`` if not found.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is an empty string.
        """
        self._validate_key(key)
        return self._search(self.root, key)

    def insert(self, key: str, value: Any) -> None:
        """Inserts a key/value pair into the tree.

        If ``key`` already exists, its value is updated in place
        (last-write-wins).  Otherwise a new entry is created and
        :attr:`size` is incremented.

        Args:
            key: The key to insert.  Must be a non-empty string.
            value: The value to associate with ``key``.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is an empty string.
        """
        self._validate_key(key)
        root = self.root

        node, idx = self._find_node_and_index(root, key)
        if node is not None:
            node.values[idx] = value
            return

        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode(leaf=False)
            self.root = temp
            temp.children.append(root)
            self._split_child(temp, 0)
            self._insert_non_full(temp, key, value)
        else:
            self._insert_non_full(root, key, value)

        self.size += 1

    def contains(self, key: str) -> bool:
        """Returns ``True`` if the tree holds an entry for ``key``.

        Unlike :meth:`search`, which returns ``None`` for both a missing key
        and a key whose value is ``None``, this method unambiguously reports
        whether the key exists.

        Args:
            key: The key to check.  Must be a non-empty string.

        Returns:
            ``True`` if the key is present, ``False`` otherwise.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is an empty string.
        """
        self._validate_key(key)
        node, _ = self._find_node_and_index(self.root, key)
        return node is not None

    def in_order_traversal(self) -> Iterator[tuple[str, Any]]:
        """Yields all ``(key, value)`` pairs in ascending key order.

        Yields:
            Tuples of ``(key, value)`` sorted by key.
        """
        yield from self._in_order(self.root)

    def clear(self) -> None:
        """Removes all entries from the tree and resets the size counter.

        After this call, :attr:`size` is 0 and the tree behaves as if newly
        constructed.
        """
        self.root = BTreeNode(leaf=True)
        self.size = 0
        self._logger.debug("MemTable B-Tree has been cleared.")

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validates that ``key`` is a non-empty string.

        Args:
            key: The key to validate.

        Raises:
            TypeError: If ``key`` is not a ``str``.
            ValueError: If ``key`` is an empty string.
        """
        if type(key) is not str:
            raise TypeError(f"key must be a str, got {type(key).__name__}.")
        if not key:
            raise ValueError("key must not be an empty string.")

    def _search(self, node: BTreeNode, key: str) -> Any | None:
        """Recursively searches for ``key`` starting from ``node``.

        Args:
            node: The subtree root to search within.
            key: The key to look up.

        Returns:
            The associated value if found, ``None`` otherwise.
        """
        i: int = bisect.bisect_left(node.keys, key)
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        if node.leaf:
            return None
        return self._search(node.children[i], key)

    def _find_node_and_index(self, node: BTreeNode, key: str) -> tuple[BTreeNode | None, int]:
        """Returns the node and index where ``key`` lives, or ``(None, -1)``.

        Args:
            node: The subtree root to search within.
            key: The key to locate.

        Returns:
            A tuple ``(node, index)`` if ``key`` is found, or
            ``(None, -1)`` if ``key`` is absent from the subtree.
        """
        i: int = 0
        n: int = len(node.keys)
        while i < n and key > node.keys[i]:
            i += 1

        if i < n and key == node.keys[i]:
            return node, i

        if node.leaf:
            return None, -1

        return self._find_node_and_index(node.children[i], key)

    def _split_child(self, parent: BTreeNode, i: int) -> None:
        """Splits the full child at index ``i`` of ``parent``.

        The child at ``parent.children[i]`` must have exactly ``2t - 1``
        keys.  After the split, its median key is promoted to ``parent``,
        and the right half of its keys/children are moved to a new sibling
        node inserted at ``parent.children[i + 1]``.

        Args:
            parent: The internal node whose ``i``-th child is to be split.
            i: Index of the full child within ``parent.children``.
        """
        t: int = self.t
        full_child: BTreeNode = parent.children[i]
        new_child: BTreeNode = BTreeNode(leaf=full_child.leaf)

        # Right half of keys/values (excluding median) go to the new child
        new_child.keys = full_child.keys[t:]
        new_child.values = full_child.values[t:]

        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

        # Insert the new child to the right of the split position
        parent.children.insert(i + 1, new_child)

        # Promote the median key/value to the parent
        parent.keys.insert(i, full_child.keys[t - 1])
        parent.values.insert(i, full_child.values[t - 1])

        # Truncate the left child to t-1 keys (drop median and right half)
        full_child.keys = full_child.keys[: t - 1]
        full_child.values = full_child.values[: t - 1]

    def _insert_non_full(self, node: BTreeNode, key: str, value: Any) -> None:
        """Inserts ``(key, value)`` into the subtree rooted at ``node``.

        ``node`` must have fewer than ``2t - 1`` keys before this call.
        Any full child encountered on the way down is split proactively so
        that every recursive call targets a non-full node.

        Args:
            node: The subtree root.  Must not be full.
            key: The key to insert.  Must not already exist in the subtree.
            value: The value to associate with ``key``.
        """
        if node.leaf:
            i: int = bisect.bisect_left(node.keys, key)
            node.keys.insert(i, key)
            node.values.insert(i, value)
        else:
            i: int = bisect.bisect_left(node.keys, key)
            if len(node.children[i].keys) == (2 * self.t) - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def _in_order(self, node: BTreeNode) -> Iterator[tuple[str, Any]]:
        """Recursively yields all entries in ``node``'s subtree in key order.

        For a leaf node, all ``(key, value)`` pairs are yielded directly.
        For an internal node, each left subtree is yielded before its
        separating key, and the rightmost subtree is yielded last.

        Args:
            node: The subtree root to traverse.

        Yields:
            Tuples of ``(key, value)`` in ascending key order.
        """
        if node.leaf:
            for i in range(len(node.keys)):
                yield (node.keys[i], node.values[i])
        else:
            for i in range(len(node.keys)):
                yield from self._in_order(node.children[i])
                yield (node.keys[i], node.values[i])
            yield from self._in_order(node.children[-1])
