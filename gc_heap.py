"""
PA 9: The Garbage Collector -- starter.

Complete GCHeap below. See the assignment, Part B,
for the full requirements.

Use gc_heap.py as specified in the assignment and imported by the harness.

Design note: GCHeap.allocate(obj) takes exactly one argument per the
spec, so an allocated object that references OTHER heap handles must
expose them itself for decref's cascade to find -- via a
get_child_handles() -> list[int] method. HeapNode below is a given
example object shaped this way, used by the cyclic-structure test.

Integration note: handles are returned as Handle, a subclass of int.
They are still plain ints to the heap API and to HeapNode, but the
interpreter (frames.py, parameters.py, dispatcher.py) can tell a heap
reference apart from an ordinary USILang number, so a field such as
age=3 is never mistaken for handle 3.

Cycle note (Requirement 8): this heap is PURE reference counting. It
does not run any backup tracing/mark-and-sweep pass, so an unreachable
cycle (Part A, Question 2) is intentionally left alive -- every node in
it still holds a reference to another, so no count can reach zero.
"""

from typing import Any, Dict, Iterator, List, Optional


class Handle(int):
    """A heap handle: an int the interpreter can recognize as a reference."""

    def __repr__(self) -> str:
        return f"Handle({int(self)})"


class HeapNode:
    """
    Given: a minimal heap object whose fields may hold OTHER handles
    (ints). get_child_handles() is what GCHeap.decref's cascade calls
    to find them -- it must reflect the CURRENT field values, not a
    snapshot taken at creation time.
    """

    def __init__(self, **fields: Any) -> None:
        self.fields = fields

    def get_child_handles(self) -> List[int]:
        return [v for v in self.fields.values() if isinstance(v, int)]


class GCHeap:
    def __init__(self) -> None:
        # handle -> stored object, and handle -> refcount. A handle is
        # present in both maps exactly while it is alive.
        self._objects: Dict[int, Any] = {}
        self._refcounts: Dict[int, int] = {}
        # Handles start at 1 so 0 is never a valid reference.
        self._next_handle = 1
        # Freed handles in the order they were reclaimed (for tracing/tests).
        self.freed_log: List[int] = []

    def allocate(self, obj: Any) -> int:
        """Store obj, return a new handle. Initial refcount is 1 (the creating reference)."""
        handle = Handle(self._next_handle)
        self._next_handle += 1
        self._objects[handle] = obj
        self._refcounts[handle] = 1
        return handle

    def incref(self, handle: int) -> None:
        self._require_alive(handle, "incref")
        self._refcounts[handle] += 1

    def decref(self, handle: int) -> None:
        """
        Decrement the refcount. When it hits 0: free the object, AND
        cascade-decref every handle returned by the freed object's
        get_child_handles() (if it has that method), recursively.
        """
        self._require_alive(handle, "decref")
        # Explicit worklist instead of Python recursion, so a long chain
        # of refcount-1 objects is torn down in one call without hitting
        # the interpreter's recursion limit.
        pending = [handle]
        while pending:
            current = pending.pop()
            self._require_alive(current, "decref")
            self._refcounts[current] -= 1
            if self._refcounts[current] > 0:
                continue
            # Read the children BEFORE forgetting the object: the cascade
            # must see the object's current field values.
            obj = self._objects.pop(current)
            del self._refcounts[current]
            self.freed_log.append(current)
            get_children = getattr(obj, "get_child_handles", None)
            if get_children is not None:
                pending.extend(get_children())

    def is_alive(self, handle: int) -> bool:
        return handle in self._refcounts

    def live_count(self) -> int:
        """Number of currently-alive (refcount > 0) handles."""
        return len(self._refcounts)

    # ---- extras used by the interpreter hooks and tests -----------------

    def refcount(self, handle: int) -> int:
        """Current refcount of `handle` (0 once freed)."""
        return self._refcounts.get(handle, 0)

    def get(self, handle: int) -> Any:
        """Dereference a live handle to the object it names."""
        self._require_alive(handle, "get")
        return self._objects[handle]

    def store(self, old: Any, new: Any) -> Any:
        """
        Write barrier for overwriting one storage slot: incref the new
        value's handle, then decref the handle it overwrote. Incref goes
        first so `x = x` can never free x's object mid-assignment.
        Non-handle values (numbers, None) are ignored. Returns `new`.
        """
        if isinstance(new, Handle):
            self.incref(new)
        if isinstance(old, Handle):
            self.decref(old)
        return new

    def _require_alive(self, handle: int, operation: str) -> None:
        if handle not in self._refcounts:
            raise ValueError(f"{operation}() on dead or unknown handle {handle!r}")


def handles_in(value: Any) -> Iterator[Handle]:
    """
    Yield the heap handle(s) a stored value holds: a bare Handle, or a
    Handle boxed in a PA 7 Cell. Anything else holds no handle.
    """
    from parameters import Cell  # lazy: parameters imports this module

    if isinstance(value, Cell):
        value = value.value
    if isinstance(value, Handle):
        yield value


def release(heap: Optional[GCHeap], value: Any) -> None:
    """Drop the reference held by one storage slot (scope exit / overwrite)."""
    if heap is None:
        return
    for handle in handles_in(value):
        heap.decref(handle)


def retain(heap: Optional[GCHeap], value: Any) -> None:
    """Record a new storage slot that now references `value`."""
    if heap is None:
        return
    for handle in handles_in(value):
        heap.incref(handle)
