"""
PA 8: The VTable Dispatcher -- starter.

Complete VTable, build_vtable, USIObject, and dispatch below. See
PA_08_The_VTable_Dispatcher.md, Part B, for the full requirements.

Each table entry follows PA 7's Closure.call(args, caller_env) interface.
The harness supplies small closure-like methods so it can test dispatch
without adding string literals to the earlier interpreter grammar. Your
dispatch implementation must invoke .call with the receiver as the first
argument, followed by the explicit method arguments.
"""

from typing import Dict, List, Optional, Protocol

from gc_heap import GCHeap, Handle


class MethodLike(Protocol):
    def call(self, args: list, caller_env) -> object:
        """PA 7 Closure-compatible call interface."""


class ClassDecl:
    """Given: a minimal class declaration -- just a name and its own (non-inherited) methods."""

    __slots__ = ("name", "methods")

    def __init__(self, name: str, methods: Dict[str, MethodLike]) -> None:
        self.name = name
        self.methods = methods  # methods declared on THIS class only


class MethodNotFoundError(AttributeError):
    """Raised when a method name has no slot anywhere in a class's VTable chain."""

    def __init__(self, class_name: str, method_name: str) -> None:
        super().__init__(f"class '{class_name}' has no method '{method_name}' (not declared or inherited)")
        self.class_name = class_name
        self.method_name = method_name


class VTable:
    def __init__(self, entries: Dict[str, MethodLike], owning_class_name: str = "?") -> None:
        """`owning_class_name` is just for error messages -- store it alongside entries."""
        # Own the dict so later edits to the caller's mapping can't change our slots.
        # Insertion order is the slot order: inherited slots first, new ones appended.
        self.entries: Dict[str, MethodLike] = dict(entries)
        self.owning_class_name = owning_class_name

    def resolve(self, method_name: str) -> MethodLike:
        """
        Return the method object bound to `method_name`. Raise a clear
        error (e.g. AttributeError) whose message names BOTH the
        owning class and the missing method if it doesn't exist.
        """
        # The table is already flattened by build_vtable, so one lookup covers
        # the whole inheritance chain; a miss here means no ancestor has it.
        try:
            return self.entries[method_name]
        except KeyError:
            raise MethodNotFoundError(self.owning_class_name, method_name) from None

    def slots(self) -> list:
        """Slot layout as (index, method_name) pairs, in slot order."""
        return list(enumerate(self.entries))

    def __repr__(self) -> str:
        return f"<VTable {self.owning_class_name} slots={list(self.entries)}>"


def build_vtable(class_node: ClassDecl, parent_vtable: Optional[VTable] = None) -> VTable:
    """
    Start from a copy of parent_vtable's entries (inherited methods),
    then overwrite only the slots class_node itself redeclares. A
    method class_node does NOT override must resolve to the EXACT SAME
    method object the parent uses -- don't wrap or duplicate it.
    Pass class_node.name as the new VTable's owning_class_name.
    """
    # Shallow copy: the dict is new, but each inherited value is the parent's
    # very same method object. Overriding an existing key keeps its slot index.
    entries: Dict[str, MethodLike] = dict(parent_vtable.entries) if parent_vtable is not None else {}
    for method_name, method in class_node.methods.items():
        entries[method_name] = method
    return VTable(entries, owning_class_name=class_node.name)


class USIObject:
    """
    PA 9: when given a heap, field writes through set_field() are
    counted references. Handles passed in the constructor's `fields`
    are NOT incref'd: like HeapNode, the object takes over the caller's
    reference (e.g. the creating reference from heap.allocate()).
    """

    def __init__(self, class_descriptor: VTable, fields: dict, heap: Optional[GCHeap] = None) -> None:
        if not isinstance(class_descriptor, VTable):
            raise TypeError("USIObject needs its runtime class's VTable as class_descriptor.")
        self.vtable = class_descriptor
        self.fields = dict(fields)
        self.heap = heap

    def get_field(self, name: str) -> object:
        return self.fields[name]

    def set_field(self, name: str, value: object) -> None:
        """`obj.name = value;` -- incref the new handle, decref the overwritten one."""
        old = self.fields.get(name)
        self.fields[name] = self.heap.store(old, value) if self.heap is not None else value

    def get_child_handles(self) -> List[int]:
        """Handles held in this object's CURRENT fields (for GCHeap.decref's cascade)."""
        return [v for v in self.fields.values() if isinstance(v, Handle)]

    @property
    def class_name(self) -> str:
        return self.vtable.owning_class_name

    def __repr__(self) -> str:
        return f"<USIObject {self.class_name} fields={self.fields!r}>"


def new_object(heap: GCHeap, class_descriptor: VTable, fields: Optional[dict] = None) -> Handle:
    """PA 9: `ClassName()` -- allocate a tracked USIObject; its handle starts at refcount 1."""
    return heap.allocate(USIObject(class_descriptor, fields or {}, heap))


def dispatch(obj: USIObject, method_name: str, args: list, caller_env) -> object:
    """
    Resolve method_name through obj's OWN VTable (never a declared/
    static type), then call the resolved method via PA 7's interface:
    resolved_method.call([obj, *args], caller_env).
    """
    if not isinstance(obj, USIObject):
        raise TypeError(f"cannot dispatch '{method_name}' on non-object {obj!r}")
    method = obj.vtable.resolve(method_name)
    # The receiver is bound as the implicit first parameter (self).
    return method.call([obj, *args], caller_env)
