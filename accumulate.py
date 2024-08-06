import copy
import operator
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping, MutableSet
from typing import Iterable, Optional, TypeVar, Union

try:
    from typing import Protocol, runtime_checkable  # type: ignore
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore
    
T = TypeVar("T")


@runtime_checkable
class Addable(Protocol):
    def __add__(self: T, other: T) -> T:
        ...


Accumulatable = Union[Addable, MutableSet, MutableMapping]


def add(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, without altering inputs

    This may make copies in certain situations
    """
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.add(a, b)
    if isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.or_(a, b)
    elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        # capture type(X) by shallow copy and clear
        # since we don't know the signature of type(X).__init__
        if isinstance(b, type(a)):
            out = copy.copy(a)
        elif isinstance(a, type(b)):
            out = copy.copy(b)
        else:
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        out.clear()
        lhs, rhs = set(a), set(b)
        # Keep the order of elements as far as possible
        for key in a:
            if key in rhs:
                out[key] = add(a[key], b[key])
            else:
                out[key] = copy.deepcopy(a[key])
        for key in b:
            if key not in lhs:
                out[key] = copy.deepcopy(b[key])
        return out
    raise ValueError(
        f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
    )


def iadd(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, assuming the first is mutable"""
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.iadd(a, b)
    elif isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.ior(a, b)
    elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        if not isinstance(b, type(a)):
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        lhs, rhs = set(a), set(b)
        # Keep the order of elements as far as possible
        for key in a:
            if key in rhs:
                a[key] = iadd(a[key], b[key])
        for key in b:
            if key not in lhs:
                a[key] = copy.deepcopy(b[key])
        return a
    raise ValueError(
        f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
    )


def accumulate(
    items: Iterable[Optional[Accumulatable]], accum: Optional[Accumulatable] = None
) -> Optional[Accumulatable]:
    gen = (x for x in items if x is not None)
    try:
        if accum is None:
            accum = next(gen)
            # we want to produce a new object so that the input is not mutated
            accum = add(accum, next(gen))
        while True:
            # subsequent additions can happen in-place, which may be more performant
            accum = iadd(accum, next(gen))
    except StopIteration:
        pass
    return accum