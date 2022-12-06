"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    $f(x, y) = x * y$

    Args:
        x: float multiplier
        y: float multiplicand

    Returns:
        The product of the multiplier and multiplicand.
    """
    return x * y


def id(x: float) -> float:
    """
    $f(x) = x$

    Args:
        x: float value

    Returns:
        The input float.
    """
    return x


def add(x: float, y: float) -> float:
    """
    $f(x, y) = x + y$

    Args:
        x: float addend
        y: float addend

    Returns:
        The sum of the two addends.
    """
    return x + y


def neg(x: float) -> float:
    """
    $f(x) = -x$

    Args:
        x: float value

    Returns:
        Flips the sign of the input value
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    $f(x) =$ 1.0 if x is less than y else 0.0

    Args:
        x: float left-hand side value
        y: float right-hand side value

    Returns:
        If the left-hand side value is less than the right-hand side value.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    $f(x) =$ 1.0 if x is equal to y else 0.0

    Args:
        x: float left-hand side value
        y: float right-hand side value

    Returns:
        If the left-hand side value is equal to the right-hand side value.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    $f(x) =$ x if x is greater than y else y

    Args:
        x: float value
        y: float value

    Returns:
        The bigger value between the two inputs.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    $f(x) = |x - y| < 1e-2$

    Args:
        x: float value
        y: float value

    Returns:
        If the absolute values of the two floats are within a range of 1e-2 to
        each other.
    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.

    Args:
        x: float value

    Returns:
        if x >= 0 returns: 1 / (1 + e^(-x))
        if x < 0 returns: e^(x) / (1 + e^(x))
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    # return 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)

    Args:
        x: float value

    Returns:
        if x > 0 returns: x
        if x <= 0 returns: 0
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""
    If $f = log$ as above, compute $d \times f'(x)$

    Args:
        x: float log argument
        d: float constant

    Returns:
        The derivative of d * log(x) which is d / x
    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """
    $f(x) = 1/x$

    Args:
        x: float value

    Returns:
        the inverse of the input value (1 / x)
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""
    If $f(x) = 1/x$ compute $d \times f'(x)$

    Args:
        x: float inverse argument
        d: float constant

    Returns:
        The derivative of d / x which is -d / (x^2)
    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    r"""
    If $f = relu$ compute $d \times f'(x)$

    Args:
        x: float relu argument
        d: float constant

    Returns:
        The derivative of relu(x) which is:
        - 0 for x <= 0
        - 1 for x > 0
    """
    return d if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def inner_map(in_list: Iterable[float]) -> Iterable[float]:
        out_list = []
        for el in in_list:
            out_list.append(fn(el))
        return out_list

    return inner_map


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def inner_zip(
        in_list1: Iterable[float], in_list2: Iterable[float]
    ) -> Iterable[float]:
        out_list = [fn(el1, el2) for el1, el2 in zip(in_list1, in_list2)]
        return out_list

    return inner_zip


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def inner_reduce(rem_list: Iterable[float]) -> float:
        val = start
        for l in rem_list:
            val = fn(val, l)
        return val
        # work_list = list(rem_list)
        # if len(work_list) == 0:
        #     return start
        # else:
        #     return fn(work_list[-1], inner_reduce(work_list[:-1]))

    return inner_reduce


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 1.0)(ls)
