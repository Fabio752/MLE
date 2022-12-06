from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_n = [val for val in vals]
    vals_n[arg] += epsilon
    vals_p = [val for val in vals]
    vals_p[arg] -= epsilon

    return (f(*vals_n) - f(*vals_p)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    seen_nodes = set()
    sorted_nodes = []

    # Recursive DFS.
    def dfs(current_node: Variable) -> None:
        if current_node.unique_id in seen_nodes:
            return
        if not current_node.is_leaf():
            for parent_node in current_node.parents:
                if not parent_node.is_constant():
                    dfs(parent_node)
        seen_nodes.add(current_node.unique_id)
        sorted_nodes.append(current_node)

    # Root node dfs call.
    dfs(variable)

    return sorted_nodes[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_nodes = topological_sort(variable)
    variable_to_derivative = {var.unique_id: 0 for var in sorted_nodes}

    # Store initial derivative for rightmost variable.
    variable_to_derivative[variable.unique_id] = deriv

    # Backpropagate for each node in topological sorted list.
    for current_node in sorted_nodes:
        if current_node.is_leaf():
            current_node.accumulate_derivative(
                variable_to_derivative[current_node.unique_id]
            )
        else:
            for next_node, next_deriv in current_node.chain_rule(
                variable_to_derivative[current_node.unique_id]
            ):
                variable_to_derivative[next_node.unique_id] += next_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
