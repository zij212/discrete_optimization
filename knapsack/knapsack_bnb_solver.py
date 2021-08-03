from collections import namedtuple
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('knapsack_solver')

Item = namedtuple("Item", ['index', 'value', 'weight', 'unit_weight_value'])


class Node:
    def __init__(self, depth, parent=None, left_child=None, right_child=None):
        # the item's sorted index = depth of node - 1
        self.depth = depth
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

        self.value = None
        self.room = None
        self.estimate = None

    def __repr__(self):
        return f"Item {self.depth - 1}, value={self.value}, room={self.room}, estimate={self.estimate}, parent={self.parent}"


class KnapsackBnBDFS:
    """
    Args:
        input_data: str
    Attributes
        items:          list of Item,
                        sorted by descending unite_weight_value
        item_count:     int,
                        number of candidates
        capacity:       int,
                        weight capacity of the knapsack
        best_node:      Node,
                        current best solution node
    Methods:
        _load_data
        _update_current_node_attributes
        _estimate_greedy_future_rewards
        _get_next_node
        solve
    """
    def __init__(self, input_data):
        self.items = []
        self.item_count = None
        self.capacity = None
        self.best_node = None

        self._load_data(input_data)

        logger.debug(f'item_count: {self.item_count}, capacity: {self.capacity}')
        logger.debug(f'items sorted by (descending unit_weight_value, weight): \n{self.items}')

    def _load_data(self, input_data):
        """
        Args:
            input_data: str

        Set:
            self.item_count: number of items to choose from
            self.capacity: capacity of the knapsack
            self.items: List of items, sorted descendingly by unit_weight_value
        """

        lines = input_data.split('\n')
        first_line = lines[0].split()
        self.item_count = int(first_line[0])
        self.capacity = int(first_line[1])

        for i in range(1, self.item_count + 1):
            line = lines[i]
            parts = line.split()
            value = int(parts[0])
            weight = int(parts[1])
            unit_weight_value = value / weight
            self.items.append(Item(i - 1, value, weight, unit_weight_value))

        self.items.sort(key=lambda x: [x.unit_weight_value, x.weight], reverse=True)

    def _get_next_node(self, current_node):
        """
        Branch and Bound DFS
        Explore up and right if the current_node is not a feasible solution, or if the expected total
        rewards is less than current best solution, or if we have reached a leaf node. Otherwise, keep
        exploring down and left.

        Args:
            current_node: Node

        Returns:
            next_node: Node, or None if ended at root node
        """

        if current_node.estimate < self.best_node.value or current_node.depth == self.item_count:

            while True:
                current_node = current_node.parent
                if current_node is None:
                    return None
                if current_node.right_child is not None:
                    continue
                else:
                    break

        next_node = Node(current_node.depth+1, parent=current_node)
        if current_node.left_child is None:
            current_node.left_child = next_node
        else:
            current_node.right_child = next_node

        return next_node

    def _estimate_greedy_future_rewards(self, depth, remaining_capacity):
        """
        At the current depth, keep greedily taking items (or fraction of an item) until there is no room left

        Args:
            depth: int
            remaining_capacity: int

        Returns:
            value: int
        """
        if depth == self.item_count or remaining_capacity == 0:
            return 0

        value = 0
        while depth < self.item_count:
            weight = self.items[depth].weight
            if weight <= remaining_capacity:
                value += self.items[depth].value
                remaining_capacity -= weight
            else:
                value += self.items[depth].unit_weight_value * remaining_capacity
                break
            depth += 1

        return value

    def _update_current_node_attributes(self, current_node):
        """
        Modify the value, room, and estimate of the current node

        Args:
            current_node: Node
        """

        if not current_node.parent:
            current_node.value = 0
            current_node.room = self.capacity
            current_node.estimate = self._estimate_greedy_future_rewards(current_node.depth, self.capacity)

        # First time visiting a non-root node
        elif not current_node.value:
            is_taken = current_node.parent.right_child is None
            current_node.room = current_node.parent.room - is_taken * self.items[current_node.depth-1].weight

            if current_node.room >= 0:
                current_node.value = current_node.parent.value + is_taken * self.items[current_node.depth-1].value
                current_node.estimate = current_node.value + self._estimate_greedy_future_rewards(
                    current_node.depth, current_node.room)
            else:
                current_node.value = -np.inf
                current_node.estimate = -np.inf

    def solve(self):

        current_node = Node(0)
        self.best_node = current_node

        while True:
            self._update_current_node_attributes(current_node)
            if current_node.value > self.best_node.value:
                self.best_node = current_node
            current_node = self._get_next_node(current_node)
            if current_node is None:
                break

        obj = self.best_node.value
        weight = 0
        current_node = self.best_node
        output_data = str(obj) + ' ' + str(1) + '\n'
        taken = [0] * self.item_count
        while True:
            item = self.items[current_node.depth-1]
            taken[item.index] = 1 if current_node is current_node.parent.left_child else 0
            weight += item.weight * taken[item.index]
            current_node = current_node.parent
            if current_node.depth == 0:
                break
        logger.debug(f"Knapsack capacity: {self.capacity}, actual weight {weight}")
        output_data += ' '.join(map(str, taken))
        return output_data


if __name__ == '__main__':
    file_location = '/Users/zinanj/projects/discrete_optimization/knapsack/data/ks_10000_0'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    ks = KnapsackBnBDFS(input_data)
    solution = ks.solve()
    print(solution)









