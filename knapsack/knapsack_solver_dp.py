from collections import namedtuple
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('knapsack_solver')

Item = namedtuple("Item", ['value', 'weight'])


class KnapsackDP:
    """
    Solve's 0/1 Knapsack Problem using Dynamic Programming

    Args:
        input_data: str
    Attributes
        items:          list of Item,
                        sorted by descending unite_weight_value
        item_count:     int,
                        number of candidates
        capacity:       int,
                        weight capacity of the knapsack
    Methods:
        _load_data
        solve
    """
    def __init__(self, input_data):
        self.items = []
        self.item_count = None
        self.capacity = None
        self.sol_matrix = None

        self._load_data(input_data)

        logger.debug(f'item_count: {self.item_count}, capacity: {self.capacity}')

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
            self.items.append(Item(value, weight))

        self.sol_matrix = np.zeros((self.item_count, self.capacity+1), int)

    def _populate_sol_matrix(self):

        item_idx = 0
        cap = 0

        while True:
            weight = self.items[item_idx].weight
            value = self.items[item_idx].value

            if item_idx == 0:
                self.sol_matrix[item_idx][weight:] = value
                item_idx = 1
            else:
                if weight > cap:
                    self.sol_matrix[item_idx][:weight] = self.sol_matrix[item_idx-1][:weight]
                    cap = weight
                else:
                    # value of taking this item
                    val = value + self.sol_matrix[item_idx-1][cap-weight]
                    # value of not taking this item
                    alt_val = self.sol_matrix[item_idx-1][cap]
                    self.sol_matrix[item_idx][cap] = max(val, alt_val)
                    cap += 1
                if cap > self.capacity:
                    item_idx += 1
                    cap = 0
            if item_idx == self.item_count:
                break
        logger.debug(f'sol_matrix:\n{self.sol_matrix}')

    def solve(self):
        self._populate_sol_matrix()

        # traceback
        item_idx = self.item_count - 1
        cap = self.capacity
        sol_arr = ['0'] * self.item_count
        obj = self.sol_matrix[item_idx][cap]
        while True:
            # not taking the current item
            if self.sol_matrix[item_idx][cap] == self.sol_matrix[item_idx-1][cap]:
                item_idx -= 1
            # navigate up
            else:
                weight = self.items[item_idx].weight
                sol_arr[item_idx] = '1'
                cap -= weight
                item_idx -= 1
            # update item_idx
            if item_idx == 0:
                break
        solution = f'{obj} 1\n{" ".join(sol_arr)}'
        return solution


if __name__ == '__main__':
    file_location = '/Users/zinanj/projects/discrete_optimization/knapsack/data/ks_30_0'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    ks = KnapsackDP(input_data)
    solution = ks.solve()
    print(solution)


