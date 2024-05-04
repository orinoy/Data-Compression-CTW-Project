import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

class ContextTree:
    def __init__(self, depth):
        self.depth = depth
        self.tree = {}
        self.context = []

    def update(self, symbol):
        context = tuple(self.context[-self.depth:])
        node = self.tree
        for symbol in context:
            if symbol not in node:
                node[symbol] = {}
            node = node[symbol]
        if 'counts' not in node:
            node['counts'] = {0: 0, 1: 0}
        node['counts'][symbol] += 1
        if 'children' not in node:
            node['children'] = {}
        self.context.append(symbol)
    def predict(self):
        context = tuple(self.context[-self.depth:])
        node = self.tree
        for symbol in context:
            if symbol not in node:
                return 0.5  # Default probability if context not found
            node = node[symbol]['children']
        if 'counts' not in node:
            return 0.5  # Default probability if counts not found
        total_count = sum(node['counts'].values())
        if total_count == 0:
            return 0.5
        return node['counts'][1] / total_count



def calculate_redundancy(training_data, max_depth):
    redundancies = []
    for length in range(2, len(training_data) + 1):
        data = training_data[:length]
        model = ContextTree(min(max_depth, length - 1))
        for symbol in data:
            model.update(symbol)
        predicted_data = [model.predict() for _ in range(length)]
        entropy_data = entropy([np.mean(data[:i+1]) for i in range(length)])
        entropy_model = entropy(predicted_data)
        redundancy = entropy_data - entropy_model
        redundancies.append(redundancy)
    return redundancies

def main():
    num_trials = 10
    max_sequence_length = 100
    max_depth = 10

    redundancies = np.zeros((max_depth, max_sequence_length - 1))
    for _ in range(num_trials):
        training_data = np.random.randint(2, size=max_sequence_length)
        for depth in range(1, max_depth + 1):
            redundancies[depth - 1] += calculate_redundancy(training_data, depth)

    redundancies /= num_trials

    plt.figure(figsize=(10, 6))
    for depth in range(max_depth):
        plt.plot(range(2, max_sequence_length + 1), redundancies[depth])
    plt.title('Redundancy vs. Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Redundancy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
