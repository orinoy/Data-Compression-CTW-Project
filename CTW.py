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
        for context_symbol in context:
            if context_symbol not in node:
                node[context_symbol] = {'counts': {0: 0, 1: 0}, 'children': {}}
            node = node[context_symbol]
        if 'counts' not in node:
            node['counts'] = {0: 0, 1: 0}
            node['counts'][symbol] = 0
        node['counts'][symbol] += 1
        if 'children' not in node:
            node['children'] = {}
        self.context.append(symbol)

    def predict(self):
        context = tuple(self.context[-self.depth:])
        node = self.tree
        for context_symbol in context:
            if context_symbol not in node:
                return 0.5  # Default probability if context not found
            node = node[context_symbol].get('children', {})
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

def generate_markov_sequence(transition_matrix, initial_state, sequence_length):
    sequence = [initial_state]
    current_state = initial_state
    for _ in range(sequence_length - 1):
        next_state_probs = transition_matrix[current_state]
        next_state = np.random.choice(len(next_state_probs), p=next_state_probs)
        sequence.append(next_state)
        current_state = next_state
    return sequence

def main():
    num_trials = 1000
    sequence_length = 50

    orders = range(2, 11)  # Orders of Markov model to evaluate
    redundancies = np.zeros((len(orders), sequence_length - 1))
    
    for order_idx, order in enumerate(orders):
        # Define transition matrix for Markov sequence of given order
        transition_matrix = np.random.rand(2**order, 2)
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        
        for _ in range(num_trials):
            # Generate Markovian sequence
            initial_state = np.random.randint(2**order)
            markov_sequence = generate_markov_sequence(transition_matrix, initial_state, sequence_length)
            
            # Calculate redundancy for the sequence
            redundancies[order_idx] += calculate_redundancy(markov_sequence, order)
    
    # Average redundancies over trials
    redundancies /= num_trials
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for order_idx, order in enumerate(orders):
        plt.plot(range(2, sequence_length + 1), redundancies[order_idx], label=f'Order {order}')
    plt.title('Redundancy vs. Sequence Length for Different Markov Orders')
    plt.xlabel('Sequence Length')
    plt.ylabel('Redundancy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
