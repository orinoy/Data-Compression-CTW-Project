import random
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the entropy of a probability distribution
def entropy(probabilities):
    return -sum(p * np.log2(p) for p in probabilities if p != 0)

# Function for Context Tree Weighting (CTW) coding
def ctw_coding(sequence):
    history = []
    encoded_sequence = []
    encoded_lengths = []
    for bit in sequence:
        if len(history) == 0:
            p0 = sequence.count(0) / len(sequence)
            p1 = sequence.count(1) / len(sequence)
        else:
            history_str = ''.join(map(str, history))
            p0 = sequence.count(0) / (history_str + '0').count('0')
            p1 = sequence.count(1) / (history_str + '1').count('1')
        encoded_bit = 0 if p0 >= p1 else 1
        encoded_sequence.append(encoded_bit)
        history.append(bit)
        encoded_lengths.append(len(encoded_sequence))
    return encoded_lengths

# Function for Minimum Description Length (MDL) coding
def mdl_coding(sequence):
    encoded_lengths = []
    compressed_length = 0
    for i in range(1, len(sequence) + 1):
        subsequence = sequence[:i]
        subsequence_entropy = entropy([subsequence.count(0) / i, subsequence.count(1) / i])
        compressed_length_i = i * subsequence_entropy
        encoded_lengths.append(compressed_length_i)
    return encoded_lengths

# Function to generate a tree source
def generate_tree_source(length, tree):
    source = []
    current_node = tree
    for _ in range(length):
        parameter = current_node['parameter']
        next_symbol = 0 if random.random() < parameter else 1
        source.append(next_symbol)
        if next_symbol == 0:
            if current_node['left'] is not None:
                current_node = current_node['left']
        else:
            if current_node['right'] is not None:
                current_node = current_node['right']
    return source

# Generate a tree source with a specified tree structure
tree_structure = {
    'parameter': 0.5,
    'left': {'parameter': 0.3, 'left': {'parameter': 0.2, 'left': None, 'right': None}, 'right': None},
    'right': {'parameter': 0.7, 'left': None, 'right': {'parameter': 0.8, 'left': None, 'right': None}}
}

max_sequence_length = 100
sequence_lengths = list(range(1, max_sequence_length + 1))

# Generate tree source sequences of increasing lengths and encode using different methods
entropy_lengths = []
ctw_lengths = []
mdl_lengths = []

for length in sequence_lengths:
    tree_source_sequence = generate_tree_source(length, tree_structure)
    
    # Calculate entropy of the sequence
    p0 = tree_source_sequence.count(0) / length
    p1 = tree_source_sequence.count(1) / length
    entropy_lengths.append(entropy([p0, p1]))
    
    # Encode the sequence using CTW coding
    ctw_encoded_sequence = ctw_coding(tree_source_sequence)
    ctw_lengths.append(ctw_encoded_sequence[-1])  # Use the final length of CTW encoding
    
    # Encode the sequence using MDL coding
    mdl_encoded_sequence = mdl_coding(tree_source_sequence)
    mdl_lengths.append(mdl_encoded_sequence[-1])  # Use the final length of MDL encoding

# Plotting
plt.plot(sequence_lengths, entropy_lengths, label='Entropy Coding')
plt.plot(sequence_lengths, ctw_lengths, label='CTW Coding')
plt.plot(sequence_lengths, mdl_lengths, label='MDL Coding')
plt.xlabel('Sequence Length')
plt.ylabel('Encoded Length')
plt.title('Tree Sources; Context-dependent memory depth for different lengths')
plt.legend()
plt.show()
