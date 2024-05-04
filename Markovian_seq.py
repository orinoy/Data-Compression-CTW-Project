import random
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the entropy of a probability distribution
def entropy(probabilities):
    return -sum(p * np.log2(p) for p in probabilities if p != 0)

# Function to generate a random Markov sequence of a given order and length
def generate_markov_sequence(order, length):
    states = [0, 1]
    transition_probabilities = {state: {next_state: random.random() for next_state in states} for state in states}
    sequence = []
    state = random.choice(states)
    for _ in range(length):
        sequence.append(state)
        next_state = random.choices(states, [transition_probabilities[state][next_state] for next_state in states])[0]
        state = next_state
    return sequence

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

# Generate Markov sequences of different orders and encode using different methods
max_sequence_length = 100
sequence_lengths = list(range(1, max_sequence_length + 1))
orders = [1, 2, 3]  # Different orders to test

entropy_lengths = [[] for _ in orders]
ctw_lengths = [[] for _ in orders]
mdl_lengths = [[] for _ in orders]

for order in orders:
    for length in sequence_lengths:
        markov_sequence = generate_markov_sequence(order, length)
        p0 = markov_sequence.count(0) / length
        p1 = markov_sequence.count(1) / length
        entropy_lengths[order - 1].append(entropy([p0, p1]))
        ctw_encoded_lengths = ctw_coding(markov_sequence)
        ctw_lengths[order - 1].append(ctw_encoded_lengths[-1])  # Use the final length of CTW encoding
        mdl_encoded_lengths = mdl_coding(markov_sequence)
        mdl_lengths[order - 1].append(mdl_encoded_lengths[-1])  # Use the final length of MDL encoding

# Plotting
plt.figure(figsize=(12, 6))
for i, order in enumerate(orders):
    plt.plot(sequence_lengths, entropy_lengths[i], label=f'Entropy (Order {order})', linestyle='--')
    plt.plot(sequence_lengths, ctw_lengths[i], label=f'CTW (Order {order})', linestyle='-')
    plt.plot(sequence_lengths, mdl_lengths[i], label=f'MDL (Order {order})', linestyle='-.')

plt.xlabel('Sequence Length')
plt.ylabel('Encoded Length')
plt.title('Markov Sequences encoded length as a function of Sequence length')
plt.legend()
plt.grid(True)
plt.show()
