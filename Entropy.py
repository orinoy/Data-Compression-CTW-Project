import random
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the entropy of a probability distribution
def entropy(probabilities):
    return -sum(p * np.log2(p) for p in probabilities if p != 0)

# Function to generate a random binary sequence of a given length
def generate_random_sequence(length):
    return [random.randint(0, 1) for _ in range(length)]

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

# Generate sequences of increasing lengths and encode using different methods
max_sequence_length = 100
sequence_lengths = list(range(1, max_sequence_length + 1))

entropy_lengths = []
ctw_lengths = []
mdl_lengths = []

for length in sequence_lengths:
    binary_sequence = generate_random_sequence(length)
    p0 = binary_sequence.count(0) / length
    p1 = binary_sequence.count(1) / length
    entropy_lengths.append(entropy([p0, p1]))
    ctw_encoded_lengths = ctw_coding(binary_sequence)
    ctw_lengths.append(ctw_encoded_lengths[-1])  # Use the final length of CTW encoding
    mdl_encoded_lengths = mdl_coding(binary_sequence)
    mdl_lengths.append(mdl_encoded_lengths[-1])  # Use the final length of MDL encoding

# Plotting
plt.plot(sequence_lengths, entropy_lengths, label='Entropy Coding')
plt.plot(sequence_lengths, ctw_lengths, label='CTW Coding')
plt.plot(sequence_lengths, mdl_lengths, label='MDL Coding')
plt.xlabel('Sequence Length')
plt.ylabel('Encoded Length')
plt.title('Comparison of Coding Methods vs. Sequence Length')
plt.legend()
plt.show()
