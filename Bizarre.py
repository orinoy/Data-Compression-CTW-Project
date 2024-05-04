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

# Function to generate a Bernoulli sequence with high probability of one symbol
def generate_bernoulli_sequence(length, theta):
    return [1 if random.random() < theta else 0 for _ in range(length)]

# Function to generate a sequence where the next symbol depends only on the 8th previous symbol
def generate_dependent_sequence(length, dependency_index):
    sequence = [random.randint(0, 1) for _ in range(length)]
    for i in range(dependency_index, length):
        sequence[i] = sequence[i - dependency_index]
    return sequence

max_sequence_length = 100
sequence_lengths = list(range(1, max_sequence_length + 1))

# Generate "bizarre" sequences and encode using different methods
entropy_lengths_bernoulli = []
ctw_lengths_bernoulli = []
mdl_lengths_bernoulli = []

entropy_lengths_dependent = []
ctw_lengths_dependent = []
mdl_lengths_dependent = []

# Bernoulli sequence with high probability of one symbol
theta = 0.99
bernoulli_sequence = generate_bernoulli_sequence(max_sequence_length, theta)
for length in sequence_lengths:
    entropy_lengths_bernoulli.append(entropy([theta, 1 - theta]))
    ctw_encoded_sequence = ctw_coding(bernoulli_sequence[:length])
    ctw_lengths_bernoulli.append(ctw_encoded_sequence[-1])  # Use the final length of CTW encoding
    mdl_encoded_sequence = mdl_coding(bernoulli_sequence[:length])
    mdl_lengths_bernoulli.append(mdl_encoded_sequence[-1])  # Use the final length of MDL encoding

# Dependent sequence where the next symbol depends only on the 8th previous symbol
dependency_index = 8
dependent_sequence = generate_dependent_sequence(max_sequence_length, dependency_index)
for length in sequence_lengths:
    entropy_lengths_dependent.append(entropy([0.5, 0.5]))
    ctw_encoded_sequence = ctw_coding(dependent_sequence[:length])
    ctw_lengths_dependent.append(ctw_encoded_sequence[-1])  # Use the final length of CTW encoding
    mdl_encoded_sequence = mdl_coding(dependent_sequence[:length])
    mdl_lengths_dependent.append(mdl_encoded_sequence[-1])  # Use the final length of MDL encoding

# Plotting
plt.plot(sequence_lengths, entropy_lengths_bernoulli, label='Entropy Coding (Bernoulli)')
plt.plot(sequence_lengths, ctw_lengths_bernoulli, label='CTW Coding (Bernoulli)')
plt.plot(sequence_lengths, mdl_lengths_bernoulli, label='MDL Coding (Bernoulli)')

plt.plot(sequence_lengths, entropy_lengths_dependent, label='Entropy Coding (Dependent)')
plt.plot(sequence_lengths, ctw_lengths_dependent, label='CTW Coding (Dependent)')
plt.plot(sequence_lengths, mdl_lengths_dependent, label='MDL Coding (Dependent)')

plt.xlabel('Sequence Length')
plt.ylabel('Encoded Length')
plt.title('Encoded length of Extreme disributions a function of Sequence length')
plt.legend()
plt.show()
