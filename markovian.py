import numpy as np

def generate_markov_sequence(transition_matrix, initial_state, sequence_length):
    sequence = [initial_state]
    current_state = initial_state
    for _ in range(sequence_length - 1):
        next_state_probs = transition_matrix[current_state]
        next_state = np.random.choice(len(next_state_probs), p=next_state_probs)
        sequence.append(next_state)
        current_state = next_state
    return sequence

# Define transition matrix (example)
transition_matrix = np.array([[0.7, 0.3],  # Transition probabilities from state 0
                              [0.4, 0.6]]) # Transition probabilities from state 1

# Define initial state (example)
initial_state = 0

# Define sequence length
sequence_length = 20

# Generate Markovian sequence
markov_sequence = generate_markov_sequence(transition_matrix, initial_state, sequence_length)
print("Generated Markovian sequence:", markov_sequence)
