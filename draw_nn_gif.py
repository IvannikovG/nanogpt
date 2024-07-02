import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# Define the neural network architecture
layer_sizes = [4, 3, 3, 2]  # 4 input, 2 hidden layers with 3 neurons each, 2 output
num_layers = len(layer_sizes)

# Initialize weights and biases with larger initial values
np.random.seed(0)
weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * 2 for i in range(1, num_layers)]
biases = [np.random.randn(layer_sizes[i], 1) * 2 for i in range(1, num_layers)]

# Predefined weight adjustments for each iteration
weight_adjustments = [
    10, 5, 2, 1.4, 1.2, 1.3, 1.222, 1.1, 1.05, 1.02
]

# Input data and target
X = np.array([[0.1, 0.2, 0.3, 0.4]])
y = np.array([[0.8, 0.9]])

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass function
def forward_pass(X, weights, biases):
    activations = [X.T]  # List to store activations of each layer

    # Iterate through each layer
    for i in range(num_layers - 1):
        z = np.dot(weights[i], activations[-1]) + biases[i]
        a = sigmoid(z)
        activations.append(a)

    return activations


def manual_adjust_weights(weights, iteration):
    min_factor = 0.5  # Minimum factor to prevent weights from going too small
    adjustment_factor = max(weight_adjustments[iteration], min_factor)
    for i in range(num_layers - 1):
        weights[i] = weights[i] * (2 / adjustment_factor)



# Function to visualize current state of the neural network
def visualize_network(weights, biases, activations, iteration):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # Adjusted figure size and subplots

    # Compute positions for neurons
    max_neurons = max(layer_sizes)
    layer_spacing = 2.0
    neuron_spacing = 1.5
    y_positions_all_layers = []

    for layer in range(num_layers):
        num_neurons = layer_sizes[layer]
        y_positions = np.linspace(-(max_neurons - 1) / 2, (max_neurons - 1) / 2, num_neurons)
        y_positions_all_layers.append(y_positions)

        for neuron in range(num_neurons):
            neuron_x = layer * layer_spacing
            neuron_y = y_positions[neuron] * neuron_spacing

            # Draw neurons
            axes[0].add_patch(plt.Circle((neuron_x, neuron_y), radius=0.3, color='lightgreen', ec='black'))
            axes[0].text(neuron_x, neuron_y, f'{activations[layer][neuron, 0]:.2f}', ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

            # Draw connections
            if layer > 0:
                prev_y_positions = y_positions_all_layers[layer - 1]
                for prev_neuron in range(layer_sizes[layer - 1]):
                    prev_y = prev_y_positions[prev_neuron] * neuron_spacing
                    weight = weights[layer - 1][neuron, prev_neuron]
                    color = 'blue' if weight > 0 else 'red'
                    axes[0].plot([neuron_x - layer_spacing, neuron_x], [prev_y, neuron_y], color=color, lw=2 * np.abs(weight), alpha=0.5)

    axes[0].set_xlim(-layer_spacing, (num_layers - 1) * layer_spacing + layer_spacing)
    axes[0].set_ylim(-(max_neurons / 2) * neuron_spacing, (max_neurons / 2) * neuron_spacing)
    axes[0].axis('off')
    axes[0].set_title(f'Iteration {iteration + 1}')

    # Plot current weights state in a table format
    axes_weights = axes[1]
    axes_weights.axis('off')
    weight_table_data = []

    for i, layer_weights in enumerate(weights):
        for j in range(layer_weights.shape[0]):
            row = [f'Layer {i + 1} Neuron {j + 1}']
            row.extend([f'{layer_weights[j, k]:.2f}' for k in range(layer_weights.shape[1])])
            weight_table_data.append(row)

    max_weights = max(len(row) for row in weight_table_data) - 1
    columns = ['Neuron'] + [f'Weight {k + 1}' for k in range(max_weights)]
    weight_table_data = [row + [''] * (max_weights + 1 - len(row)) for row in weight_table_data]

    table = axes_weights.table(cellText=weight_table_data, colLabels=columns, loc='center', cellLoc='center', cellColours=[['lightgray' if i % 2 == 0 else 'white' for _ in range(len(columns))] for i in range(len(weight_table_data))])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the figure as PNG
    plt.tight_layout()  # Ensure tight layout
    plt.savefig(f'iteration_{iteration}.png')
    plt.close()

# Training loop
num_iterations = 10  # Reduced number of iterations for demonstration purposes
frames = []
for iteration in range(num_iterations):
    # Forward pass
    activations = forward_pass(X, weights, biases)

    # Manually adjust weights
    manual_adjust_weights(weights, iteration)

    # Visualize and save current state of the neural network
    visualize_network(weights, biases, activations, iteration)

    # Append image to frames list for creating GIF
    frames.append(imageio.imread(f'iteration_{iteration}.png'))

# Save frames as a GIF
imageio.mimsave('training_animation_fixed.gif', frames, fps=2)

# Clean up: Remove temporary images
for i in range(num_iterations):
    os.remove(f'iteration_{i}.png')

print("GIF animation created successfully.")
