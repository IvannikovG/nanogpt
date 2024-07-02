import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from mpl_toolkits.mplot3d import Axes3D


# Create a simplified three-dimensional surface with one large bulge in the center and smaller bulges spiraling around
def true_surface(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.sin(r) / r + 0.3 * np.sin(3 * x) * np.cos(3 * y) + 0.2 * np.sin(5 * x) * np.cos(5 * y)
    return z


# Create an initial rough approximation function for the surface with improved convergence
def rough_approximation(x, y, iteration):
    np.random.seed(42 + iteration)  # For reproducibility
    noise_level = 0.2 / np.log(iteration + 1)  # Increased initial variance
    return true_surface(x, y) + np.random.normal(0, noise_level, x.shape)


# Generate a color gradient based on function values
def color_gradient(Z, cmap, norm):
    colors = cmap(norm(Z))
    return colors


# X and Y values for the surface plot
x = np.linspace(-3, 3, 75)  # Reduced scale
y = np.linspace(-3, 3, 75)  # Reduced scale
X, Y = np.meshgrid(x, y)

# Number of iterations
num_iterations = 20  # Reduced number of iterations

# Generate frames for the GIF
frames = []

# Initial frame (true surface)
fig_init = plt.figure(figsize=(8, 6))
ax_init = fig_init.add_subplot(111, projection='3d')
Z_true = true_surface(X, Y)
surf = ax_init.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.8)
cbar = fig_init.colorbar(surf, ax=ax_init, shrink=0.5)
ax_init.set_title('Initial True Surface', fontsize=16)
ax_init.set_xlabel('X', fontsize=12)
ax_init.set_ylabel('Y', fontsize=12)
ax_init.set_zlabel('Z', fontsize=12)
ax_init.set_xlim(-3, 3)
ax_init.set_ylim(-3, 3)
ax_init.set_zlim(-0.5, 1.5)
filename_init = 'frame_init.png'
plt.savefig(filename_init)
frames.append(filename_init)
plt.close(fig_init)

# Extract color map and normalization from initial plot
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=np.min(Z_true), vmax=np.max(Z_true))

# First prediction frame
fig_first_pred = plt.figure(figsize=(8, 6))
ax_first_pred = fig_first_pred.add_subplot(111, projection='3d')
Z_first_pred = rough_approximation(X, Y, 1)
ax_first_pred.plot_surface(X, Y, Z_first_pred, cmap='viridis', alpha=0.8)
ax_first_pred.set_title('First Rough Approximation', fontsize=16)
ax_first_pred.set_xlabel('X', fontsize=12)
ax_first_pred.set_ylabel('Y', fontsize=12)
ax_first_pred.set_zlabel('Z', fontsize=12)
ax_first_pred.set_xlim(-3, 3)
ax_first_pred.set_ylim(-3, 3)
ax_first_pred.set_zlim(-0.5, 1.5)
filename_first_pred = 'frame_first_pred.png'
plt.savefig(filename_first_pred)
frames.append(filename_first_pred)
plt.close(fig_first_pred)

# Iteration frames
for iteration in range(2, num_iterations + 1):
    Z_approx = rough_approximation(X, Y, iteration)

    fig_iter = plt.figure(figsize=(8, 6))
    ax_iter = fig_iter.add_subplot(111, projection='3d')
    ax_iter.plot_surface(X, Y, Z_approx, cmap='viridis', alpha=0.8)
    ax_iter.set_title(f'Iteration {iteration}', fontsize=16)
    ax_iter.set_xlabel('X', fontsize=12)
    ax_iter.set_ylabel('Y', fontsize=12)
    ax_iter.set_zlabel('Z', fontsize=12)
    ax_iter.set_xlim(-3, 3)
    ax_iter.set_ylim(-3, 3)
    ax_iter.set_zlim(-0.5, 1.5)
    filename_iter = f'frame_{iteration}.png'
    plt.savefig(filename_iter)
    frames.append(filename_iter)
    plt.close(fig_iter)

# Final frame (smooth approximation and comparison)
fig_final = plt.figure(figsize=(14, 6))  # Adjusted figure size for better alignment

# Plot initial true surface
ax_init_comp = fig_final.add_subplot(141, projection='3d')
ax_init_comp.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.8)
ax_init_comp.set_title('Initial True Surface', fontsize=16)
ax_init_comp.set_xlabel('X', fontsize=12)
ax_init_comp.set_ylabel('Y', fontsize=12)
ax_init_comp.set_zlabel('Z', fontsize=12)
ax_init_comp.set_xlim(-3, 3)
ax_init_comp.set_ylim(-3, 3)
ax_init_comp.set_zlim(-0.5, 1.5)

# Plot final smooth approximation
ax_final_comp = fig_final.add_subplot(142, projection='3d')
ax_final_comp.plot_surface(X, Y, Z_approx, cmap='viridis', alpha=0.8)
ax_final_comp.set_title('Final Smooth Approximation', fontsize=16)
ax_final_comp.set_xlabel('X', fontsize=12)
ax_final_comp.set_ylabel('Y', fontsize=12)
ax_final_comp.set_zlabel('Z', fontsize=12)
ax_final_comp.set_xlim(-3, 3)
ax_final_comp.set_ylim(-3, 3)
ax_final_comp.set_zlim(-0.5, 1.5)

# Plot first rough approximation
ax_first_pred_comp = fig_final.add_subplot(143, projection='3d')
ax_first_pred_comp.plot_surface(X, Y, Z_first_pred, cmap='viridis', alpha=0.8)
ax_first_pred_comp.set_title('First Rough Approximation', fontsize=16)
ax_first_pred_comp.set_xlabel('X', fontsize=12)
ax_first_pred_comp.set_ylabel('Y', fontsize=12)
ax_first_pred_comp.set_zlabel('Z', fontsize=12)
ax_first_pred_comp.set_xlim(-3, 3)
ax_first_pred_comp.set_ylim(-3, 3)
ax_first_pred_comp.set_zlim(-0.5, 1.5)

# Plot labels for clarity
ax_values = fig_final.add_subplot(144)
ax_values.axis('off')
ax_values.text(0.5, 0.9, f'Initial Value: High Variance', ha='center', fontsize=14)
ax_values.text(0.5, 0.7, f'First Prediction', ha='center', fontsize=14)
ax_values.text(0.5, 0.5, f'Final Value: Smooth Approximation', ha='center', fontsize=14)

filename_final = 'frame_final.png'
plt.savefig(filename_final, bbox_inches='tight')  # Adjusted to fit images closely
frames.append(filename_final)
plt.close(fig_final)

# Create the GIF with adjusted frame duration
gif_filename = 'surface_approximation.gif'
with imageio.get_writer(gif_filename, mode='I', duration=0.2) as writer:
    for filename in frames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f'Generated GIF: {gif_filename}')

# Clean up the image files
for filename in frames:
    os.remove(filename)
