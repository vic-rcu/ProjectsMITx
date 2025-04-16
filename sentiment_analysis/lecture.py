import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def f(x1, x2):
    return x1 + x2 + x1 * x2 + x1**2 + x2**2

# Create a grid of x1, x2 values
x1 = np.linspace(-5, 5, 100)  # Range for x1
x2 = np.linspace(-5, 5, 100)  # Range for x2
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)  # Compute function values

# --- 3D Surface Plot ---
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap="viridis", edgecolor='k', alpha=0.8)
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$f(x_1, x_2)$")
ax1.set_title("3D Surface Plot")

# --- Contour Plot ---
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X1, X2, Z, levels=20, cmap="viridis")
plt.colorbar(contour, ax=ax2)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_title("Contour Plot")

plt.show()