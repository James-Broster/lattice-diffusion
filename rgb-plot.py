import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

# Create a custom RGB plot with custom colors
def create_rgb_plot(intensity_matrices, colors, m, n):
    if len(intensity_matrices) != len(colors):
        raise ValueError("Number of intensity matrices and colors must be the same.")

    # Initialize the final 2m x 2n x 3 (RGB) array
    final_array = np.zeros((2*m, 2*n, 3))

    # For each color, assign its intensity to the appropriate positions
    for idx, (matrix, color) in enumerate(zip(intensity_matrices, colors)):
        # Expand the m x n matrix to 2m x 2n
        expanded_matrix = np.repeat(np.repeat(matrix, 2, axis=1), 2, axis=0)

        # Apply the color to the expanded matrix
        colored_matrix = np.zeros((2*m, 2*n, 3))
        for i in range(3): # RGB channels
            colored_matrix[:, :, i] = expanded_matrix * color[i]

        # Assign to the final array
        if idx == 0: # Top-left
            final_array[0::2, 0::2, :] += colored_matrix[0::2, 0::2, :]
        elif idx == 1: # Top-right
            final_array[0::2, 1::2, :] += colored_matrix[0::2, 1::2, :]
        elif idx == 2: # Bottom-left
            final_array[1::2, 0::2, :] += colored_matrix[1::2, 0::2, :]
        elif idx == 3: # Bottom-right
            final_array[1::2, 1::2, :] += colored_matrix[1::2, 1::2, :]

    # Clip values to be in the range [0, 1]
    final_array = np.clip(final_array, 0, 1)

    # Plot the final array
    plt.imshow(final_array)
    plt.axis('off')
    plt.show()

# Convert a HEX code to Matplotlib RGB
def hex_to_rgb(hex_code, rgb01_o=True):
    """
    hex_code: str
    The input HEX code for conversion
    
    rgb01_o: Boolean
    Outputs the RGB in 0 to 1 range for matplotlib if True
    """

    h = hex_code.lstrip('#')
    rgb =  tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    if rgb01_o is True:
        rgb01 = tuple(np.array(rgb) / 255)
        return rgb01
        
    else:
        return rgb


# Define custom colors as RGB tuples
hex_colors_list = ['#00BCFA', '#143CC7', '#DB4300', '#710BDB'] # Aqua, Blue, RedOrange, Purple # #00DBC1 
custom_colors = [hex_to_rgb(hex_code) for hex_code in hex_colors_list]
#custom_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)] # Red, Green, Blue, Yellow

# Function to generate random intensity matrices
def generate_random_matrices(m, n):
    return [np.random.rand(m, n) for _ in range(4)]

# Generate random matrices for m = 2, n = 3
m, n = 200, 200
random_intensity_matrices = generate_random_matrices(m, n)

# Create and display the RGB plot with custom colors
create_rgb_plot(random_intensity_matrices, custom_colors, m, n)