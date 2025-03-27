import pickle

def load_and_print_transformation_matrix(filename="transformation_matrix_2d.pkl"):
    # Open the pickle file and load the contents
    with open(filename, "rb") as f:
        transformation_matrix = pickle.load(f)
    
    # Print the contents of the transformation matrix
    print("Transformation Matrix:")
    print(transformation_matrix)
    
    # Optionally, you can print the shape to confirm the dimensions
    import numpy as np
    matrix = np.array(transformation_matrix)
    print("Shape of the transformation matrix:", matrix.shape)

# Call the function to load and print the transformation matrix
load_and_print_transformation_matrix()
