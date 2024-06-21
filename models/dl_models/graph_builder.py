
import torch
import numpy as np
from scipy.spatial.distance import cdist
from haversine import haversine, Unit


def haversine_distance(coords1, coords2):
    """
    Calculate the Haversine distance between coordinates on Earth's surface.

    Parameters:
    - coords1: 2D numpy array, coordinates (latitude, longitude) of
    the first set of points
    - coords2: 2D numpy array, coordinates (latitude, longitude) of
    the second set of points

    Returns:
    - distances: 2D numpy array, Haversine distances between the sets
    of coordinates (in kilometers)
    """
    distances = []
    for coord1 in coords1:
        row_distances = []
        for coord2 in coords2:
            distance = haversine(coord1, coord2, unit=Unit.KILOMETERS)
            row_distances.append(distance)
        distances.append(row_distances)
    return np.array(distances)


def construct_adjacency_matrix(
    features,
    threshold,
    distance_matric="euclidean",
    distance_function=None,
    distance_function_args=None
):
    """
    Construct adjacency matrix based on distance between node features.

    Parameters:
    - features: 2D numpy array, node features matrix
    - threshold: float, threshold value for creating edges in adjacency matrix
    - distance_matric: string or none, optional, custom distance function 
    (default: Euclidean distance)
        Supported distance functions from scipy.spatial.distance.cdist:
        - 'euclidean': Euclidean distance
        - 'cityblock': Manhattan distance
        - 'cosine': Cosine distance
        - 'minkowski': Minkowski distance
        - 'correlation': Correlation distance
        - 'hamming': Hamming distance
        - 'jaccard': Jaccard distance
        - and more (refer to SciPy documentation for cdist)
    - distance_function: function or None, custom distance function
    (default: None)
    - distance_function_args: dict or None, arguments for the distance
    function (if needed)

    Returns:
    - adjacency_matrix: torch.sparse.COO, resulting adjacency matrix
    in COO format
    """

    # Check if both distance_matrix and distance_function are None or not None
    # at the same time

    if type(features) is list or type(features) is torch.Tensor:
        features = np.array(features)

    if (distance_matric is None and distance_function is None) or (distance_matric is not None and distance_function is not None):
        raise ValueError("Either distance_matrix or distance_function should be provided, and not both or none.")

    if distance_matric is not None:
        # If distance_matrix is provided, use it to construct adjacency matrix
        if distance_matric == "haversine":
            distances = haversine_distance(features)
        else:
            distances = cdist(features, features,  distance_matric)
    else:
        if distance_function_args is None:
            # Use the specified distance function if distance_matrix is not provided
            distances = distance_function(features)
        else:
            distances = distance_function(features, **distance_function_args)

    # Construct adjacency matrix based on the threshold
    adjacency_matrix = np.where(distances < threshold, 1, 0)
    np.fill_diagonal(adjacency_matrix, 0)  # Set diagonal elements to 0 (nodes do not connect to themselves)

    # Convert adjacency matrix to PyTorch COO format
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to_sparse()

    return adjacency_matrix
