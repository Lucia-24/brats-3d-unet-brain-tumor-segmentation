import numpy as np

def random_flip(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly flip along one spatial axis.

    X shape: (C, X, Y, Z)
    y shape: (X, Y, Z)
    """
    axes = [
        (1, 0),  # X axis
        (2, 1),  # Y axis
        (3, 2),  # Z axis
    ]

    x_axis, y_axis = axes[np.random.randint(0, len(axes))]

    X = np.flip(X, axis=x_axis).copy()
    y = np.flip(y, axis=y_axis).copy()

    return X, y

def random_rotate_90(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate by 0, 90, 180, or 270 degrees
    in one of the anatomical planes.

    X shape: (C, X, Y, Z)
    y shape: (X, Y, Z)
    """
    k = np.random.randint(0, 4)

    plane_options = [
        ((1, 2), (0, 1)),  # X-Y plane
        ((1, 3), (0, 2)),  # X-Z plane
        ((2, 3), (1, 2)),  # Y-Z plane
    ]

    x_axes, y_axes = plane_options[np.random.randint(0, len(plane_options))]

    X = np.rot90(X, k=k, axes=x_axes).copy()
    y = np.rot90(y, k=k, axes=y_axes).copy()

    return X, y


def intensity_shift(X: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    shift = np.float32(np.random.uniform(-shift_range, shift_range))
    return (X + shift).astype(np.float32)


def gaussian_noise(X: np.ndarray, std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, std, X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def augment_patch(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation pipeline.

    Spatial transforms (X and y):
        - random flip
        - random 90-degree rotation

    Intensity transforms (X only):
        - intensity shift
        - Gaussian noise
    """

    # Spatial augmentations
    if np.random.rand() < 0.5:
        X, y = random_flip(X, y)

    if np.random.rand() < 0.5:
        X, y = random_rotate_90(X, y)

    # Intensity augmentations
    if np.random.rand() < 0.4:
        X = intensity_shift(X)

    if np.random.rand() < 0.4:
        X = gaussian_noise(X)

    return X.astype(np.float32), y.astype(np.float32)