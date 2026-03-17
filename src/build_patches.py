from pathlib import Path
import numpy as np

PATCH_SIZE = (128, 128, 128)

def to_binary_mask(y):
    """
    Convert BraTS labels to binary:
    0 = background
    1 = any tumor-related region
    """
    return (y > 0).astype(np.uint8)

def pad_to_min_size(X, y, target_shape):
    """
    Pad X and y so that spatial dimensions are at least target_shape.
    
    X shape: (C, X, Y, Z)
    y shape: (X, Y, Z)
    """
    _, x, y_dim, z = X.shape
    tx, ty, tz = target_shape

    pad_x = max(0, tx - x)
    pad_y = max(0, ty - y_dim)
    pad_z = max(0, tz - z)

    # split padding before/after
    px1, px2 = pad_x // 2, pad_x - pad_x // 2
    py1, py2 = pad_y // 2, pad_y - pad_y // 2
    pz1, pz2 = pad_z // 2, pad_z - pad_z // 2
    
    # Numpy np.pad() is a function that adds values around the edges of an array
    X_padded = np.pad(
        X,
        ((0, 0), (px1, px2), (py1, py2), (pz1, pz2)),
        mode="constant",
        constant_values=0
    )

    y_padded = np.pad(
        y,
        ((px1, px2), (py1, py2), (pz1, pz2)),
        mode="constant",
        constant_values=0
    )

    return X_padded, y_padded

def center_crop_patch(X, y, patch_size):
    """
    Extract a centered patch of size patch_size.
    
    X shape: (C, X, Y, Z)
    y shape: (X, Y, Z)
    """
    _, x, y_dim, z = X.shape
    px, py, pz = patch_size

    start_x = (x - px) // 2
    start_y = (y_dim - py) // 2
    start_z = (z - pz) // 2

    X_patch = X[:, start_x:start_x+px, start_y:start_y+py, start_z:start_z+pz]
    y_patch = y[start_x:start_x+px, start_y:start_y+py, start_z:start_z+pz]

    return X_patch, y_patch

def main():
    processed_dir = Path("processed_patients")
    output_dir = Path("patches_binary")
    output_dir.mkdir(exist_ok=True)

    patient_ids = np.load(processed_dir / "patient_ids.npy", allow_pickle=True)

    saved_ids = []

    for patient_id in patient_ids:
        X = np.load(processed_dir / f"{patient_id}_X.npy")
        y = np.load(processed_dir / f"{patient_id}_y.npy")

        # convert segmentation to binary
        y_binary = to_binary_mask(y)

        X_padded, y_padded = pad_to_min_size(X, y_binary, PATCH_SIZE)

        # extract centered patch
        X_patch, y_patch = center_crop_patch(X_padded, y_padded, PATCH_SIZE)

        np.save(output_dir / f"{patient_id}_X_patch.npy", X_patch.astype(np.float32))
        np.save(output_dir / f"{patient_id}_y_patch.npy", y_patch.astype(np.uint8))

        saved_ids.append(patient_id)

        print(f"{patient_id} -> X patch: {X_patch.shape}, y patch: {y_patch.shape}")

    np.save(output_dir / "patient_ids.npy", np.array(saved_ids, dtype=object), allow_pickle=True)

    print("\nTotal patch patients saved:", len(saved_ids))
    print(f"Saved binary patches to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()