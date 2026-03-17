from pathlib import Path
import numpy as np

def main():
    output_dir = Path("processed_patients")

    patient_ids = np.load(output_dir / "patient_ids.npy", allow_pickle=True)

    x_shapes = []
    y_shapes = []

    for patient_id in patient_ids:
        X = np.load(output_dir / f"{patient_id}_X.npy")
        y = np.load(output_dir / f"{patient_id}_y.npy")

        x_shapes.append(X.shape)
        y_shapes.append(y.shape)

    print("Number of patients:", len(patient_ids))

    print("\nFirst 5 X shapes:")
    for shape in x_shapes[:5]:
        print(shape)

    print("\nFirst 5 y shapes:")
    for shape in y_shapes[:5]:
        print(shape)

    x_spatial = np.array([shape[1:] for shape in x_shapes])
    y_spatial = np.array(y_shapes)

    print("\nX spatial shape summary:")
    print("Min:", x_spatial.min(axis=0))
    print("Max:", x_spatial.max(axis=0))
    print("Mean:", x_spatial.mean(axis=0))

    print("\ny spatial shape summary:")
    print("Min:", y_spatial.min(axis=0))
    print("Max:", y_spatial.max(axis=0))
    print("Mean:", y_spatial.mean(axis=0))

    unique_x_shapes, counts = np.unique(x_shapes, axis=0, return_counts=True)

    print("\nMost common X shapes:")
    sorted_idx = np.argsort(counts)[::-1]

    for i in sorted_idx[:10]:
        print(unique_x_shapes[i], "count =", counts[i])


if __name__ == "__main__":
    main()