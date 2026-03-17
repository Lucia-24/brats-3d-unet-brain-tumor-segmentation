from pathlib import Path
import kagglehub
import numpy as np
import nibabel as nib


def open_path():
    path = Path(
        kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    )
    print("Path to dataset files:", path)
    return path

def get_training_path(path):
    training_path = path / "BraTs2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    print("Training path:", training_path)
    return training_path

def collect_patient_data(training_path):
    patients = []

    for folder in training_path.iterdir():
        if folder.is_dir():
            patient_id = folder.name

            t1_files = list(folder.glob("*_t1.nii"))
            t1ce_files = list(folder.glob("*_t1ce.nii"))
            t2_files = list(folder.glob("*_t2.nii"))
            flair_files = list(folder.glob("*_flair.nii"))
            seg_files = list(folder.glob("*_seg.nii"))

            patients.append({
                "patient_id": patient_id,
                "folder": folder,
                "t1": t1_files[0] if t1_files else None,
                "t1ce": t1ce_files[0] if t1ce_files else None,
                "t2": t2_files[0] if t2_files else None,
                "flair": flair_files[0] if flair_files else None,
                "seg": seg_files[0] if seg_files else None
            })

    print("Number of patient folders found:", len(patients))
    return patients

def load_nifti_file(file_path):
    if file_path is None:
        return None

    img = nib.load(file_path)
    data = np.array(img.get_fdata())
    return data

def preprocess_patient(patient):
    t1_data = load_nifti_file(patient["t1"])
    t1ce_data = load_nifti_file(patient["t1ce"])
    t2_data = load_nifti_file(patient["t2"])
    flair_data = load_nifti_file(patient["flair"])
    seg_data = load_nifti_file(patient["seg"])

    if any(x is None for x in [t1_data, t1ce_data, t2_data, flair_data, seg_data]):
        print(f"Skipping {patient['patient_id']} because one or more files are missing.")
        return None, None

    scans = [flair_data.copy(), t1_data.copy(), t1ce_data.copy(), t2_data.copy()]

    for scan in scans:
        scan[scan < 10] = 0

    scans = np.stack(scans, axis=0).astype(np.float32)

    brain_mask = np.any(scans > 0, axis=0)
    nonzero_coords = np.argwhere(brain_mask)

    if nonzero_coords.size == 0:
        print(f"Skipping {patient['patient_id']} because mask is empty.")
        return None, None

    x_min, y_min, z_min = nonzero_coords.min(axis=0)
    x_max, y_max, z_max = nonzero_coords.max(axis=0)

    cropped_scans = scans[:, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    seg_cropped = seg_data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1].astype(np.uint8)

    for i in range(cropped_scans.shape[0]):
        scan = cropped_scans[i]
        nonzero = scan > 0

        if np.any(nonzero):
            mean = scan[nonzero].mean()
            std = scan[nonzero].std()

            if std > 0:
                cropped_scans[i][nonzero] = (scan[nonzero] - mean) / std

    return cropped_scans, seg_cropped

def main():
    data_path = open_path()
    training_path = get_training_path(data_path)
    patients = collect_patient_data(training_path)

    output_dir = Path("processed_patients")
    output_dir.mkdir(exist_ok=True)

    processed_ids = []

    for patient in patients:
        X_patient, y_patient = preprocess_patient(patient)

        if X_patient is not None and y_patient is not None:
            patient_id = patient["patient_id"]

            np.save(output_dir / f"{patient_id}_X.npy", X_patient)
            np.save(output_dir / f"{patient_id}_y.npy", y_patient)

            processed_ids.append(patient_id)

            print(f"{patient_id} -> X: {X_patient.shape}, y: {y_patient.shape}")

    np.save(output_dir / "patient_ids.npy", np.array(processed_ids, dtype=object), allow_pickle=True)
    print("\nTotal processed patients:", len(processed_ids))
    print(f"Saved per-patient files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()