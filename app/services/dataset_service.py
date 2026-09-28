import os
import zipfile
import json
import uuid
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

def extract_zip_securely(zip_path, extract_to):
    """
    Ekstraksi ZIP dengan proteksi Zip-Slip (PRD Section 7.3).
    Mencegah file keluar dari direktori tujuan ekstraksi.
    """
    os.makedirs(extract_to, exist_ok=True)
    real_extract_to = os.path.realpath(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as archive:
        for member in archive.infolist():
            target_path = os.path.realpath(os.path.join(extract_to, member.filename))
            if not target_path.startswith(real_extract_to + os.sep) and target_path != real_extract_to:
                raise ValueError(f"Percobaan Zip-Slip terdeteksi pada file: {member.filename}")
        archive.extractall(extract_to)

def find_yaml_files(root_dir):
    """
    Mencari file konfigurasi YAML dalam dataset (PRD Section 8).
    Prioritas utama: data.yaml, dataset.yaml, kemudian *.yaml lain.
    """
    found_yamls = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(('.yaml', '.yml')):
                full_p = os.path.normpath(os.path.join(root, f))
                found_yamls.append(full_p)

    def sort_key(p):
        fname = os.path.basename(p).lower()
        if fname == 'data.yaml':
            return 0
        if fname == 'dataset.yaml':
            return 1
        return 2

    found_yamls.sort(key=sort_key)
    return found_yamls

def auto_fix_yaml_path(yaml_path):
    """
    Memperbaiki nilai 'path' di dalam data.yaml agar selalu mengarah ke direktori
    tempat file data.yaml dan folder images/labels berada saat ini.
    Mencegah error Ultralytics 'missing path E:\...' akibat path hardcoded dari PC lain.
    """
    if not os.path.exists(yaml_path):
        return

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return

        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        current_path = data.get('path')
        needs_fix = False

        if not current_path:
            needs_fix = True
        elif not os.path.exists(current_path):
            needs_fix = True
        else:
            train_folder = data.get('train', 'images/train')
            full_train = os.path.join(current_path, train_folder)
            if not os.path.exists(full_train):
                needs_fix = True

        if needs_fix:
            data['path'] = yaml_dir.replace('\\', '/')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)
            print(f"[DatasetService] Auto-fixed YAML path to: {data['path']}")
    except Exception as e:
        print(f"[DatasetService Notice] Gagal auto-fix YAML path: {e}")

def parse_yaml_safe(yaml_path):
    """
    Parsing YAML menggunakan yaml.safe_load (PRD Section 9).
    """
    auto_fix_yaml_path(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data or {}

def normalize_class_names(names_data):
    """
    Menyeragamkan struktur class names dari list atau dict ke dict {id: name}.
    """
    result = {}
    if isinstance(names_data, list):
        for i, name in enumerate(names_data):
            result[i] = str(name)
    elif isinstance(names_data, dict):
        for k, v in names_data.items():
            result[int(k)] = str(v)
    return result

def validate_dataset_structure(dataset_dir, yaml_path):
    """
    Validasi menyeluruh dataset YOLO (PRD Section 11):
    - Cek eksistensi folder images train & val
    - Hitung jumlah file gambar
    - Validasi format dan nilai bounding box (0.0 <= value <= 1.0)
    - Cek gambar rusak (corrupt)
    """
    report = {
        "status": "Valid",
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "num_classes": 0,
        "class_names": {},
        "invalid_labels": 0,
        "missing_labels": 0,
        "corrupt_images": 0,
        "errors": [],
        "warnings": []
    }

    if not os.path.exists(yaml_path):
        report["status"] = "Invalid"
        report["errors"].append("File YAML tidak ditemukan.")
        return report

    try:
        data = parse_yaml_safe(yaml_path)
    except Exception as e:
        report["status"] = "Invalid"
        report["errors"].append(f"Gagal membaca YAML: {str(e)}")
        return report

    # 1. Validasi Class Names
    names_raw = data.get('names', {})
    classes_dict = normalize_class_names(names_raw)
    report["class_names"] = classes_dict
    report["num_classes"] = len(classes_dict)

    if report["num_classes"] == 0:
        report["warnings"].append("Tidak ada daftar 'names' atau class yang terdefinisi di YAML.")

    # 2. Resolusi Path Dataset
    yaml_dir = os.path.dirname(yaml_path)
    base_path = data.get('path', '')
    if base_path:
        if not os.path.isabs(base_path):
            base_dir = os.path.normpath(os.path.join(yaml_dir, base_path))
        else:
            base_dir = base_path
    else:
        base_dir = yaml_dir

    def resolve_folder(folder_key):
        val = data.get(folder_key)
        if not val:
            return None
        if os.path.isabs(val):
            return val
        p1 = os.path.normpath(os.path.join(base_dir, val))
        if os.path.exists(p1):
            return p1
        p2 = os.path.normpath(os.path.join(yaml_dir, val))
        if os.path.exists(p2):
            return p2
        return p1

    train_dir = resolve_folder('train')
    val_dir = resolve_folder('val')
    test_dir = resolve_folder('test')

    if not train_dir or not os.path.exists(train_dir):
        report["status"] = "Invalid"
        report["errors"].append(f"Folder images train tidak ditemukan: {train_dir}")
    if not val_dir or not os.path.exists(val_dir):
        report["status"] = "Invalid"
        report["errors"].append(f"Folder images val tidak ditemukan: {val_dir}")

    if report["status"] == "Invalid":
        return report

    # 3. Hitung dan Validasi Gambar & Label
    def inspect_subset(images_path):
        count = 0
        if not images_path or not os.path.exists(images_path):
            return count

        # Cari folder label pasangan (ganti 'images' dengan 'labels')
        labels_path = None
        norm_img_path = os.path.normpath(images_path)
        if 'images' in norm_img_path:
            possible_lbl = norm_img_path.replace('images', 'labels')
            if os.path.exists(possible_lbl):
                labels_path = possible_lbl

        for root, _, files in os.walk(images_path):
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in IMAGE_EXTENSIONS:
                    count += 1
                    img_file = os.path.join(root, f)

                    # Cek corrupt image (baca header PIL)
                    try:
                        with Image.open(img_file) as im:
                            im.verify()
                    except Exception:
                        report["corrupt_images"] += 1

                    # Cek label
                    if labels_path:
                        base_name = os.path.splitext(f)[0]
                        lbl_file = os.path.join(labels_path, base_name + ".txt")
                        if not os.path.exists(lbl_file):
                            report["missing_labels"] += 1
                        else:
                            try:
                                with open(lbl_file, 'r', encoding='utf-8') as lf:
                                    for line in lf:
                                        parts = line.strip().split()
                                        if not parts:
                                            continue
                                        # Format YOLO standar: cls x y w h
                                        if len(parts) >= 5:
                                            cls_id = int(parts[0])
                                            coords = [float(x) for x in parts[1:5]]
                                            for c in coords:
                                                if c < -0.1 or c > 1.1:
                                                    report["invalid_labels"] += 1
                                                    break
                            except Exception:
                                report["invalid_labels"] += 1
        return count

    report["train_count"] = inspect_subset(train_dir)
    report["val_count"] = inspect_subset(val_dir)
    report["test_count"] = inspect_subset(test_dir) if test_dir else 0

    if report["train_count"] == 0:
        report["status"] = "Invalid"
        report["errors"].append("Tidak ada file gambar pada folder train.")
    if report["val_count"] == 0:
        report["status"] = "Invalid"
        report["errors"].append("Tidak ada file gambar pada folder val.")

    if report["invalid_labels"] > 0 or report["missing_labels"] > 0 or report["corrupt_images"] > 0:
        if report["status"] != "Invalid":
            report["status"] = "Warning"

    return report

def get_dataset_preview_samples(yaml_path, max_samples=12):
    """
    Mengambil sampel gambar dan label untuk grid preview (PRD Section 12).
    """
    samples = []
    if not os.path.exists(yaml_path):
        return samples

    try:
        data = parse_yaml_safe(yaml_path)
    except Exception:
        return samples

    yaml_dir = os.path.dirname(yaml_path)
    base_path = data.get('path', '')
    base_dir = os.path.normpath(os.path.join(yaml_dir, base_path)) if base_path else yaml_dir

    train_val = data.get('train') or data.get('val')
    if not train_val:
        return samples

    train_dir = os.path.normpath(os.path.join(base_dir, train_val))
    if not os.path.exists(train_dir):
        train_dir = os.path.normpath(os.path.join(yaml_dir, train_val))

    if not os.path.exists(train_dir):
        return samples

    labels_dir = train_dir.replace('images', 'labels') if 'images' in train_dir else None
    class_names = normalize_class_names(data.get('names', {}))

    labeled_samples = []
    unlabeled_samples = []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                img_path = os.path.join(root, f)
                boxes = []
                lbl_file = None
                candidate_lbls = []
                if labels_dir:
                    rel_dir = os.path.relpath(root, train_dir)
                    candidate_lbls.append(os.path.normpath(os.path.join(labels_dir, rel_dir, os.path.splitext(f)[0] + ".txt")))
                    candidate_lbls.append(os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))
                candidate_lbls.append(os.path.join(root.replace('images', 'labels'), os.path.splitext(f)[0] + ".txt"))
                candidate_lbls.append(os.path.join(root, os.path.splitext(f)[0] + ".txt"))
                for cl in candidate_lbls:
                    if os.path.exists(cl):
                        lbl_file = cl
                        break

                if lbl_file:
                    try:
                        with open(lbl_file, 'r', encoding='utf-8') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cid = int(parts[0])
                                    cx, cy, w, h = [float(v) for v in parts[1:5]]
                                    boxes.append({
                                        "class_id": cid,
                                        "class_name": class_names.get(cid, str(cid)),
                                        "cx": cx,
                                        "cy": cy,
                                        "w": w,
                                        "h": h
                                    })
                    except Exception:
                        pass

                sample_data = {
                    "filename": f,
                    "filepath": img_path,
                    "boxes": boxes
                }
                if boxes:
                    labeled_samples.append(sample_data)
                    if len(labeled_samples) >= max_samples:
                        return labeled_samples[:max_samples]
                else:
                    if len(unlabeled_samples) < max_samples:
                        unlabeled_samples.append(sample_data)

    combined = labeled_samples + unlabeled_samples
    return combined[:max_samples]
