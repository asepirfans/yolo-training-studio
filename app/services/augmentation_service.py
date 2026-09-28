import os
import copy
import json
import math
import random
import shutil
import time
import uuid
import base64
import threading
from datetime import datetime
import cv2
import numpy as np
import yaml
from werkzeug.utils import secure_filename

from app.models import db, Dataset, Project
from app.services.dataset_service import parse_yaml_safe, normalize_class_names

# In-memory background jobs storage
# Format: { job_id: { status: 'running'|'completed'|'failed', progress: int, processed: int, total: int, message: str, new_dataset_id: str, error: str } }
_AUGMENTATION_JOBS = {}
_JOBS_LOCK = threading.Lock()

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def get_job_status(job_id):
    with _JOBS_LOCK:
        return _AUGMENTATION_JOBS.get(job_id)


def update_job_status(job_id, **kwargs):
    with _JOBS_LOCK:
        if job_id in _AUGMENTATION_JOBS:
            _AUGMENTATION_JOBS[job_id].update(kwargs)


# ==============================================================================
# Bounding Box Transformation Helpers (YOLO format: normalized cx, cy, w, h)
# ==============================================================================

def yolo_to_corners(cx, cy, w, h, img_w, img_h):
    """Konversi koordinat YOLO ternormalisasi ke 4 titik sudut piksel."""
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)


def corners_to_yolo(corners, img_w, img_h, min_area=16.0):
    """
    Konversi kembali 4 titik sudut yang telah ditransformasikan ke YOLO (cx, cy, w, h)
    dengan memotong ke batas gambar [0, W] dan [0, H].
    Mengembalikan None jika kotak terpotong habis atau terlalu kecil.
    """
    x_min = np.min(corners[:, 0])
    x_max = np.max(corners[:, 0])
    y_min = np.min(corners[:, 1])
    y_max = np.max(corners[:, 1])

    # Klip ke batas kanvas
    x_min = max(0.0, min(float(img_w), float(x_min)))
    x_max = max(0.0, min(float(img_w), float(x_max)))
    y_min = max(0.0, min(float(img_h), float(y_min)))
    y_max = max(0.0, min(float(img_h), float(y_max)))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w * box_h < min_area or box_w <= 1 or box_h <= 1:
        return None

    cx = (x_min + box_w / 2.0) / float(img_w)
    cy = (y_min + box_h / 2.0) / float(img_h)
    nw = box_w / float(img_w)
    nh = box_h / float(img_h)

    # Validasi rentang 0.0 - 1.0
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


# ==============================================================================
# Single Transformation Functions
# ==============================================================================

def apply_horizontal_flip(image, boxes):
    """Flip citra secara horizontal (kiri-kanan)."""
    h, w = image.shape[:2]
    flipped_img = cv2.flip(image, 1)
    new_boxes = []
    for b in boxes:
        nb = copy.deepcopy(b)
        nb['cx'] = max(0.0, min(1.0, 1.0 - b['cx']))
        new_boxes.append(nb)
    return flipped_img, new_boxes


def apply_vertical_flip(image, boxes):
    """Flip citra secara vertikal (atas-bawah)."""
    h, w = image.shape[:2]
    flipped_img = cv2.flip(image, 0)
    new_boxes = []
    for b in boxes:
        nb = copy.deepcopy(b)
        nb['cy'] = max(0.0, min(1.0, 1.0 - b['cy']))
        new_boxes.append(nb)
    return flipped_img, new_boxes


def apply_rotation(image, boxes, angle_degrees):
    """
    Rotasi citra dan kalkulasi ulang 4 sudut bounding box.
    Menggunakan border constant abu-abu netral standar YOLO (114, 114, 114)
    agar tidak timbul refleksi cermin aneh pada pinggiran citra.
    """
    if abs(angle_degrees) < 0.1:
        return image, boxes

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated_img = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    new_boxes = []
    for b in boxes:
        corners = yolo_to_corners(b['cx'], b['cy'], b['w'], b['h'], w, h)
        # Transformasi sudut: [x, y, 1] * M^T
        ones = np.ones((4, 1), dtype=np.float32)
        corners_homo = np.hstack([corners, ones])
        transformed_corners = corners_homo.dot(rot_mat.T)

        yolo_coords = corners_to_yolo(transformed_corners, w, h)
        if yolo_coords is not None:
            nb = copy.deepcopy(b)
            nb['cx'], nb['cy'], nb['w'], nb['h'] = yolo_coords
            new_boxes.append(nb)

    return rotated_img, new_boxes


def apply_zoom_scale(image, boxes, scale_factor):
    """
    Skala / Zoom In / Zoom Out relatif terhadap pusat citra.
    scale_factor > 1.0 = Zoom In (objek membesar)
    scale_factor < 1.0 = Zoom Out (objek mengecil)
    Menggunakan border constant abu-abu netral standar YOLO (114, 114, 114).
    """
    if abs(scale_factor - 1.0) < 0.02:
        return image, boxes

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    scale_mat = cv2.getRotationMatrix2D(center, 0.0, scale_factor)
    scaled_img = cv2.warpAffine(image, scale_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    new_boxes = []
    for b in boxes:
        corners = yolo_to_corners(b['cx'], b['cy'], b['w'], b['h'], w, h)
        ones = np.ones((4, 1), dtype=np.float32)
        corners_homo = np.hstack([corners, ones])
        transformed_corners = corners_homo.dot(scale_mat.T)

        yolo_coords = corners_to_yolo(transformed_corners, w, h)
        if yolo_coords is not None:
            nb = copy.deepcopy(b)
            nb['cx'], nb['cy'], nb['w'], nb['h'] = yolo_coords
            new_boxes.append(nb)

    return scaled_img, new_boxes


def apply_translation(image, boxes, shift_x_pct, shift_y_pct):
    """
    Pergeseran (Translasi) horizontal & vertikal dalam persentase (-0.15 s.d. 0.15).
    Menggunakan border constant abu-abu netral standar YOLO (114, 114, 114).
    """
    if abs(shift_x_pct) < 0.005 and abs(shift_y_pct) < 0.005:
        return image, boxes

    h, w = image.shape[:2]
    tx = float(shift_x_pct) * w
    ty = float(shift_y_pct) * h

    trans_mat = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    shifted_img = cv2.warpAffine(image, trans_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    new_boxes = []
    for b in boxes:
        corners = yolo_to_corners(b['cx'], b['cy'], b['w'], b['h'], w, h)
        corners[:, 0] += tx
        corners[:, 1] += ty

        yolo_coords = corners_to_yolo(corners, w, h)
        if yolo_coords is not None:
            nb = copy.deepcopy(b)
            nb['cx'], nb['cy'], nb['w'], nb['h'] = yolo_coords
            new_boxes.append(nb)

    return shifted_img, new_boxes


def apply_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    """Penyesuaian kecerahan dan kontras piksel citra."""
    img_f = image.astype(np.float32)
    if contrast_factor != 1.0:
        img_f = (img_f - 128.0) * contrast_factor + 128.0
    if brightness_factor != 1.0:
        img_f = img_f * brightness_factor
    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_saturation(image, sat_factor=1.0):
    """Penyesuaian saturasi warna via ruang warna HSV."""
    if abs(sat_factor - 1.0) < 0.02:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_grayscale(image):
    """Konversi ke citra skala abu-abu (tetap 3 channel BGR)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_gaussian_blur(image, kernel_size=5):
    """Simulasi blur halus / out of focus."""
    k = max(3, int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1)
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_motion_blur(image, kernel_size=7, angle_degrees=0):
    """Simulasi getaran konveyor atau gerakan cepat kamera."""
    k = max(3, int(kernel_size))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[int((k - 1) / 2), :] = np.ones(k, dtype=np.float32)
    # Rotasi kernel sesuai sudut gerakan
    center = ((k - 1) / 2.0, (k - 1) / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (k, k))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)


def apply_sensor_noise(image, sigma=12.0):
    """Simulasi noise sensor kamera industri."""
    if sigma <= 0:
        return image
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_cutout(image, boxes, num_patches=2, patch_size_pct=0.08):
    """
    Menambahkan patch kotak acak (occlusion) untuk melatih ketahanan deteksi
    terhadap objek yang tertutup sebagian. Bounding box tetap valid.
    """
    h, w = image.shape[:2]
    out_img = image.copy()
    pw = max(10, int(w * patch_size_pct))
    ph = max(10, int(h * patch_size_pct))

    for _ in range(num_patches):
        x = random.randint(0, max(0, w - pw))
        y = random.randint(0, max(0, h - ph))
        # Warna patch acak gelap atau abu-abu
        color = [random.randint(15, 45), random.randint(15, 45), random.randint(15, 45)]
        out_img[y:y + ph, x:x + pw] = color

    return out_img, boxes


# ==============================================================================
# Pipeline Augmentasi Utama (Menerapkan Konfigurasi Terpilih)
# ==============================================================================

def augment_single_image(image, boxes, config, return_applied=False):
    """
    Menerapkan pipeline augmentasi pada satu gambar berdasarkan konfigurasi JSON.
    Mengembalikan (augmented_image, transformed_boxes) atau (augmented_image, transformed_boxes, applied_transforms)
    jika return_applied=True.
    """
    img = image.copy()
    bxs = copy.deepcopy(boxes)
    applied = []

    # 1. Flip Horizontal
    if config.get('hflip', False):
        prob = float(config.get('hflip_p', 0.5))
        if random.random() < prob:
            img, bxs = apply_horizontal_flip(img, bxs)
            applied.append("Flip Horizontal")

    # 2. Flip Vertikal
    if config.get('vflip', False):
        prob = float(config.get('vflip_p', 0.5))
        if random.random() < prob:
            img, bxs = apply_vertical_flip(img, bxs)
            applied.append("Flip Vertikal")

    # 3. Rotasi
    if config.get('rotate', False):
        max_deg = float(config.get('rotate_limit', 15.0))
        angle = random.uniform(-max_deg, max_deg)
        if abs(angle) >= 0.5:
            img, bxs = apply_rotation(img, bxs, angle)
            sign = "+" if angle > 0 else ""
            applied.append(f"Rotasi {sign}{angle:.1f}°")

    # 4. Zoom / Scale
    if config.get('scale', False):
        min_scale = float(config.get('scale_min', 0.85))
        max_scale = float(config.get('scale_max', 1.15))
        scale_factor = random.uniform(min_scale, max_scale)
        if abs(scale_factor - 1.0) >= 0.02:
            img, bxs = apply_zoom_scale(img, bxs, scale_factor)
            pct = int(round((scale_factor - 1.0) * 100))
            sign = "+" if pct > 0 else ""
            applied.append(f"Scale {sign}{pct}%")

    # 5. Translasi / Shift
    if config.get('translate', False):
        shift_limit = float(config.get('translate_limit', 0.08))
        sx = random.uniform(-shift_limit, shift_limit)
        sy = random.uniform(-shift_limit, shift_limit)
        if abs(sx) >= 0.005 or abs(sy) >= 0.005:
            img, bxs = apply_translation(img, bxs, sx, sy)
            applied.append(f"Shift ({int(round(sx*100)):+d}%, {int(round(sy*100)):+d}%)")

    # 6. Brightness & Contrast
    if config.get('brightness', False) or config.get('contrast', False):
        b_range = float(config.get('brightness_limit', 0.2)) if config.get('brightness', False) else 0.0
        c_range = float(config.get('contrast_limit', 0.2)) if config.get('contrast', False) else 0.0
        b_fact = random.uniform(1.0 - b_range, 1.0 + b_range)
        c_fact = random.uniform(1.0 - c_range, 1.0 + c_range)
        img = apply_brightness_contrast(img, b_fact, c_fact)
        if abs(b_fact - 1.0) >= 0.02 and config.get('brightness', False):
            bpct = int(round((b_fact - 1.0) * 100))
            applied.append(f"Brightness {bpct:+d}%")
        if abs(c_fact - 1.0) >= 0.02 and config.get('contrast', False):
            cpct = int(round((c_fact - 1.0) * 100))
            applied.append(f"Contrast {cpct:+d}%")

    # 7. Saturasi Warna
    if config.get('saturation', False):
        sat_range = float(config.get('saturation_limit', 0.3))
        sat_fact = random.uniform(1.0 - sat_range, 1.0 + sat_range)
        if abs(sat_fact - 1.0) >= 0.03:
            img = apply_saturation(img, sat_fact)
            spct = int(round((sat_fact - 1.0) * 100))
            applied.append(f"Saturasi {spct:+d}%")

    # 8. Grayscale
    if config.get('grayscale', False):
        gray_p = float(config.get('grayscale_p', 0.3))
        if random.random() < gray_p:
            img = apply_grayscale(img)
            applied.append("Grayscale")

    # 9. Blur (Motion / Gaussian)
    if config.get('blur', False):
        blur_mode = config.get('blur_type', 'gaussian')
        if blur_mode == 'motion':
            ksize = int(config.get('blur_ksize', 7))
            angle = random.uniform(0, 180)
            img = apply_motion_blur(img, ksize, angle)
            applied.append(f"Motion Blur (k={ksize})")
        else:
            ksize = int(config.get('blur_ksize', 5))
            img = apply_gaussian_blur(img, ksize)
            applied.append(f"Gaussian Blur (k={ksize})")

    # 10. Sensor Noise
    if config.get('noise', False):
        sigma = float(config.get('noise_sigma', 12.0))
        img = apply_sensor_noise(img, sigma)
        applied.append(f"Sensor Noise (σ={sigma:.0f})")

    # 11. Cutout / Random Erasing
    if config.get('cutout', False):
        n_patches = int(config.get('cutout_patches', 2))
        p_size = float(config.get('cutout_size', 0.08))
        img, bxs = apply_cutout(img, bxs, num_patches=n_patches, patch_size_pct=p_size)
        applied.append(f"Cutout {n_patches}x")

    if return_applied:
        return img, bxs, applied
    return img, bxs


# ==============================================================================
# Helper Live Preview (Base64 data URIs untuk tampilan UI)
# ==============================================================================

def cv2_to_base64_jpeg(image_bgr, quality=85):
    """Mengonversi citra OpenCV BGR ke base64 data URI."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


def generate_live_preview_samples(yaml_path, sample_filename, config, num_variations=4):
    """
    Mengambil 1 gambar sampel dari dataset dan menghasilkan N variasi augmentasi acak
    berdasarkan konfigurasi untuk ditampilkan di Live Preview UI.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {yaml_path}")

    data = parse_yaml_safe(yaml_path)
    yaml_dir = os.path.dirname(yaml_path)
    base_path = data.get('path', '')
    base_dir = os.path.normpath(os.path.join(yaml_dir, base_path)) if base_path else yaml_dir

    train_val = data.get('train') or data.get('val')
    train_dir = os.path.normpath(os.path.join(base_dir, train_val))
    if not os.path.exists(train_dir):
        train_dir = os.path.normpath(os.path.join(yaml_dir, train_val))

    labels_dir = train_dir.replace('images', 'labels') if 'images' in train_dir else None
    class_names = normalize_class_names(data.get('names', {}))

    # Temukan path fisik gambar sampel
    img_path = None
    if sample_filename:
        for root, _, files in os.walk(train_dir):
            if sample_filename in files:
                img_path = os.path.join(root, sample_filename)
                break

    # Fallback jika nama file tidak diberikan atau tidak ditemukan: ambil gambar pertama yang punya label
    if not img_path:
        for root, _, files in os.walk(train_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    img_path = os.path.join(root, f)
                    sample_filename = f
                    break
            if img_path:
                break

    if not img_path or not os.path.exists(img_path):
        raise FileNotFoundError("Tidak ditemukan gambar sampel di folder dataset.")

    # Baca gambar asli
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise RuntimeError(f"Gagal membaca gambar {img_path}")

    # Baca label asli jika ada
    boxes = []
    base_no_ext = os.path.splitext(sample_filename)[0]
    candidate_lbls = []
    if labels_dir:
        candidate_lbls.append(os.path.join(labels_dir, f"{base_no_ext}.txt"))
    candidate_lbls.append(os.path.join(os.path.dirname(img_path).replace('images', 'labels'), f"{base_no_ext}.txt"))
    candidate_lbls.append(os.path.join(os.path.dirname(img_path), f"{base_no_ext}.txt"))

    lbl_file = None
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

    # Format gambar asli
    orig_b64 = cv2_to_base64_jpeg(orig_img)
    original_data = {
        "filename": sample_filename,
        "image_url": orig_b64,
        "width": orig_img.shape[1],
        "height": orig_img.shape[0],
        "boxes": boxes
    }

    # Buat N variasi augmentasi
    variations = []
    for i in range(num_variations):
        aug_img, aug_boxes, applied = augment_single_image(orig_img, boxes, config, return_applied=True)
        aug_b64 = cv2_to_base64_jpeg(aug_img)
        variations.append({
            "variation_id": i + 1,
            "title": f"Variasi #{i + 1}",
            "applied_transforms": applied,
            "summary": ", ".join(applied) if applied else "Asli (Tidak ada parameter terpilih)",
            "image_url": aug_b64,
            "boxes": aug_boxes
        })

    return {
        "success": True,
        "original": original_data,
        "variations": variations
    }


# ==============================================================================
# Batch Dataset Augmentation Generator (Background Task)
# ==============================================================================

def run_batch_augmentation_worker(app, job_id, source_dataset_id, new_dataset_name, config, target_subset='train', multiplier=2, include_original=True, replace_source=False):
    """
    Worker background thread untuk mengeksekusi augmentasi seluruh dataset
    dan membuat entri Dataset baru lengkap dengan ZIP & data.yaml.
    Jika replace_source=True, dataset mentah asal akan dihapus sehingga tidak ada dataset ganda di studio.
    """
    with app.app_context():
        try:
            update_job_status(job_id, status='running', progress=2, message='Menyiapkan dataset sumber...')

            src_dataset = Dataset.query.get(source_dataset_id)
            if not src_dataset or not src_dataset.yaml_path or not os.path.exists(src_dataset.yaml_path):
                raise FileNotFoundError("Dataset sumber tidak valid atau data.yaml tidak ditemukan.")

            src_yaml = parse_yaml_safe(src_dataset.yaml_path)
            src_yaml_dir = os.path.dirname(src_dataset.yaml_path)
            base_p = src_yaml.get('path', '')
            base_dir = os.path.normpath(os.path.join(src_yaml_dir, base_p)) if base_p else src_yaml_dir

            class_names = normalize_class_names(src_yaml.get('names', {}))
            nc = src_yaml.get('nc', len(class_names))

            # Identifikasi folder subset
            subset_paths = {}
            for sname in ['train', 'val', 'test']:
                rel = src_yaml.get(sname)
                if rel:
                    full_p = os.path.normpath(os.path.join(base_dir, rel))
                    if not os.path.exists(full_p):
                        full_p = os.path.normpath(os.path.join(src_yaml_dir, rel))
                    if os.path.exists(full_p):
                        subset_paths[sname] = full_p

            if not subset_paths:
                raise RuntimeError("Tidak ditemukan folder subset citra (train/val/test) pada data.yaml.")

            # Buat direktori tujuan dataset baru di storage/datasets/
            storage_root = app.config.get('DATASETS_DIR')
            if not storage_root:
                storage_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'storage', 'datasets')
            new_ds_id = f"dataset_{uuid.uuid4().hex[:8]}"
            new_ds_dir = os.path.join(storage_root, new_ds_id)
            new_extracted_dir = os.path.join(new_ds_dir, 'extracted', 'dataset')

            for sname in ['train', 'val', 'test']:
                os.makedirs(os.path.join(new_extracted_dir, 'images', sname), exist_ok=True)
                os.makedirs(os.path.join(new_extracted_dir, 'labels', sname), exist_ok=True)

            # Hitung total file yang akan diproses
            total_images_to_process = 0
            file_queue = [] # List of tuples: (subset_name, img_path, lbl_path, filename)

            for sname, sdir in subset_paths.items():
                lbl_dir = sdir.replace('images', 'labels') if 'images' in sdir else None
                for root, _, files in os.walk(sdir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                            img_p = os.path.join(root, f)
                            base_no_ext = os.path.splitext(f)[0]
                            lbl_p = None
                            if lbl_dir:
                                test_lp = os.path.join(lbl_dir, f"{base_no_ext}.txt")
                                if os.path.exists(test_lp):
                                    lbl_p = test_lp
                            if not lbl_p:
                                test_lp2 = os.path.join(root.replace('images', 'labels'), f"{base_no_ext}.txt")
                                if os.path.exists(test_lp2):
                                    lbl_p = test_lp2
                            if not lbl_p:
                                test_lp3 = os.path.join(root, f"{base_no_ext}.txt")
                                if os.path.exists(test_lp3):
                                    lbl_p = test_lp3

                            file_queue.append((sname, img_p, lbl_p, f))
                            total_images_to_process += 1

            if total_images_to_process == 0:
                raise RuntimeError("Tidak ada gambar yang ditemukan untuk diaugmentasi.")

            update_job_status(job_id, total=total_images_to_process, processed=0, progress=5, message='Memulai proses augmentasi...')

            counts = {'train': 0, 'val': 0, 'test': 0}
            processed_counter = 0

            # Proses setiap gambar
            for sname, img_path, lbl_path, fname in file_queue:
                dst_img_dir = os.path.join(new_extracted_dir, 'images', sname)
                dst_lbl_dir = os.path.join(new_extracted_dir, 'labels', sname)

                # Baca kotak bounding box awal
                orig_boxes = []
                if lbl_path and os.path.exists(lbl_path):
                    try:
                        with open(lbl_path, 'r', encoding='utf-8') as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cid = int(parts[0])
                                    cx, cy, w, h = [float(v) for v in parts[1:5]]
                                    orig_boxes.append({
                                        "class_id": cid,
                                        "cx": cx,
                                        "cy": cy,
                                        "w": w,
                                        "h": h
                                    })
                    except Exception:
                        pass

                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    processed_counter += 1
                    continue

                # 1. Salin gambar & label asli jika diminta
                if include_original:
                    dst_orig_img = os.path.join(dst_img_dir, fname)
                    shutil.copy2(img_path, dst_orig_img)
                    dst_orig_lbl = os.path.join(dst_lbl_dir, f"{os.path.splitext(fname)[0]}.txt")
                    if lbl_path and os.path.exists(lbl_path):
                        shutil.copy2(lbl_path, dst_orig_lbl)
                    else:
                        open(dst_orig_lbl, 'w').close()
                    counts[sname] += 1

                # 2. Terapkan augmentasi jika subset memenuhi syarat
                # target_subset: 'train' (hanya train) | 'train_val' (train dan val) | 'all' (semua)
                should_augment = False
                if target_subset == 'train' and sname == 'train':
                    should_augment = True
                elif target_subset == 'train_val' and sname in ['train', 'val']:
                    should_augment = True
                elif target_subset == 'all':
                    should_augment = True

                if should_augment:
                    # Jalankan augmentasi sebanyak multiplier
                    for m_idx in range(multiplier):
                        aug_img, aug_boxes = augment_single_image(orig_img, orig_boxes, config)
                        aug_fname = f"aug_m{m_idx + 1}_{fname}"
                        dst_aug_img = os.path.join(dst_img_dir, aug_fname)
                        cv2.imwrite(dst_aug_img, aug_img)

                        aug_lbl_name = f"aug_m{m_idx + 1}_{os.path.splitext(fname)[0]}.txt"
                        dst_aug_lbl = os.path.join(dst_lbl_dir, aug_lbl_name)
                        with open(dst_aug_lbl, 'w', encoding='utf-8') as alf:
                            for b in aug_boxes:
                                alf.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

                        counts[sname] += 1

                processed_counter += 1
                progress_pct = 5 + int((processed_counter / total_images_to_process) * 80)
                update_job_status(
                    job_id,
                    processed=processed_counter,
                    progress=progress_pct,
                    message=f"Memproses {fname} ({processed_counter}/{total_images_to_process})..."
                )

            # Buat data.yaml baru
            update_job_status(job_id, progress=88, message='Membuat data.yaml dan mengompresi arsip ZIP...')

            yaml_names = src_yaml.get('names', {})
            if isinstance(yaml_names, dict):
                sorted_names = [yaml_names[k] for k in sorted(yaml_names.keys())] if yaml_names else ['object']
            elif isinstance(yaml_names, list):
                sorted_names = yaml_names
            else:
                sorted_names = ['object']

            new_yaml_content = {
                'path': os.path.abspath(new_extracted_dir).replace('\\', '/'),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(sorted_names),
                'names': sorted_names
            }

            new_yaml_path = os.path.join(new_extracted_dir, 'data.yaml')
            with open(new_yaml_path, 'w', encoding='utf-8') as nyf:
                yaml.dump(new_yaml_content, nyf, default_flow_style=False, sort_keys=False)

            # Buat file ZIP fisik lengkap standar YOLO
            clean_name = secure_filename(new_dataset_name).strip() or f"dataset_{new_ds_id[-6:]}"
            zip_filename = f"{clean_name}.zip"
            zip_path = os.path.join(new_ds_dir, zip_filename)

            import zipfile
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(new_extracted_dir):
                    for file in files:
                        if file.lower().endswith('.zip'):
                            continue
                        full_file = os.path.join(root, file)
                        rel_file = os.path.relpath(full_file, new_extracted_dir)
                        zipf.write(full_file, rel_file)

            # Registrasi ke Database Studio
            update_job_status(job_id, progress=95, message='Mendaftarkan dataset ke database studio...')

            class_names_dict = {i: name for i, name in enumerate(sorted_names)}
            target_pid = src_dataset.project_id or (Project.query.first().id if Project.query.first() else None)

            new_dataset_row = Dataset(
                id=new_ds_id,
                name=new_dataset_name,
                upload_filename=zip_filename,
                storage_path=new_ds_dir,
                yaml_path=new_yaml_path,
                num_classes=len(sorted_names),
                class_names_json=json.dumps(class_names_dict),
                train_count=counts['train'],
                val_count=counts['val'],
                test_count=counts['test'],
                project_id=target_pid
            )
            db.session.add(new_dataset_row)
            db.session.commit()

            # Bersihkan dataset sumber mentah jika replace_source=True agar tidak ada duplikasi
            if replace_source and source_dataset_id != new_ds_id:
                try:
                    old_ds = Dataset.query.get(source_dataset_id)
                    if old_ds:
                        old_storage = old_ds.storage_path
                        db.session.delete(old_ds)
                        db.session.commit()
                        if old_storage and os.path.exists(old_storage):
                            shutil.rmtree(old_storage, ignore_errors=True)
                except Exception as del_err:
                    print(f"[ReplaceSource Cleanup Error] {del_err}")

            update_job_status(
                job_id,
                status='completed',
                progress=100,
                message=f"Selesai! Berhasil menghasilkan {counts['train']} train, {counts['val']} val, {counts['test']} test.",
                new_dataset_id=new_ds_id,
                train_count=counts['train'],
                val_count=counts['val'],
                test_count=counts['test']
            )

        except Exception as e:
            print(f"[BatchAugmentation Error] {e}")
            update_job_status(
                job_id,
                status='failed',
                error=str(e),
                message=f"Gagal melakukan augmentasi: {str(e)}"
            )


def start_augmentation_job(app, source_dataset_id, new_dataset_name, config, target_subset='train', multiplier=2, include_original=True, replace_source=False):
    """
    Membuat job ID baru dan meluncurkan thread background untuk augmentasi.
    """
    job_id = f"aug_{uuid.uuid4().hex[:8]}"
    with _JOBS_LOCK:
        _AUGMENTATION_JOBS[job_id] = {
            "status": "pending",
            "progress": 0,
            "processed": 0,
            "total": 0,
            "message": "Menunggu giliran eksekusi...",
            "new_dataset_id": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat()
        }

    t = threading.Thread(
        target=run_batch_augmentation_worker,
        args=(app, job_id, source_dataset_id, new_dataset_name, config, target_subset, multiplier, include_original, replace_source),
        daemon=True
    )
    t.start()

    return job_id
