import os
import json
import uuid
import time
import shutil
import random
import cv2
import numpy as np
from PIL import Image
import yaml
from werkzeug.utils import secure_filename
from config import Config, STORAGE_ROOT
from ..models import db
from ..models.labeling import LabelingSession
from ..models.dataset import Dataset

LABELING_DIR = os.path.join(STORAGE_ROOT, 'labeling')
os.makedirs(LABELING_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_session_dir(session_id):
    return os.path.join(LABELING_DIR, session_id)

def create_labeling_session(name, project_id, files):
    """
    Membuat sesi pelabelan baru dan menyimpan gambar-gambar yang diupload.
    Mendukung:
    - Upload gambar individual / multi-gambar
    - Upload 1 folder gambar mentah
    - Upload 1 folder dataset pra-label (berisi subfolder images/, labels/, dan classes.txt / data.yaml)
    """
    session_id = f"lbl_{uuid.uuid4().hex[:8]}"
    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')
    base_images_dir = os.path.join(session_dir, 'base_images')
    labels_dir = os.path.join(session_dir, 'labels')
    crops_dir = os.path.join(session_dir, 'crops')
    overlays_dir = os.path.join(session_dir, 'overlays')

    for d in [images_dir, base_images_dir, labels_dir, crops_dir, overlays_dir]:
        os.makedirs(d, exist_ok=True)

    saved_images_count = 0
    raw_classes_found = []
    pending_label_files = []  # list of (basename, file_storage)

    for file in files:
        if not file or not file.filename:
            continue
        # Normalize relative path dari upload webkitdirectory
        norm_rel = file.filename.replace('\\', '/')
        base_name = os.path.basename(norm_rel)
        name_lower = base_name.lower()
        ext = os.path.splitext(base_name)[1].lower()

        # 1. Cek berkas definisi kelas (classes.txt, class.txt, names.txt)
        if name_lower in ['classes.txt', 'class.txt', 'names.txt']:
            try:
                content = file.read().decode('utf-8', errors='ignore')
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                for idx, cname in enumerate(lines):
                    color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
                    raw_classes_found.append({"id": idx, "name": cname, "color": color})
            except Exception as e:
                print(f"[CreateSession] Error parsing classes file: {e}")
            continue

        # 2. Cek berkas data.yaml / data.yml
        if name_lower in ['data.yaml', 'data.yml']:
            try:
                content = file.read().decode('utf-8', errors='ignore')
                parsed_yaml = yaml.safe_load(content)
                if parsed_yaml and 'names' in parsed_yaml:
                    names_val = parsed_yaml['names']
                    if isinstance(names_val, list):
                        for idx, cname in enumerate(names_val):
                            color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
                            raw_classes_found.append({"id": idx, "name": str(cname), "color": color})
                    elif isinstance(names_val, dict):
                        for k, cname in names_val.items():
                            cid = int(k)
                            color = DEFAULT_PALETTE[cid % len(DEFAULT_PALETTE)]
                            raw_classes_found.append({"id": cid, "name": str(cname), "color": color})
            except Exception as e:
                print(f"[CreateSession] Error parsing data.yaml: {e}")
            continue

        # 3. Cek classes.json
        if name_lower == 'classes.json':
            try:
                content = file.read().decode('utf-8', errors='ignore')
                parsed_json = json.loads(content)
                if isinstance(parsed_json, list):
                    raw_classes_found.extend(parsed_json)
            except Exception:
                pass
            continue

        # 4. Berkas Gambar
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            filename = secure_filename(base_name)
            if not filename:
                filename = f"img_{uuid.uuid4().hex[:6]}{ext}"

            dest_path = os.path.join(images_dir, filename)
            if os.path.exists(dest_path):
                name_part, ext_part = os.path.splitext(filename)
                filename = f"{name_part}_{uuid.uuid4().hex[:4]}{ext_part}"
                dest_path = os.path.join(images_dir, filename)

            file.save(dest_path)
            # Salin juga ke base_images sebagai master murni yang tidak terdestruksi
            shutil.copy2(dest_path, os.path.join(base_images_dir, filename))
            saved_images_count += 1
            continue

        # 5. Berkas Label YOLO .txt
        if ext == '.txt':
            sec_lbl = secure_filename(base_name) or base_name
            pending_label_files.append((sec_lbl, file))

    # Simpan file label yang diunggah
    for lbl_name, lbl_file in pending_label_files:
        lbl_base = os.path.splitext(lbl_name)[0]
        dest_lbl = os.path.join(labels_dir, f"{lbl_base}.txt")
        lbl_file.save(dest_lbl)

    # Inisialisasi kelas: gunakan kelas yang ditemukan dari upload (jika ada), atau list kosong []
    initial_classes = raw_classes_found
    classes_file = os.path.join(session_dir, 'classes.json')
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(initial_classes, f, indent=2)

    # Hitung gambar yang sudah berlabel dari berkas label yang tersimpan
    images_list = get_session_images(session_id)
    labeled_count = sum(1 for img in images_list if img['is_labeled'])

    session_record = LabelingSession(
        id=session_id,
        name=name or f"Labeling Session {session_id}",
        project_id=project_id,
        classes_json=json.dumps(initial_classes),
        storage_path=session_dir,
        crop_path=crops_dir,
        image_count=len(images_list),
        labeled_count=labeled_count
    )
    db.session.add(session_record)
    db.session.commit()

    return session_record

def get_session_images(session_id):
    """
    Mengambil daftar file gambar dalam sesi beserta status apakah sudah dilabeli.
    """
    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')
    labels_dir = os.path.join(session_dir, 'labels')

    if not os.path.exists(images_dir):
        return []

    images = []
    files = sorted(os.listdir(images_dir))
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            base_no_ext = os.path.splitext(f)[0]
            label_file = os.path.join(labels_dir, f"{base_no_ext}.txt")
            is_labeled = False
            bbox_count = 0
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r', encoding='utf-8') as lf:
                        lines = [l.strip() for l in lf.readlines() if l.strip()]
                        bbox_count = len(lines)
                        is_labeled = (bbox_count > 0)
                except Exception:
                    pass
            images.append({
                "filename": f,
                "is_labeled": is_labeled,
                "bbox_count": bbox_count
            })

    return images

def load_annotations(session_id, filename):
    """
    Membaca bounding box format YOLO (.txt) untuk gambar tertentu.
    """
    session_dir = get_session_dir(session_id)
    labels_dir = os.path.join(session_dir, 'labels')
    base_no_ext = os.path.splitext(filename)[0]
    label_file = os.path.join(labels_dir, f"{base_no_ext}.txt")

    bboxes = []
    if not os.path.exists(label_file):
        return bboxes

    overlays = get_session_overlays(session_id, filename)

    used_ov_ids = set()
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Cek apakah bbox ini cocok dengan salah satu crop overlay yang tersimpan
                    matched_ov = None
                    min_dist = 0.05
                    for ov in overlays:
                        if ov.get('id') in used_ov_ids:
                            continue
                        dist = ((ov.get('xc', 0.0) - xc)**2 + (ov.get('yc', 0.0) - yc)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            matched_ov = ov

                    if matched_ov:
                        used_ov_ids.add(matched_ov.get('id'))
                        bboxes.append({
                            "id": matched_ov.get('id', uuid.uuid4().hex[:6]),
                            "class_id": class_id,
                            "xc": xc,
                            "yc": yc,
                            "w": w,
                            "h": h,
                            "is_crop_layer": True,
                            "scale": matched_ov.get('scale', 1.0),
                            "crop_filename": matched_ov.get('crop_filename')
                        })
                    else:
                        bboxes.append({
                            "id": uuid.uuid4().hex[:6],
                            "class_id": class_id,
                            "xc": xc,
                            "yc": yc,
                            "w": w,
                            "h": h
                        })
    except Exception as e:
        print(f"[LabelingService] Error load annotation {label_file}: {e}")

    return bboxes

def save_annotations(session_id, filename, bboxes):
    """
    Menyimpan bounding box format YOLO (.txt) untuk gambar tertentu.
    bboxes: list of dict {'class_id': int, 'xc': float, 'yc': float, 'w': float, 'h': float}
    """
    session_dir = get_session_dir(session_id)
    labels_dir = os.path.join(session_dir, 'labels')
    base_no_ext = os.path.splitext(filename)[0]
    label_file = os.path.join(labels_dir, f"{base_no_ext}.txt")

    with open(label_file, 'w', encoding='utf-8') as f:
        for b in bboxes:
            cid = int(b.get('class_id', 0))
            xc = max(0.0, min(1.0, float(b.get('xc', 0.5))))
            yc = max(0.0, min(1.0, float(b.get('yc', 0.5))))
            w = max(0.001, min(1.0, float(b.get('w', 0.1))))
            h = max(0.001, min(1.0, float(b.get('h', 0.1))))
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Update counter labeled_count pada record session
    session_record = LabelingSession.query.get(session_id)
    if session_record:
        images = get_session_images(session_id)
        session_record.labeled_count = sum(1 for img in images if img['is_labeled'])
        session_record.image_count = len(images)
        db.session.commit()

    return True

DEFAULT_PALETTE = [
    "#38bdf8", "#f43f5e", "#10b981", "#f59e0b", "#a855f7",
    "#06b6d4", "#ec4899", "#84cc16", "#eab308", "#6366f1"
]

def get_session_classes(session_id):
    """
    Mengambil daftar kelas aktif dari classes.json atau database.
    """
    session_dir = get_session_dir(session_id)
    classes_file = os.path.join(session_dir, 'classes.json')
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    session_record = LabelingSession.query.get(session_id)
    if session_record:
        return session_record.get_classes()
    return []

def save_session_classes(session_id, classes_list):
    """
    Menyimpan daftar kelas ke classes.json dan sinkronisasi ke tabel database.
    """
    session_dir = get_session_dir(session_id)
    classes_file = os.path.join(session_dir, 'classes.json')
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(classes_list, f, indent=2)

    session_record = LabelingSession.query.get(session_id)
    if session_record:
        session_record.set_classes(classes_list)
        db.session.commit()
    return classes_list

def add_class_to_session(session_id, name, color=None):
    """
    Menambah kelas deteksi baru secara manual ke sesi.
    """
    classes = get_session_classes(session_id)
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Nama kelas tidak boleh kosong.")

    # Cek apakah nama kelas sudah ada
    for c in classes:
        if c.get('name', '').strip().lower() == clean_name.lower():
            return c, classes

    next_id = 0
    if classes:
        next_id = max(int(c.get('id', 0)) for c in classes) + 1

    if not color:
        color = DEFAULT_PALETTE[next_id % len(DEFAULT_PALETTE)]

    new_class = {
        "id": next_id,
        "name": clean_name,
        "color": color
    }
    classes.append(new_class)
    save_session_classes(session_id, classes)
    return new_class, classes

def delete_class_from_session(session_id, class_id):
    """
    Menghapus kelas dari sesi pelabelan dan membersihkan BBox terkait dari berkas label.
    """
    classes = get_session_classes(session_id)
    target_cid = int(class_id)
    filtered = [c for c in classes if int(c.get('id', -1)) != target_cid]
    save_session_classes(session_id, filtered)

    # Bersihkan anotasi bboxes yang memakai class_id ini
    session_dir = get_session_dir(session_id)
    labels_dir = os.path.join(session_dir, 'labels')
    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            if f.endswith('.txt'):
                lpath = os.path.join(labels_dir, f)
                try:
                    with open(lpath, 'r', encoding='utf-8') as lf:
                        lines = lf.readlines()
                    kept = []
                    for l in lines:
                        parts = l.strip().split()
                        if len(parts) >= 5 and int(parts[0]) != target_cid:
                            kept.append(l)
                    with open(lpath, 'w', encoding='utf-8') as lf:
                        lf.writelines(kept)
                except Exception:
                    pass

    # Perbarui jumlah gambar yang berlabel pada session
    session_record = LabelingSession.query.get(session_id)
    if session_record:
        images = get_session_images(session_id)
        session_record.labeled_count = sum(1 for img in images if img['is_labeled'])
        db.session.commit()

    return filtered

def crop_object_with_feathering(session_id, filename, bbox, blur_radius=15):
    """
    Memotong objek dari BBox dan memberikan efek blur / feathering pada tepian alpha mask.
    Tepi potongan meluruh halus ke transparan (RGBA) sehingga menyatu alami saat di-paste.
    """
    session_dir = get_session_dir(session_id)
    img_path = os.path.join(session_dir, 'images', filename)
    crops_dir = os.path.join(session_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {filename} tidak ditemukan.")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar {filename}.")

    img_h, img_w = img.shape[:2]

    # Ekstrak koordinat piksel
    xc = float(bbox.get('xc', 0.5)) * img_w
    yc = float(bbox.get('yc', 0.5)) * img_h
    bw = float(bbox.get('w', 0.2)) * img_w
    bh = float(bbox.get('h', 0.2)) * img_h

    x1 = int(max(0, xc - bw / 2))
    y1 = int(max(0, yc - bh / 2))
    x2 = int(min(img_w, xc + bw / 2))
    y2 = int(min(img_h, yc + bh / 2))

    crop_bgr = img[y1:y2, x1:x2]
    crop_h, crop_w = crop_bgr.shape[:2]
    if crop_h < 4 or crop_w < 4:
        raise ValueError("Ukuran crop terlalu kecil.")

    # Buat Alpha Mask dengan tepi blur / feathering (soft edge)
    # Langkah 1: Buat mask putih di dalam, hitam di luar
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    
    # Inset margin untuk batas meluruh
    feather_margin = max(3, min(crop_w // 8, crop_h // 8, blur_radius))
    cv2.rectangle(
        mask, 
        (feather_margin, feather_margin), 
        (crop_w - feather_margin, crop_h - feather_margin), 
        255, 
        -1
    )

    # Langkah 2: Gaussian Blur pada mask untuk menghasilkan gradien transparansi lembut
    ksize = feather_margin * 2 + 1
    if ksize % 2 == 0:
        ksize += 1
    feathered_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Langkah 3: Gabungkan BGR dengan Alpha Mask menjadi RGBA
    b, g, r = cv2.split(crop_bgr)
    rgba = cv2.merge([b, g, r, feathered_mask])

    # Simpan hasil crop RGBA PNG
    crop_id = f"crop_{uuid.uuid4().hex[:8]}"
    crop_filename = f"{crop_id}.png"
    crop_path = os.path.join(crops_dir, crop_filename)
    cv2.imwrite(crop_path, rgba)

    class_id = int(bbox.get('class_id', 0))

    return {
        "crop_id": crop_id,
        "filename": crop_filename,
        "crop_filename": crop_filename,
        "width": crop_w,
        "height": crop_h,
        "class_id": class_id,
        "url": f"/labeling/{session_id}/crops/{crop_filename}"
    }

def get_session_crops(session_id):
    """
    Mengambil seluruh daftar crop aset objek bertransparansi halus yang tersimpan.
    """
    session_dir = get_session_dir(session_id)
    crops_dir = os.path.join(session_dir, 'crops')
    if not os.path.exists(crops_dir):
        return []

    crops = []
    for f in sorted(os.listdir(crops_dir)):
        if f.endswith('.png'):
            crop_path = os.path.join(crops_dir, f)
            try:
                im = Image.open(crop_path)
                crops.append({
                    "crop_id": os.path.splitext(f)[0],
                    "filename": f,
                    "width": im.width,
                    "height": im.height,
                    "url": f"/labeling/{session_id}/crops/{f}"
                })
            except Exception:
                pass
    return crops

def synthesize_blend_data(session_id, bg_filename, crop_filename, target_x_pct=0.5, target_y_pct=0.5, scale=1.0, class_id=0):
    """
    Menempelkan objek crop bertepi halus ke background image tertentu dan
    secara otomatis men-generate gambar sintetis + bounding box YOLO baru.
    """
    session_dir = get_session_dir(session_id)
    bg_path = os.path.join(session_dir, 'images', bg_filename)
    crop_path = os.path.join(session_dir, 'crops', crop_filename)

    if not os.path.exists(bg_path) or not os.path.exists(crop_path):
        raise FileNotFoundError("Background image atau Crop asset tidak ditemukan.")

    bg_img = cv2.imread(bg_path)
    crop_rgba = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
    if bg_img is None or crop_rgba is None:
        raise ValueError("Gagal membaca gambar background atau crop.")

    bg_h, bg_w = bg_img.shape[:2]
    crop_h, crop_w = crop_rgba.shape[:2]

    # Scaling crop jika diminta
    if scale != 1.0 and scale > 0.1:
        new_w = max(8, int(crop_w * scale))
        new_h = max(8, int(crop_h * scale))
        crop_rgba = cv2.resize(crop_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        crop_h, crop_w = crop_rgba.shape[:2]

    # Tentukan posisi tengah penempelan
    cx = int(target_x_pct * bg_w)
    cy = int(target_y_pct * bg_h)

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clip area jika keluar dari batas background
    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(bg_w, x2)
    bg_y2 = min(bg_h, y2)

    crop_x1 = max(0, -x1)
    crop_y1 = max(0, -y1)
    crop_x2 = crop_w - max(0, x2 - bg_w)
    crop_y2 = crop_h - max(0, y2 - bg_h)

    if bg_x2 <= bg_x1 or bg_y2 <= bg_y1 or crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError("Posisi penempelan berada di luar batas background.")

    # Alpha blending halus
    sub_bg = bg_img[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)
    sub_crop = crop_rgba[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_bgr = sub_crop[:, :, :3].astype(np.float32)
    alpha = (sub_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Formula blending: result = bg * (1 - alpha) + crop * alpha
    blended = sub_bg * (1.0 - alpha) + crop_bgr * alpha
    bg_img[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)

    # Simpan sebagai gambar sintetis baru di folder images
    synth_name = f"synth_{uuid.uuid4().hex[:8]}.jpg"
    synth_img_path = os.path.join(session_dir, 'images', synth_name)
    cv2.imwrite(synth_img_path, bg_img)

    # Hitung koordinat bounding box YOLO baru
    box_w = float(bg_x2 - bg_x1) / bg_w
    box_h = float(bg_y2 - bg_y1) / bg_h
    box_xc = float(bg_x1 + bg_x2) / (2.0 * bg_w)
    box_yc = float(bg_y1 + bg_y2) / (2.0 * bg_h)

    # Simpan label otomatis
    labels_dir = os.path.join(session_dir, 'labels')
    synth_label_path = os.path.join(labels_dir, f"{os.path.splitext(synth_name)[0]}.txt")
    with open(synth_label_path, 'w', encoding='utf-8') as f:
        f.write(f"{class_id} {box_xc:.6f} {box_yc:.6f} {box_w:.6f} {box_h:.6f}\n")

    # Update session count
    session_record = LabelingSession.query.get(session_id)
    if session_record:
        images = get_session_images(session_id)
        session_record.image_count = len(images)
        session_record.labeled_count = sum(1 for img in images if img['is_labeled'])
        db.session.commit()

    return {
        "success": True,
        "filename": synth_name,
        "bbox": {
            "class_id": class_id,
            "xc": box_xc,
            "yc": box_yc,
            "w": box_w,
            "h": box_h
        }
    }

def get_session_overlays(session_id, filename):
    session_dir = get_session_dir(session_id)
    base_no_ext = os.path.splitext(filename)[0]
    overlay_file = os.path.join(session_dir, 'overlays', f"{base_no_ext}.json")
    if os.path.exists(overlay_file):
        try:
            with open(overlay_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_session_overlays(session_id, filename, overlays):
    session_dir = get_session_dir(session_id)
    base_no_ext = os.path.splitext(filename)[0]
    overlays_dir = os.path.join(session_dir, 'overlays')
    os.makedirs(overlays_dir, exist_ok=True)
    overlay_file = os.path.join(overlays_dir, f"{base_no_ext}.json")
    with open(overlay_file, 'w', encoding='utf-8') as f:
        json.dump(overlays, f, indent=2)

def render_composite_image(session_id, filename):
    """
    Merender ulang gambar komposit di images/<filename> dari master base_images/<filename>
    ditambah seluruh crop overlay aktif dengan Gaussian soft-edge feathering.
    Jika tidak ada overlay, gambar komposit kembali bersih persis sama dengan base image asli.
    """
    session_dir = get_session_dir(session_id)
    base_path = os.path.join(session_dir, 'base_images', filename)
    img_path = os.path.join(session_dir, 'images', filename)

    if not os.path.exists(base_path):
        if os.path.exists(img_path):
            os.makedirs(os.path.join(session_dir, 'base_images'), exist_ok=True)
            shutil.copy2(img_path, base_path)
        else:
            return False

    base_img = cv2.imread(base_path)
    if base_img is None:
        return False

    overlays = get_session_overlays(session_id, filename)
    if not overlays:
        cv2.imwrite(img_path, base_img)
        return True

    comp_img = base_img.copy()
    bg_h, bg_w = comp_img.shape[:2]

    for ov in overlays:
        crop_name = ov.get('crop_filename')
        crop_path = os.path.join(session_dir, 'crops', crop_name)
        if not os.path.exists(crop_path):
            continue
        crop_rgba = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        if crop_rgba is None or crop_rgba.shape[2] != 4:
            continue

        scale = float(ov.get('scale', 1.0))
        crop_h, crop_w = crop_rgba.shape[:2]
        if scale != 1.0 and scale > 0.05:
            new_w = max(6, int(crop_w * scale))
            new_h = max(6, int(crop_h * scale))
            crop_rgba = cv2.resize(crop_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_h, crop_w = crop_rgba.shape[:2]

        cx = int(float(ov.get('xc', 0.5)) * bg_w)
        cy = int(float(ov.get('yc', 0.5)) * bg_h)

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        bg_x1 = max(0, x1)
        bg_y1 = max(0, y1)
        bg_x2 = min(bg_w, x2)
        bg_y2 = min(bg_h, y2)

        crop_x1 = max(0, -x1)
        crop_y1 = max(0, -y1)
        crop_x2 = crop_w - max(0, x2 - bg_w)
        crop_y2 = crop_h - max(0, y2 - bg_h)

        if bg_x2 <= bg_x1 or bg_y2 <= bg_y1 or crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        sub_bg = comp_img[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)
        sub_crop = crop_rgba[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_bgr = sub_crop[:, :, :3].astype(np.float32)
        alpha = (sub_crop[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        blended = sub_bg * (1.0 - alpha) + crop_bgr * alpha
        comp_img[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)

    cv2.imwrite(img_path, comp_img)
    return True

def place_crop_on_image(session_id, bg_filename, crop_filename, pos_x_pct=0.5, pos_y_pct=0.5, scale=1.0, class_id=0, create_new=False):
    """
    Menempatkan hasil crop bertepi halus ke gambar latar melalui drag-and-drop.
    Menggunakan layer non-destruktif sehingga objek crop masih dapat dipindahkan atau dihapus.
    """
    session_dir = get_session_dir(session_id)
    bg_path = os.path.join(session_dir, 'images', bg_filename)
    crop_path = os.path.join(session_dir, 'crops', crop_filename)

    if not os.path.exists(bg_path) or not os.path.exists(crop_path):
        raise FileNotFoundError("Background image atau Crop asset tidak ditemukan.")

    bg_img = cv2.imread(bg_path)
    crop_rgba = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
    if bg_img is None or crop_rgba is None:
        raise ValueError("Gagal membaca gambar background atau crop asset.")

    bg_h, bg_w = bg_img.shape[:2]
    crop_h, crop_w = crop_rgba.shape[:2]

    new_w = max(6, int(crop_w * scale))
    new_h = max(6, int(crop_h * scale))

    box_w = float(new_w) / bg_w
    box_h = float(new_h) / bg_h
    box_xc = float(pos_x_pct)
    box_yc = float(pos_y_pct)

    layer_id = f"crop_{uuid.uuid4().hex[:6]}"

    if create_new:
        # Buat gambar sintetis baru terpisah
        synth_name = f"synth_{uuid.uuid4().hex[:8]}.jpg"
        target_path = os.path.join(session_dir, 'images', synth_name)
        shutil.copy2(bg_path, target_path)
        
        # Salin base image
        base_dir = os.path.join(session_dir, 'base_images')
        os.makedirs(base_dir, exist_ok=True)
        shutil.copy2(bg_path, os.path.join(base_dir, synth_name))
        
        overlays = [{
            "id": layer_id,
            "crop_filename": crop_filename,
            "xc": box_xc,
            "yc": box_yc,
            "w": box_w,
            "h": box_h,
            "scale": scale,
            "class_id": int(class_id)
        }]
        save_session_overlays(session_id, synth_name, overlays)
        render_composite_image(session_id, synth_name)

        new_box = {
            "id": layer_id,
            "class_id": int(class_id),
            "xc": box_xc,
            "yc": box_yc,
            "w": box_w,
            "h": box_h,
            "is_crop_layer": True,
            "crop_filename": crop_filename
        }
        save_annotations(session_id, synth_name, [new_box])

        return {
            "success": True,
            "filename": synth_name,
            "bbox": new_box,
            "all_boxes": [new_box],
            "timestamp": int(time.time() * 1000)
        }
    else:
        # Non-destruktif pada gambar saat ini
        base_dir = os.path.join(session_dir, 'base_images')
        os.makedirs(base_dir, exist_ok=True)
        base_path = os.path.join(base_dir, bg_filename)
        if not os.path.exists(base_path):
            shutil.copy2(bg_path, base_path)

        overlays = get_session_overlays(session_id, bg_filename)
        overlays.append({
            "id": layer_id,
            "crop_filename": crop_filename,
            "xc": box_xc,
            "yc": box_yc,
            "w": box_w,
            "h": box_h,
            "scale": scale,
            "class_id": int(class_id)
        })
        save_session_overlays(session_id, bg_filename, overlays)
        render_composite_image(session_id, bg_filename)

        existing_boxes = load_annotations(session_id, bg_filename)
        new_box = {
            "id": layer_id,
            "class_id": int(class_id),
            "xc": box_xc,
            "yc": box_yc,
            "w": box_w,
            "h": box_h,
            "is_crop_layer": True,
            "crop_filename": crop_filename
        }
        existing_boxes.append(new_box)
        save_annotations(session_id, bg_filename, existing_boxes)

        return {
            "success": True,
            "filename": bg_filename,
            "bbox": new_box,
            "all_boxes": existing_boxes,
            "timestamp": int(time.time() * 1000)
        }

def update_crop_layer(session_id, filename, bbox_id, new_xc, new_yc, new_w=None, new_h=None, scale=None, class_id=None):
    """
    Mengubah posisi atau dimensi crop layer di kanvas dan me-render ulang komposit.
    """
    session_dir = get_session_dir(session_id)
    
    # Load bboxes lebih awal agar overlay yang cocok saat ini dapat ditemukan ID-nya
    bboxes = load_annotations(session_id, filename)
    overlays = get_session_overlays(session_id, filename)
    updated = False

    calc_w = new_w
    calc_h = new_h

    for ov in overlays:
        if ov.get('id') == bbox_id:
            ov['xc'] = float(new_xc)
            ov['yc'] = float(new_yc)

            if scale is not None and float(scale) > 0.05:
                ov['scale'] = float(scale)

            if new_w is not None and new_h is not None:
                ov['w'] = float(new_w)
                ov['h'] = float(new_h)
            elif scale is not None:
                # Hitung ulang w dan h berdasarkan scale dan ukuran asli crop
                crop_name = ov.get('crop_filename')
                crop_path = os.path.join(session_dir, 'crops', crop_name) if crop_name else None
                base_path = os.path.join(session_dir, 'base_images', filename)
                if crop_path and os.path.exists(crop_path) and os.path.exists(base_path):
                    crop_img = cv2.imread(crop_path)
                    base_img = cv2.imread(base_path)
                    if crop_img is not None and base_img is not None:
                        ov['w'] = float(crop_img.shape[1] * float(scale)) / base_img.shape[1]
                        ov['h'] = float(crop_img.shape[0] * float(scale)) / base_img.shape[0]

            calc_w = ov.get('w')
            calc_h = ov.get('h')

            if class_id is not None:
                ov['class_id'] = int(class_id)

            updated = True
            break

    # Update BBox koordinat di bboxes
    matched_box = False
    for b in bboxes:
        if b.get('id') == bbox_id:
            b['xc'] = float(new_xc)
            b['yc'] = float(new_yc)
            if calc_w is not None:
                b['w'] = float(calc_w)
            if calc_h is not None:
                b['h'] = float(calc_h)
            if class_id is not None:
                b['class_id'] = int(class_id)
            matched_box = True
            break

    if not matched_box and updated:
        bboxes.append({
            "id": bbox_id,
            "class_id": int(class_id) if class_id is not None else 0,
            "xc": float(new_xc),
            "yc": float(new_yc),
            "w": float(calc_w or 0.1),
            "h": float(calc_h or 0.1),
            "is_crop_layer": True
        })

    if updated:
        save_session_overlays(session_id, filename, overlays)
        render_composite_image(session_id, filename)

    save_annotations(session_id, filename, bboxes)

    return {
        "success": True,
        "filename": filename,
        "bboxes": bboxes,
        "timestamp": int(time.time() * 1000)
    }

def delete_crop_layer(session_id, filename, bbox_id):
    """
    Menghapus crop overlay dan me-render ulang gambar latar belakang ke kondisi murni.
    """
    # 1. Load bboxes terlebih dahulu saat overlay ID masih cocok
    bboxes = load_annotations(session_id, filename)
    filtered_bboxes = [b for b in bboxes if b.get('id') != bbox_id]

    # 2. Filter overlays
    overlays = get_session_overlays(session_id, filename)
    filtered_overlays = [ov for ov in overlays if ov.get('id') != bbox_id]
    save_session_overlays(session_id, filename, filtered_overlays)
    render_composite_image(session_id, filename)

    # 3. Simpan bboxes yang telah terfilter
    save_annotations(session_id, filename, filtered_bboxes)

    return {
        "success": True,
        "filename": filename,
        "bboxes": filtered_bboxes,
        "timestamp": int(time.time() * 1000)
    }

def delete_session_image(session_id, filename):
    """
    Menghapus satu gambar dan seluruh berkas label/overlay terkait dari sesi pelabelan.
    """
    session_dir = get_session_dir(session_id)
    base_no_ext = os.path.splitext(filename)[0]

    paths_to_remove = [
        os.path.join(session_dir, 'images', filename),
        os.path.join(session_dir, 'base_images', filename),
        os.path.join(session_dir, 'labels', f"{base_no_ext}.txt"),
        os.path.join(session_dir, 'overlays', f"{base_no_ext}.json")
    ]
    for p in paths_to_remove:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    session_record = LabelingSession.query.get(session_id)
    if session_record:
        images = get_session_images(session_id)
        session_record.image_count = len(images)
        session_record.labeled_count = sum(1 for img in images if img['is_labeled'])
        db.session.commit()

    return {
        "success": True,
        "filename": filename
    }

def bootstrap_auto_label_from_session(session_id, epochs=10, conf_threshold=0.35, overwrite=False):
    """
    Active Learning Bootstrap:
    Melatih model cepat (YOLOv8 nano) dari sebagian gambar yang sudah dilabeli (misal 10-30 gambar),
    lalu secara otomatis melabeli seluruh sisa gambar lainnya pada sesi ini.
    """
    import torch
    from ultralytics import YOLO

    session_record = LabelingSession.query.get_or_404(session_id)
    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')
    labels_dir = os.path.join(session_dir, 'labels')

    all_images = get_session_images(session_id)
    labeled_images = [img['filename'] for img in all_images if img['is_labeled']]
    
    if overwrite:
        target_images = [img['filename'] for img in all_images]
    else:
        target_images = [img['filename'] for img in all_images if not img['is_labeled']]

    if len(labeled_images) < 2:
        raise ValueError("Minimal beri bounding box pada minimal 2-3 gambar terlebih dahulu sebagai acuan pola objek.")

    if len(target_images) == 0:
        return {
            "success": True,
            "message": "Semua gambar dalam sesi ini sudah memiliki label.",
            "train_images": len(labeled_images),
            "auto_labeled_count": 0,
            "total_detections": 0
        }

    classes = get_session_classes(session_id)
    if not classes:
        raise ValueError("Belum ada kelas deteksi yang terdaftar pada sesi ini.")

    temp_id = f"boot_{uuid.uuid4().hex[:6]}"
    temp_dir = os.path.join(STORAGE_ROOT, 'temp_bootstrap', temp_id)
    t_train_img = os.path.join(temp_dir, 'images', 'train')
    t_val_img = os.path.join(temp_dir, 'images', 'val')
    t_train_lbl = os.path.join(temp_dir, 'labels', 'train')
    t_val_lbl = os.path.join(temp_dir, 'labels', 'val')

    for d in [t_train_img, t_val_img, t_train_lbl, t_val_lbl]:
        os.makedirs(d, exist_ok=True)

    shuffled = list(labeled_images)
    random.seed(42)
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * 0.8))
    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:] if len(shuffled) > 1 else [shuffled[0]]

    for fn in train_files:
        shutil.copy2(os.path.join(images_dir, fn), os.path.join(t_train_img, fn))
        lbl_fn = f"{os.path.splitext(fn)[0]}.txt"
        if os.path.exists(os.path.join(labels_dir, lbl_fn)):
            shutil.copy2(os.path.join(labels_dir, lbl_fn), os.path.join(t_train_lbl, lbl_fn))

    for fn in val_files:
        shutil.copy2(os.path.join(images_dir, fn), os.path.join(t_val_img, fn))
        lbl_fn = f"{os.path.splitext(fn)[0]}.txt"
        if os.path.exists(os.path.join(labels_dir, lbl_fn)):
            shutil.copy2(os.path.join(labels_dir, lbl_fn), os.path.join(t_val_lbl, lbl_fn))

    yaml_path = os.path.join(temp_dir, 'data.yaml')
    names_list = [c['name'] for c in classes]
    data_yaml_dict = {
        'path': os.path.abspath(temp_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(names_list),
        'names': names_list
    }
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(data_yaml_dict, yf, default_flow_style=False, sort_keys=False)

    base_weights = 'yolov8n.pt'
    if not os.path.exists(base_weights):
        alt = os.path.join('yolo_training_studio', 'yolov8n.pt')
        if os.path.exists(alt):
            base_weights = alt

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO(base_weights)
    model.train(
        data=yaml_path,
        epochs=max(1, min(30, int(epochs))),
        imgsz=416,
        batch=min(8, len(train_files)),
        device=device,
        workers=0,
        verbose=False,
        plots=False,
        save=True,
        project=os.path.join(temp_dir, 'runs'),
        name='quick_bootstrap'
    )

    trained_weights = os.path.join(temp_dir, 'runs', 'quick_bootstrap', 'weights', 'best.pt')
    if not os.path.exists(trained_weights):
        trained_weights = os.path.join(temp_dir, 'runs', 'quick_bootstrap', 'weights', 'last.pt')

    if not os.path.exists(trained_weights):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("Gagal melatih model bootstrap cepat.")

    infer_model = YOLO(trained_weights)
    idx_to_class_id = {i: c['id'] for i, c in enumerate(classes)}

    auto_labeled_count = 0
    total_new_boxes = 0

    for fn in target_images:
        img_p = os.path.join(images_dir, fn)
        if not os.path.exists(img_p):
            continue
        try:
            preds = infer_model.predict(source=img_p, conf=float(conf_threshold), verbose=False)[0]
            boxes = []
            if preds.boxes is not None and len(preds.boxes) > 0:
                for b in preds.boxes:
                    c_idx = int(b.cls[0].item())
                    cid = idx_to_class_id.get(c_idx, c_idx)
                    xywhn = b.xywhn[0].tolist()
                    boxes.append({
                        "id": uuid.uuid4().hex[:6],
                        "class_id": cid,
                        "xc": float(xywhn[0]),
                        "yc": float(xywhn[1]),
                        "w": float(xywhn[2]),
                        "h": float(xywhn[3]),
                        "conf": float(b.conf[0].item()) if b.conf is not None else None
                    })
                    total_new_boxes += 1
                save_annotations(session_id, fn, boxes)
                auto_labeled_count += 1
        except Exception as e:
            print(f"[BootstrapAutoLabel] Error infer {fn}: {e}")

    # Bersihkan folder bootstrap sementara
    shutil.rmtree(temp_dir, ignore_errors=True)

    session_record = LabelingSession.query.get(session_id)
    if session_record:
        all_imgs = get_session_images(session_id)
        session_record.labeled_count = sum(1 for img in all_imgs if img['is_labeled'])
        db.session.commit()

    return {
        "success": True,
        "train_images": len(labeled_images),
        "auto_labeled_count": auto_labeled_count,
        "total_detections": total_new_boxes,
        "classes_count": len(classes)
    }

def run_auto_labeling(session_id, model_path, target_mode='current', current_filename=None, conf_threshold=0.35, overwrite=False):
    """
    Menjalankan automatic labeling pada gambar menggunakan model YOLO (.pt).
    Mendukung upload model kustom atau memilih best model studio.
    """
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di {model_path}")

    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')

    # Load YOLO model
    model = YOLO(model_path)
    model_names = getattr(model, 'names', {}) or {}

    # Dapatkan kelas sesi saat ini
    session_classes = get_session_classes(session_id)
    name_to_class_id = {str(c['name']).strip().lower(): int(c['id']) for c in session_classes}

    # Sinkronisasi kelas dari model ke sesi jika belum ada
    classes_updated = False
    for m_idx, m_name in model_names.items():
        clean_m_name = str(m_name).strip()
        if clean_m_name.lower() not in name_to_class_id:
            next_id = 0
            if session_classes:
                next_id = max(int(c.get('id', 0)) for c in session_classes) + 1
            color = DEFAULT_PALETTE[next_id % len(DEFAULT_PALETTE)]
            new_cls = {
                "id": next_id,
                "name": clean_m_name,
                "color": color
            }
            session_classes.append(new_cls)
            name_to_class_id[clean_m_name.lower()] = next_id
            classes_updated = True

    if classes_updated:
        save_session_classes(session_id, session_classes)

    # Tentukan gambar yang akan diproses
    if target_mode == 'current' and current_filename:
        images_to_process = [current_filename]
    else:
        images_to_process = [img['filename'] for img in get_session_images(session_id)]

    processed_count = 0
    total_detections = 0
    current_image_boxes = []

    for fname in images_to_process:
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue

        try:
            results = model.predict(source=img_path, conf=float(conf_threshold), verbose=False)
            res = results[0]

            bboxes = [] if overwrite else load_annotations(session_id, fname)

            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_idx = int(box.cls[0].item())
                    cls_name = model_names.get(cls_idx, f"class_{cls_idx}")
                    target_cid = name_to_class_id.get(str(cls_name).strip().lower(), 0)

                    xywhn = box.xywhn[0].tolist()
                    xc, yc, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]

                    bboxes.append({
                        "id": uuid.uuid4().hex[:6],
                        "class_id": target_cid,
                        "xc": float(xc),
                        "yc": float(yc),
                        "w": float(w),
                        "h": float(h),
                        "conf": float(box.conf[0].item()) if box.conf is not None else None
                    })
                    total_detections += 1

            save_annotations(session_id, fname, bboxes)
            processed_count += 1

            if fname == current_filename:
                current_image_boxes = bboxes
        except Exception as e:
            print(f"[LabelingService] Error auto-labeling {fname}: {e}")

    # Update session labeled count
    session_record = LabelingSession.query.get(session_id)
    if session_record:
        all_imgs = get_session_images(session_id)
        session_record.labeled_count = sum(1 for img in all_imgs if img['is_labeled'])
        db.session.commit()

    return {
        "success": True,
        "processed_count": processed_count,
        "total_detections": total_detections,
        "classes": session_classes,
        "current_boxes": current_image_boxes
    }

def split_and_export_dataset(session_id, train_pct=70, val_pct=20, test_pct=10, custom_dataset_name=None, register_to_studio=True):
    """
    Membagi dataset yang telah dilabeli menjadi Train / Val / Test sesuai rasio persentase.
    Membuat file data.yaml Ultralytics standar dan mendaftarkannya ke Studio Dataset.
    """
    session_record = LabelingSession.query.get_or_404(session_id)
    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')
    labels_dir = os.path.join(session_dir, 'labels')

    # Kumpulkan seluruh gambar dalam sesi (termasuk gambar background tanpa BBox)
    images = get_session_images(session_id)
    if not images:
        raise ValueError("Tidak ada gambar yang ditemukan dalam sesi pelabelan ini.")

    labeled_images = [img['filename'] for img in images if img['is_labeled']]
    bg_images = [img['filename'] for img in images if not img['is_labeled']]

    # Validasi: Minimal ada 1 gambar berlabel atau kelas yang didefinisikan
    classes_list = session_record.get_classes()
    if not labeled_images and not classes_list:
        raise ValueError("Belum ada gambar yang memiliki bounding box atau kelas deteksi yang didefinisikan.")

    random.seed(42)

    def calculate_split_counts(count, t_pct, v_pct, te_pct):
        """Menghitung kuota train, val, test secara proporsional dengan pembulatan tepat."""
        if count <= 0:
            return 0, 0, 0
        if te_pct == 0:
            if v_pct > 0 and count > 1:
                nv = max(1, int(round(count * (v_pct / 100.0))))
            elif v_pct > 0 and count == 1:
                nv = 0  # Jika hanya 1 gambar, masukkan ke train
            else:
                nv = 0
            if nv >= count:
                nv = max(0, count - 1)
            nt = count - nv
            nte = 0
            return nt, nv, nte
        else:
            nt = max(0, int(round(count * (t_pct / 100.0))))
            nv = max(0, int(round(count * (v_pct / 100.0))))
            nte = count - nt - nv
            if nte < 0:
                nte = 0
            # Sesuaikan jika total selisih pembulatan tidak sama dengan count
            diff = count - (nt + nv + nte)
            nt += diff
            return max(0, nt), max(0, nv), max(0, nte)

    # 1. Split gambar yang memiliki BBox (berlabel)
    shuffled_labeled = list(labeled_images)
    random.shuffle(shuffled_labeled)
    nt_l, nv_l, nte_l = calculate_split_counts(len(shuffled_labeled), train_pct, val_pct, test_pct)
    train_labeled = shuffled_labeled[:nt_l]
    val_labeled = shuffled_labeled[nt_l:nt_l + nv_l]
    test_labeled = shuffled_labeled[nt_l + nv_l:]

    # 2. Split gambar background (tanpa BBox) secara proporsional
    shuffled_bg = list(bg_images)
    random.shuffle(shuffled_bg)
    nt_b, nv_b, nte_b = calculate_split_counts(len(shuffled_bg), train_pct, val_pct, test_pct)
    train_bg = shuffled_bg[:nt_b]
    val_bg = shuffled_bg[nt_b:nt_b + nv_b]
    test_bg = shuffled_bg[nt_b + nv_b:]

    # 3. Gabungkan gambar berlabel dan background untuk tiap subset
    train_files = train_labeled + train_bg
    val_files = val_labeled + val_bg
    test_files = test_labeled + test_bg

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # Buat direktori dataset baru di storage/datasets/
    dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
    dataset_dir = os.path.join(STORAGE_ROOT, 'datasets', dataset_id)
    extracted_dir = os.path.join(dataset_dir, 'extracted', 'dataset')

    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(extracted_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(extracted_dir, 'labels', split_name), exist_ok=True)

    # Salin file gambar dan label
    for split_name, flist in splits.items():
        for fname in flist:
            src_img = os.path.join(images_dir, fname)
            dst_img = os.path.join(extracted_dir, 'images', split_name, fname)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            base_no_ext = os.path.splitext(fname)[0]
            src_lbl = os.path.join(labels_dir, f"{base_no_ext}.txt")
            dst_lbl = os.path.join(extracted_dir, 'labels', split_name, f"{base_no_ext}.txt")
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # Jika kosong, buat file kosong
                open(dst_lbl, 'w').close()

    # Buat data.yaml
    classes_list = session_record.get_classes()
    nc = len(classes_list)
    names = [c.get('name', f"class_{c.get('id', i)}") for i, c in enumerate(classes_list)]
    if not names:
        nc = 1
        names = ['object']

    data_yaml_content = {
        'path': os.path.abspath(extracted_dir).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc,
        'names': names
    }

    yaml_path = os.path.join(extracted_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.dump(data_yaml_content, yf, default_flow_style=False, sort_keys=False)

    # Tentukan nama dataset
    ds_name = custom_dataset_name or f"{session_record.name} (Labeled)"

    # Buat file ZIP fisik lengkap standar YOLO Ultralytics
    clean_name = secure_filename(ds_name).strip() or f"dataset_{dataset_id[-6:]}"
    zip_filename = f"{clean_name}.zip"
    zip_path = os.path.join(dataset_dir, zip_filename)

    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.lower().endswith('.zip'):
                    continue
                full_file = os.path.join(root, file)
                rel_file = os.path.relpath(full_file, extracted_dir)
                zipf.write(full_file, rel_file)

    # Tentukan project_id tujuan (fallback ke active_project_id atau proj_default jika None)
    target_pid = session_record.project_id
    if not target_pid:
        try:
            from flask import session as flask_session
            target_pid = flask_session.get('active_project_id')
        except Exception:
            pass
    if not target_pid or target_pid == 'all':
        first_proj = Project.query.first()
        target_pid = first_proj.id if first_proj else None

    # Registrasi ke model Dataset Studio jika register_to_studio=True
    new_dataset = None
    if register_to_studio:
        class_names_dict = {i: name for i, name in enumerate(names)}
        new_dataset = Dataset(
            id=dataset_id,
            name=ds_name,
            upload_filename=zip_filename,
            storage_path=dataset_dir,
            yaml_path=yaml_path,
            num_classes=nc,
            class_names_json=json.dumps(class_names_dict),
            train_count=len(train_files),
            val_count=len(val_files),
            test_count=len(test_files),
            project_id=target_pid
        )
        db.session.add(new_dataset)
        db.session.commit()

    return {
        "success": True,
        "dataset_id": dataset_id,
        "dataset_name": ds_name,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "total_images": len(train_files) + len(val_files) + len(test_files),
        "labeled_count": len(labeled_images),
        "bg_count": len(bg_images),
        "yaml_path": yaml_path,
        "zip_filename": zip_filename,
        "download_url": f"/datasets/{dataset_id}/download"
    }

