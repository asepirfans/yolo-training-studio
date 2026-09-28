import os
import time
import uuid
import zipfile
from collections import Counter
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import json
import shutil
from config import Config, RUNS_DIR
from ..models.training_run import TrainingRun
from ..models.dataset import Dataset

def get_available_inference_models(project_id=None):
    """
    Mengumpulkan semua model yang tersedia untuk inferensi:
    Hanya mengambil model yang di-train di studio, khususnya bobot terbaik (best.pt).
    """
    models = []
    
    try:
        query = TrainingRun.query
        if project_id:
            query = query.filter(TrainingRun.project_id == project_id)
        runs = query.order_by(TrainingRun.created_at.desc()).all()
        
        # Identifikasi run terbaik
        from .result_service import find_best_training_run
        best_run = find_best_training_run(project_id=project_id, runs_list=runs)
        best_run_id = best_run.id if best_run else None

        for r in runs:
            out_dir = r.output_path or os.path.join(RUNS_DIR, r.id, "outputs", "train")
            best_path = os.path.join(out_dir, "weights", "best.pt")
            
            if os.path.exists(best_path):
                is_best = bool(best_run_id and r.id == best_run_id)
                best_tag = "★ [HASIL TERBAIK] " if is_best else ""
                score_label = f"mAP50-95: {r.best_map5095}" if r.best_map5095 is not None else f"mAP50: {r.best_map50 or 'N/A'}"

                models.append({
                    "id": f"{r.id}_best",
                    "name": f"{best_tag}Run {r.id} — best.pt ({score_label})",
                    "path": os.path.abspath(best_path),
                    "model_base": r.model_name,
                    "dataset_name": r.dataset.name if r.dataset else "Unknown",
                    "type": "trained_best",
                    "run_id": r.id,
                    "is_best": is_best,
                    "best_map50": r.best_map50,
                    "best_map5095": r.best_map5095,
                    "project_id": getattr(r, 'project_id', None),
                    "project_name": r.project.name if getattr(r, 'project', None) else "Project Utama"
                })

        # Urutkan agar model terbaik (is_best=True) tampil paling pertama
        models.sort(key=lambda m: (not m.get("is_best", False)))
    except Exception as e:
        print(f"[InferenceService] Error query runs: {e}")

    return models

def calculate_box_iou(box1, box2):
    """
    Menghitung IoU (Intersection over Union) antara dua bounding box [x1, y1, x2, y2].
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return float(inter_area) / float(union_area) if union_area > 0 else 0.0

def parse_yolo_label(label_text, img_w, img_h, class_names=None):
    """
    Parse isi file label .txt format YOLO menjadi list of bounding box [x1, y1, x2, y2].
    Format baris YOLO: <class_id> <x_center> <y_center> <width> <height> (0.0 - 1.0)
    """
    boxes = []
    if not label_text:
        return boxes
        
    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cid = int(float(parts[0]))
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                
                x1 = int(round(xc - w / 2.0))
                y1 = int(round(yc - h / 2.0))
                x2 = int(round(xc + w / 2.0))
                y2 = int(round(yc + h / 2.0))
                
                # Batasi dalam dimensi gambar
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))
                
                cname = class_names.get(cid, str(cid)) if class_names else str(cid)
                boxes.append({
                    "class_id": cid,
                    "class_name": cname,
                    "box": [x1, y1, x2, y2]
                })
            except Exception:
                continue
    return boxes

def evaluate_predictions_against_ground_truth(gt_boxes, pred_boxes, iou_thresh=0.45):
    """
    Mengevaluasi kesesuaian antara prediksi YOLO dan label Ground Truth pembanding:
    - Status 'OK' jika semua box terdeteksi dengan kelas dan posisi yang cocok persis.
    - Status 'NG' jika ada salah kelas, missed box, atau false positive box.
    """
    matched_pairs = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # Hitung semua pairwise IoU
    candidates = []
    for g_idx, g in enumerate(gt_boxes):
        for p_idx, p in enumerate(pred_boxes):
            iou = calculate_box_iou(g["box"], p["box"])
            if iou >= iou_thresh:
                candidates.append((iou, g_idx, p_idx))
                
    # Urutkan IoU tertinggi lebih dulu (Greedy matching)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    class_mismatches = []
    for iou, g_idx, p_idx in candidates:
        if g_idx not in matched_gt_indices and p_idx not in matched_pred_indices:
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(p_idx)
            
            g = gt_boxes[g_idx]
            p = pred_boxes[p_idx]
            
            # Cek kesesuaian kelas
            same_class = (g["class_id"] == p["class_id"]) or (
                str(g.get("class_name", "")).strip().lower() == str(p.get("class_name", "")).strip().lower()
            )
            
            if same_class:
                matched_pairs.append({
                    "gt": g,
                    "pred": p,
                    "iou": round(iou, 3),
                    "status": "MATCH"
                })
            else:
                class_mismatches.append({
                    "gt": g,
                    "pred": p,
                    "iou": round(iou, 3),
                    "status": "CLASS_MISMATCH"
                })
                
    unmatched_gt = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt_indices]
    unmatched_pred = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred_indices]
    
    # Penentuan status OK vs NG
    is_ok = (len(class_mismatches) == 0 and len(unmatched_gt) == 0 and len(unmatched_pred) == 0)
    
    reasons = []
    if is_ok:
        if len(gt_boxes) == 0:
            reasons.append("Sesuai: Tidak ada objek pada label maupun deteksi model.")
        else:
            avg_iou = round(sum(p["iou"] for p in matched_pairs) / len(matched_pairs) * 100, 1) if matched_pairs else 100
            reasons.append(f"Semua {len(gt_boxes)} objek terdeteksi sempurna sesuai label (Rata-rata IoU: {avg_iou}%).")
    else:
        if class_mismatches:
            cm_desc = [f"Label '{cm['gt']['class_name']}' terdeteksi '{cm['pred']['class_name']}'" for cm in class_mismatches]
            reasons.append(f"Salah Kelas: {', '.join(cm_desc)}")
        if unmatched_gt:
            reasons.append(f"Kurang Deteksi: {len(unmatched_gt)} objek label tidak terdeteksi (missed)")
        if unmatched_pred:
            reasons.append(f"Lebih Deteksi: {len(unmatched_pred)} deteksi ekstra yang tidak ada di label pembanding")
            
    verdict = "OK" if is_ok else "NG"
    detail_text = "; ".join(reasons)
    
    return {
        "verdict": verdict,
        "detail": detail_text,
        "matched_count": len(matched_pairs),
        "class_mismatch_count": len(class_mismatches),
        "missed_count": len(unmatched_gt),
        "false_alarm_count": len(unmatched_pred),
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "matched_pairs": matched_pairs,
        "class_mismatches": class_mismatches,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred
    }

def draw_ground_truth_and_verdict(img_bgr, gt_boxes, eval_result):
    """
    Menggambar visualisasi bounding box Ground Truth (garis Cyan/Gold)
    dan stempel status [ OK - PASSED ] atau [ NG - MISMATCH ] pada gambar.
    """
    annotated = img_bgr.copy()
    
    # 1. Gambar Ground Truth boxes (garis cyan dengan tag GT: <nama>)
    for g in gt_boxes:
        box = g["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Warna BGR Cyan Cerah: (255, 220, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 0), 2)
        
        gt_label = f"GT: {g.get('class_name', g.get('class_id', ''))}"
        (tw, th), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 4), (255, 220, 0), -1)
        cv2.putText(annotated, gt_label, (x1 + 3, tag_y1 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 25), 1, cv2.LINE_AA)
        
    # 2. Gambar Banner Stempel Status di Pojok Kiri Atas
    verdict = eval_result.get("verdict", "OK")
    if verdict == "OK":
        badge_text = "OK - PASSED"
        bg_color = (34, 197, 94)    # Emerald green in BGR
        border_color = (22, 163, 74)
    else:
        badge_text = "NG - MISMATCH"
        bg_color = (59, 68, 239)    # Red in BGR
        border_color = (29, 38, 220)
        
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness = 2
    (bw, bh), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
    
    pad_x, pad_y = 12, 8
    badge_w = bw + (pad_x * 2)
    badge_h = bh + (pad_y * 2)
    
    start_x, start_y = 14, 14
    # Shadow
    cv2.rectangle(annotated, (start_x - 1, start_y - 1), (start_x + badge_w + 1, start_y + badge_h + 1), (0, 0, 0), -1)
    # Background
    cv2.rectangle(annotated, (start_x, start_y), (start_x + badge_w, start_y + badge_h), bg_color, -1)
    cv2.rectangle(annotated, (start_x, start_y), (start_x + badge_w, start_y + badge_h), border_color, 1)
    # Text
    cv2.putText(annotated, badge_text, (start_x + pad_x, start_y + pad_y + bh - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return annotated

def get_dataset_split_samples(dataset_id, split='val', limit=100):
    """
    Mengambil file gambar dan label .txt pembanding langsung dari folder dataset studio.
    """
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return [], {}
        
    storage_path = dataset.storage_path
    extracted_path = os.path.join(storage_path, "extracted")
    
    class_names = {}
    if dataset.class_names_json:
        try:
            raw_classes = json.loads(dataset.class_names_json)
            class_names = {int(k): str(v) for k, v in raw_classes.items()}
        except Exception:
            pass

    images_dir = None
    labels_dir = None
    
    # 1. Cari folder images/<split>
    target_split = split.lower().strip()
    for root, dirs, _ in os.walk(extracted_path):
        norm_root = root.replace("\\", "/").lower()
        if f"/images/{target_split}" in norm_root or norm_root.endswith(f"/images/{target_split}"):
            images_dir = root
            # Cari folder labels yang berpasangan
            pot_label = root.replace("images", "labels").replace("Images", "Labels")
            if os.path.exists(pot_label):
                labels_dir = pot_label
            break
            
    if not images_dir or not os.path.exists(images_dir):
        # Fallback pencarian folder split
        for root, dirs, _ in os.walk(extracted_path):
            norm_root = root.replace("\\", "/").lower()
            if os.path.basename(root).lower() == target_split and "image" in norm_root:
                images_dir = root
                break

    if not images_dir or not os.path.exists(images_dir):
        return [], class_names
        
    if not labels_dir or not os.path.exists(labels_dir):
        for root, dirs, _ in os.walk(extracted_path):
            norm_root = root.replace("\\", "/").lower()
            if os.path.basename(root).lower() == target_split and "label" in norm_root:
                labels_dir = root
                break

    samples = []
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted(os.listdir(images_dir))
    
    for img_file in image_files:
        ext = os.path.splitext(img_file)[1].lower()
        if ext not in allowed_exts:
            continue
            
        stem = os.path.splitext(img_file)[0]
        img_full_path = os.path.join(images_dir, img_file)
        
        lbl_content = None
        has_gt = False
        if labels_dir:
            lbl_file = os.path.join(labels_dir, f"{stem}.txt")
            if os.path.exists(lbl_file):
                has_gt = True
                try:
                    with open(lbl_file, "r", encoding="utf-8", errors="ignore") as lf:
                        lbl_content = lf.read()
                except Exception:
                    lbl_content = ""
                    
        samples.append({
            "filename": img_file,
            "image_path": img_full_path,
            "label_content": lbl_content,
            "has_gt": has_gt
        })
        
        if limit and len(samples) >= limit:
            break
            
    return samples, class_names

def run_batch_inference(model_path, files=None, label_files=None, dataset_id=None, split='val', limit=100, conf=0.25, iou=0.45, imgsz=640, device='0', eval_mode=True):
    """
    Menjalankan inferensi YOLO pada daftar file gambar yang diunggah atau dari dataset studio.
    Mendukung dua tipe:
    - eval_mode=True  : Evaluasi akurasi terhadap label ground-truth (OK vs NG).
    - eval_mode=False : Deteksi langsung murni (pure inference) tanpa pembandingan / badge status.
    Menghasilkan folder dataset terstruktur: images/ (asli), labels/ (.txt format YOLO), classes.txt, dan visualisasi/.
    """
    session_id = f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    session_dir = os.path.join(Config.STORAGE_ROOT, "temp_inference", session_id)
    input_dir = os.path.join(session_dir, "input")
    labels_dir = os.path.join(session_dir, "labels")
    annotated_dir = os.path.join(session_dir, "annotated")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    saved_samples = []  # list of dict: {"filename", "image_path", "label_content", "has_gt"}
    allowed_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    dataset_class_names = {}

    # A. Jika pengujian bersumber dari Dataset Studio
    if dataset_id:
        ds_samples, ds_classes = get_dataset_split_samples(dataset_id, split=split, limit=int(limit) if limit else 100)
        dataset_class_names = ds_classes
        if not ds_samples:
            raise ValueError(f"Tidak ada gambar valid yang ditemukan pada split '{split}' di dataset tersebut.")
            
        for s in ds_samples:
            dest_img = os.path.join(input_dir, s["filename"])
            shutil.copy2(s["image_path"], dest_img)
            saved_samples.append({
                "filename": s["filename"],
                "image_path": dest_img,
                "label_content": s["label_content"],
                "has_gt": s.get("has_gt", False)
            })

    # B. Jika pengujian bersumber dari Upload File / Folder Manual
    elif files:
        # Kumpulkan label files jika ada (hanya diproses jika eval_mode aktif)
        label_map = {}  # {stem: content}
        
        all_incoming_files = list(files)
        if label_files and eval_mode:
            all_incoming_files.extend(label_files)
            
        if eval_mode:
            for f in all_incoming_files:
                if not f or not f.filename:
                    continue
                ext = os.path.splitext(f.filename)[1].lower()
                if ext == '.txt':
                    stem = os.path.splitext(os.path.basename(f.filename))[0]
                    try:
                        content = f.read().decode('utf-8', errors='ignore')
                        label_map[stem] = content
                        f.seek(0)
                    except Exception:
                        pass

        # Simpan gambar asli ke folder input (tanpa bounding box)
        for f in files:
            if not f or not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in allowed_img_exts:
                continue
                
            filename = secure_filename(os.path.basename(f.filename))
            if not filename:
                filename = f"img_{uuid.uuid4().hex[:6]}{ext}"
                
            save_path = os.path.join(input_dir, filename)
            f.save(save_path)
            
            stem = os.path.splitext(filename)[0]
            has_gt = (eval_mode and stem in label_map)
            lbl_content = label_map.get(stem, "") if eval_mode else ""
            
            saved_samples.append({
                "filename": filename,
                "image_path": save_path,
                "label_content": lbl_content,
                "has_gt": has_gt
            })
            
    if not saved_samples:
        raise ValueError("Tidak ada gambar valid yang dapat diproses.")

    # Load Model
    device_param = device
    if device == '0' or device == 0:
        try:
            import torch
            if not torch.cuda.is_available():
                device_param = 'cpu'
        except Exception:
            device_param = 'cpu'

    model = YOLO(model_path)
    model_names = model.names or {}
    
    # Gabungkan class names: model names diutamakan, dataset names sebagai pelengkap
    merged_class_names = {**dataset_class_names, **model_names}

    # Simpan file classes.txt di session directory untuk kebutuhan ekspor dataset
    try:
        classes_file = os.path.join(session_dir, "classes.txt")
        with open(classes_file, "w", encoding="utf-8") as cf:
            # Urutkan berdasarkan index ID kelas
            sorted_keys = sorted(merged_class_names.keys(), key=lambda x: int(x) if str(x).isdigit() else 9999)
            for k in sorted_keys:
                cf.write(f"{merged_class_names[k]}\n")
    except Exception as e:
        print(f"[InferenceService] Gagal menulis classes.txt: {e}")

    # Jalankan inferensi & evaluasi
    results_list = []
    total_detections = 0
    class_counter = Counter()
    total_latency_ms = 0.0
    
    total_ok = 0
    total_ng = 0
    has_any_gt = False
    
    for sample in saved_samples:
        filename = sample["filename"]
        file_path = sample["image_path"]
        lbl_content = sample["label_content"]
        has_gt = sample.get("has_gt", False) and eval_mode
        stem = os.path.splitext(filename)[0]
        
        t0 = time.time()
        predict_results = model.predict(
            source=file_path,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=device_param,
            save=False,
            verbose=False
        )
        t_elapsed = (time.time() - t0) * 1000.0
        total_latency_ms += t_elapsed
        
        res = predict_results[0]
        orig_img_bgr = res.orig_img  # NumPy BGR image
        h_orig, w_orig = orig_img_bgr.shape[:2]
        
        # Parse detections & siapkan data label YOLO
        detections = []
        img_class_counts = Counter()
        yolo_label_lines = []
        
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            cls_ids = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy_list = boxes.xyxy.int().cpu().tolist()
            
            # Coba ambil normalized xywhn dari ultralytics boxes
            try:
                xywhn_list = boxes.xywhn.cpu().tolist()
            except Exception:
                xywhn_list = None
            
            for idx, (cid, c_conf, coords) in enumerate(zip(cls_ids, confs, xyxy_list)):
                cname = merged_class_names.get(cid, f"Class {cid}")
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(c_conf * 100, 1),
                    "box": coords
                })
                img_class_counts[cname] += 1
                class_counter[cname] += 1
                total_detections += 1
                
                # Hitung format YOLO (class_id x_center y_center width height) ter-normalisasi 0..1
                if xywhn_list and idx < len(xywhn_list):
                    xc, yc, bw, bh = xywhn_list[idx]
                else:
                    x1, y1, x2, y2 = coords
                    xc = ((x1 + x2) / 2.0) / float(w_orig)
                    yc = ((y1 + y2) / 2.0) / float(h_orig)
                    bw = (x2 - x1) / float(w_orig)
                    bh = (y2 - y1) / float(h_orig)
                    
                # Pastikan batas normalisasi 0..1
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                yolo_label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Simpan file label YOLO (.txt) untuk dataset re-usable
        pred_label_file = os.path.join(labels_dir, f"{stem}.txt")
        try:
            with open(pred_label_file, "w", encoding="utf-8") as lf:
                if yolo_label_lines:
                    lf.write("\n".join(yolo_label_lines) + "\n")
                else:
                    # File kosong untuk background image / tanpa deteksi
                    lf.write("")
        except Exception as e:
            print(f"[InferenceService] Gagal menulis label {pred_label_file}: {e}")

        # Plot dasar YOLO
        annotated_bgr = res.plot()
        
        # Cek apakah evaluasi perbandingan aktif & ada label pembanding (Ground Truth)
        evaluation = None
        if eval_mode and has_gt:
            has_any_gt = True
            gt_boxes = parse_yolo_label(lbl_content or "", w_orig, h_orig, class_names=merged_class_names)
            eval_res = evaluate_predictions_against_ground_truth(gt_boxes, detections, iou_thresh=float(iou))
            
            if eval_res["verdict"] == "OK":
                total_ok += 1
            else:
                total_ng += 1
                
            # Gambar visualisasi Ground Truth & cap stempel status OK/NG
            annotated_bgr = draw_ground_truth_and_verdict(annotated_bgr, gt_boxes, eval_res)
            evaluation = eval_res

        annotated_filename = f"pred_{filename}"
        annotated_path = os.path.join(annotated_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_bgr)
        
        speed_info = getattr(res, 'speed', {})
        speed_ms = round(sum(speed_info.values()) if speed_info else t_elapsed, 1)
        
        results_list.append({
            "original_filename": filename,
            "annotated_filename": annotated_filename,
            "annotated_url": f"/testing/image/{session_id}/{annotated_filename}",
            "detection_count": len(detections),
            "detections": detections,
            "class_counts": dict(img_class_counts),
            "speed_ms": speed_ms,
            "has_evaluation": (eval_mode and has_gt),
            "evaluation": evaluation
        })
        
    avg_latency = round(total_latency_ms / len(saved_samples), 1) if saved_samples else 0.0
    total_evaluated = total_ok + total_ng
    pass_rate = round((total_ok / total_evaluated * 100), 1) if total_evaluated > 0 else None

    return {
        "session_id": session_id,
        "total_images": len(saved_samples),
        "total_detections": total_detections,
        "class_summary": dict(class_counter),
        "avg_latency_ms": avg_latency,
        "has_evaluation": has_any_gt,
        "eval_mode": eval_mode,
        "evaluation_summary": {
            "total_evaluated": total_evaluated,
            "total_ok": total_ok,
            "total_ng": total_ng,
            "pass_rate": pass_rate
        } if eval_mode else None,
        "results": results_list
    }

def create_session_zip(session_id):
    """
    Mengompres dataset hasil pengujian ke ZIP file yang terstruktur rapi dan siap diolah kembali:
    - images/      : File gambar asli (original tanpa bounding box / bersih)
    - labels/      : File label .txt format YOLO (class_id xc yc w h) hasil prediksi
    - visualisasi/ : File gambar teranotasi (ber-bounding box untuk referensi cepat)
    - classes.txt  : Daftar nama kelas dataset
    """
    session_dir = os.path.join(Config.STORAGE_ROOT, "temp_inference", session_id)
    input_dir = os.path.join(session_dir, "input")
    labels_dir = os.path.join(session_dir, "labels")
    annotated_dir = os.path.join(session_dir, "annotated")
    classes_file = os.path.join(session_dir, "classes.txt")
    
    if not os.path.exists(session_dir):
        return None
        
    zip_filename = f"dataset_test_results_{session_id}.zip"
    zip_path = os.path.join(session_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Gambar Asli (images/) — Bersih tanpa bounding box
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                fpath = os.path.join(input_dir, file)
                if os.path.isfile(fpath):
                    zipf.write(fpath, arcname=f"images/{file}")
                    
        # 2. Label Prediksi Format YOLO (labels/) — .txt normalized
        if os.path.exists(labels_dir):
            for file in os.listdir(labels_dir):
                fpath = os.path.join(labels_dir, file)
                if os.path.isfile(fpath):
                    zipf.write(fpath, arcname=f"labels/{file}")
                    
        # 3. classes.txt — Daftar nama kelas
        if os.path.exists(classes_file):
            zipf.write(classes_file, arcname="classes.txt")
            
        # 4. Visualisasi (visualisasi/) — Gambar ber-bounding box untuk preview
        if os.path.exists(annotated_dir):
            for file in os.listdir(annotated_dir):
                fpath = os.path.join(annotated_dir, file)
                if os.path.isfile(fpath):
                    zipf.write(fpath, arcname=f"visualisasi/{file}")
                
    return zip_path


# =====================================================================
# Realtime Live Webcam Inference with Memory Model Cache
# =====================================================================
import base64
import numpy as np

_LIVE_MODEL_CACHE = {}

def get_cached_yolo_model(model_path):
    """
    Mengambil instance model YOLO dari memory cache agar inferensi realtime webcam
    tidak terbebani proses pembacaan file .pt dari disk berulang-ulang.
    """
    global _LIVE_MODEL_CACHE
    abs_path = os.path.abspath(model_path)
    
    if abs_path in _LIVE_MODEL_CACHE:
        return _LIVE_MODEL_CACHE[abs_path]
        
    # Batasi cache maksimal 2 model untuk menghemat VRAM
    if len(_LIVE_MODEL_CACHE) >= 2:
        oldest_key = min(_LIVE_MODEL_CACHE.keys(), key=lambda k: _LIVE_MODEL_CACHE[k].get('last_used', 0))
        del _LIVE_MODEL_CACHE[oldest_key]
        
    model = YOLO(abs_path)
    _LIVE_MODEL_CACHE[abs_path] = {
        'model': model,
        'last_used': time.time()
    }
    return _LIVE_MODEL_CACHE[abs_path]

def predict_single_frame(model_path, frame_data, conf=0.25, iou=0.45, imgsz=640, device='0'):
    """
    Inferensi real-time ultra-cepat untuk 1 frame webcam dari client.
    Mendukung frame_data berupa:
    - bytes (multipart file upload)
    - str (base64 dataURL)
    - numpy array (BGR image)
    """
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {model_path}")
        
    # 1. Decode frame ke BGR
    img = None
    if isinstance(frame_data, np.ndarray):
        img = frame_data
    elif isinstance(frame_data, bytes):
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(frame_data, str):
        # Base64 string
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        decoded = base64.b64decode(frame_data)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    if img is None:
        raise ValueError("Gagal mendecode frame gambar dari webcam.")
        
    h, w = img.shape[:2]
    
    # 2. Ambil model dari cache
    cached = get_cached_yolo_model(model_path)
    model = cached['model']
    cached['last_used'] = time.time()
    
    # 3. Tentukan device
    device_param = device
    if device == '0' or device == 0:
        try:
            import torch
            if not torch.cuda.is_available():
                device_param = 'cpu'
        except Exception:
            device_param = 'cpu'
            
    # 4. Inferensi
    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device_param,
        verbose=False
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    
    # 5. Ekstrak deteksi
    detections = []
    class_counts = Counter()
    res = results[0]
    names = res.names or {}
    
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            c = float(box.conf[0].cpu().numpy())
            cid = int(box.cls[0].cpu().numpy())
            cname = names.get(cid, str(cid))
            class_counts[cname] += 1
            
            detections.append({
                "box": [round(coords[0], 1), round(coords[1], 1), round(coords[2], 1), round(coords[3], 1)],
                "conf": round(c, 3),
                "class_id": cid,
                "class_name": cname
            })
            
    return {
        "success": True,
        "detections": detections,
        "counts": dict(class_counts),
        "total_detections": len(detections),
        "latency_ms": latency_ms,
        "width": w,
        "height": h
    }


