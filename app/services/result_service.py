import os
import csv
import glob
from config import RUNS_DIR

def parse_results_csv(output_dir):
    """
    Membaca dan mem-parsing results.csv yang dihasilkan Ultralytics (PRD Section 55).
    Mengembalikan riwayat metrik per epoch serta metrik terbaik (best mAP50 & mAP50-95).
    """
    csv_path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(csv_path):
        return {
            "epochs": [],
            "latest": {},
            "best": {
                "epoch": 0,
                "map50": 0.0,
                "map5095": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        }

    epochs_data = []
    best_map50 = 0.0
    best_map5095 = 0.0
    best_epoch = 0
    best_p = 0.0
    best_r = 0.0

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader, [])]
            
            # Map kolom Ultralytics
            def find_col(candidates):
                for cand in candidates:
                    for i, h in enumerate(headers):
                        if cand.lower() in h.lower():
                            return i
                return -1

            col_epoch = find_col(["epoch"])
            col_box_loss = find_col(["train/box_loss", "box_loss"])
            col_cls_loss = find_col(["train/cls_loss", "cls_loss"])
            col_dfl_loss = find_col(["train/dfl_loss", "dfl_loss"])
            col_val_box = find_col(["val/box_loss"])
            col_val_cls = find_col(["val/cls_loss"])
            col_p = find_col(["metrics/precision(B)", "precision"])
            col_r = find_col(["metrics/recall(B)", "recall"])
            col_map50 = find_col(["metrics/mAP50(B)", "mAP50"])
            col_map5095 = find_col(["metrics/mAP50-95(B)", "mAP50-95"])

            best_fitness = -1.0
            best_map50 = 0.0
            best_map5095 = 0.0
            best_epoch = 0
            best_p = 0.0
            best_r = 0.0

            for row in reader:
                if not row or len(row) < len(headers):
                    continue
                def _get(idx, default=0.0):
                    if idx >= 0 and idx < len(row):
                        try:
                            val = row[idx].strip()
                            return float(val) if val not in ('nan', 'inf', '') else default
                        except Exception:
                            return default
                    return default

                ep = int(_get(col_epoch, 0))
                map50 = _get(col_map50, 0.0)
                map5095 = _get(col_map5095, 0.0)
                prec = _get(col_p, 0.0)
                rec = _get(col_r, 0.0)

                item = {
                    "epoch": ep,
                    "train_box_loss": round(_get(col_box_loss, 0.0), 4),
                    "train_cls_loss": round(_get(col_cls_loss, 0.0), 4),
                    "train_dfl_loss": round(_get(col_dfl_loss, 0.0), 4),
                    "val_box_loss": round(_get(col_val_box, 0.0), 4),
                    "val_cls_loss": round(_get(col_val_cls, 0.0), 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "map50": round(map50, 4),
                    "map5095": round(map5095, 4)
                }
                epochs_data.append(item)

                # Catat nilai puncak tertinggi masing-masing metrik
                if map50 > best_map50:
                    best_map50 = map50
                if map5095 > best_map5095:
                    best_map5095 = map5095

                # Evaluasi fitness standar Ultralytics YOLO: 0.1 * mAP50 + 0.9 * mAP50-95
                # Formula ini identik dengan kriteria penentuan bobot 'best.pt' di engine YOLO
                current_fitness = (0.1 * map50) + (0.9 * map5095)
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_epoch = ep
                    best_p = prec
                    best_r = rec

        latest = epochs_data[-1] if epochs_data else {}
        return {
            "epochs": epochs_data,
            "latest": latest,
            "best": {
                "epoch": best_epoch,
                "fitness": round(best_fitness, 4) if best_fitness >= 0 else 0.0,
                "map50": round(best_map50, 4),
                "map5095": round(best_map5095, 4),
                "precision": round(best_p, 4),
                "recall": round(best_r, 4)
            }
        }
    except Exception as e:
        print(f"[Results CSV Error] {e}")
        return {"epochs": [], "latest": {}, "best": {"epoch": 0, "map50": 0.0, "map5095": 0.0}}

def discover_result_artifacts(output_dir):
    """
    Memindai seluruh file gambar hasil evaluasi yang ada di output training (PRD Section 37-39, 56).
    Mendukung format nama YOLO standar maupun variasi YOLOv8 (misal: BoxPR_curve.png).
    """
    artifacts = {
        "summary_chart": None,
        "confusion_matrix": None,
        "confusion_matrix_normalized": None,
        "curves": [],
        "labels": [],
        "val_batches": []
    }

    if not os.path.exists(output_dir):
        return artifacts

    def find_file(patterns):
        for pat in patterns:
            target = os.path.join(output_dir, pat)
            if os.path.exists(target):
                return pat
        return None

    # 1. Summary results.png
    artifacts["summary_chart"] = find_file(["results.png"])

    # 2. Confusion Matrices
    artifacts["confusion_matrix"] = find_file(["confusion_matrix.png"])
    artifacts["confusion_matrix_normalized"] = find_file(["confusion_matrix_normalized.png"])

    # 3. Curves (PR, P, R, F1)
    curve_candidates = [
        ("PR Curve", ["PR_curve.png", "BoxPR_curve.png"]),
        ("F1 Curve", ["F1_curve.png", "BoxF1_curve.png"]),
        ("Precision Curve", ["P_curve.png", "BoxP_curve.png"]),
        ("Recall Curve", ["R_curve.png", "BoxR_curve.png"])
    ]
    for title, patterns in curve_candidates:
        f = find_file(patterns)
        if f:
            artifacts["curves"].append({"title": title, "filename": f})

    # 4. Labels
    for pat in ["labels.jpg", "labels_correlogram.jpg"]:
        if os.path.exists(os.path.join(output_dir, pat)):
            artifacts["labels"].append(pat)

    # 5. Validation predictions vs labels
    val_files = sorted(glob.glob(os.path.join(output_dir, "val_batch*.jpg")))
    for vf in val_files:
        fname = os.path.basename(vf)
        artifacts["val_batches"].append(fname)

    return artifacts

def get_weight_paths(output_dir):
    """
    Mengambil path file best.pt dan last.pt jika tersedia (PRD Section 40).
    """
    weights_dir = os.path.join(output_dir, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    last_pt = os.path.join(weights_dir, "last.pt")

    return {
        "best_available": os.path.exists(best_pt),
        "best_path": best_pt if os.path.exists(best_pt) else None,
        "best_size_mb": round(os.path.getsize(best_pt) / (1024 * 1024), 1) if os.path.exists(best_pt) else 0,
        "last_available": os.path.exists(last_pt),
        "last_path": last_pt if os.path.exists(last_pt) else None,
        "last_size_mb": round(os.path.getsize(last_pt) / (1024 * 1024), 1) if os.path.exists(last_pt) else 0
    }

def find_best_training_run(project_id=None, runs_list=None):
    """
    Menentukan TrainingRun yang secara MANUAL ditandai sebagai terbaik oleh user (is_best == True).
    Hanya mengembalikan run jika user memang telah menyematkan flag terbaik secara manual.
    """
    if runs_list is not None:
        for r in runs_list:
            if r and bool(getattr(r, 'is_best', False)):
                return r
        return None
    else:
        from ..models.training_run import TrainingRun
        query = TrainingRun.query.filter_by(is_best=True)
        if project_id and project_id != 'all':
            query = query.filter_by(project_id=project_id)
        return query.first()
