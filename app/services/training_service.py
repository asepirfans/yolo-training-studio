import os
import sys
import json
import time
import uuid
import shutil
import subprocess
import psutil
from datetime import datetime
from config import RUNS_DIR, Config
from ..models import db
from ..models.training_run import TrainingRun

def generate_run_id():
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:4]
    return f"run_{now_str}_{short_uuid}"

def get_run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)

def create_training_run(dataset, payload):
    """
    Membuat training run baru dan menyimpan snapshot konfigurasi (PRD Section 21-23, 68).
    """
    run_id = generate_run_id()
    run_dir = get_run_dir(run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Salin dataset_snapshot.yaml jika ada
    if dataset.yaml_path and os.path.exists(dataset.yaml_path):
        try:
            shutil.copy2(dataset.yaml_path, os.path.join(run_dir, "dataset_snapshot.yaml"))
        except Exception:
            pass

    # 2. Susun konfigurasi lengkap
    config = {
        "run_id": run_id,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_yaml": dataset.yaml_path,
        "model": payload.get("model", Config.DEFAULT_MODEL),
        "training": payload.get("training", {}),
        "optimization": payload.get("optimization", {}),
        "augmentation": payload.get("augmentation", {}),
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "env_info": {
            "python_version": sys.version.split()[0],
            "app_version": "1.0.0"
        }
    }

    config_file = os.path.join(run_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    status_file = os.path.join(run_dir, "status.json")
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({"status": "QUEUED", "current_epoch": 0, "total_epochs": config["training"].get("epochs", 100)}, f, indent=2)

    # Buat file training.log kosong
    log_file = os.path.join(run_dir, "training.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== TRAINING LOG: {run_id} ===\n")

    # 3. Simpan ke database
    total_ep = int(config["training"].get("epochs", 100))
    run_record = TrainingRun(
        id=run_id,
        dataset_id=dataset.id,
        model_name=config["model"],
        status="QUEUED",
        config_json=json.dumps(config),
        current_epoch=0,
        total_epochs=total_ep,
        output_path=os.path.join(run_dir, "outputs", "train"),
        project_id=payload.get("project_id") or dataset.project_id
    )
    db.session.add(run_record)
    db.session.commit()

    # Cek apakah bisa langsung dijalankan (jika tidak ada run aktif)
    check_and_start_next_run()

    return run_record

def check_and_start_next_run():
    """
    Manajemen antrean (PRD Section 59): MAX_CONCURRENT_TRAINING=1.
    """
    active_count = TrainingRun.query.filter(TrainingRun.status.in_(['RUNNING', 'STARTING'])).count()
    if active_count < Config.MAX_CONCURRENT_TRAINING:
        next_run = TrainingRun.query.filter_by(status='QUEUED').order_by(TrainingRun.created_at.asc()).first()
        if next_run:
            start_training_process(next_run.id)

def start_training_process(run_id):
    """
    Menjalankan training_worker.py dalam subprocess background (PRD Section 21).
    """
    run_record = TrainingRun.query.get(run_id)
    if not run_record:
        return False

    run_dir = get_run_dir(run_id)
    worker_script = os.path.join(os.path.dirname(__file__), "training_worker.py")

    run_record.status = "STARTING"
    run_record.started_at = datetime.utcnow()
    db.session.commit()

    try:
        # Jalankan background process tanpa blocking HTTP
        process = subprocess.Popen(
            [sys.executable, worker_script, run_dir],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )

        run_record.pid = process.pid
        run_record.status = "RUNNING"
        db.session.commit()

        # Simpan PID ke file
        with open(os.path.join(run_dir, "pid"), "w", encoding="utf-8") as pf:
            pf.write(str(process.pid))

        return True
    except Exception as e:
        run_record.status = "FAILED"
        run_record.error_message = f"Gagal menjalankan worker process: {str(e)}"
        db.session.commit()
        return False

def stop_training_run(run_id):
    """
    Menghentikan proses training beserta seluruh anak prosesnya secara aman (PRD Section 30).
    """
    run_record = TrainingRun.query.get(run_id)
    if not run_record:
        return False, "Run tidak ditemukan"

    run_dir = get_run_dir(run_id)
    pid = run_record.pid

    # Cek status file
    if pid:
        try:
            parent = psutil.Process(pid)
            # Hentikan semua child process terlebih dahulu (seperti PyTorch multiprocessing workers)
            children = parent.children(recursive=True)
            for ch in children:
                try:
                    ch.terminate()
                except Exception:
                    pass
            parent.terminate()
            parent.wait(timeout=3)
        except Exception:
            try:
                # Force kill jika terminate tidak merespon
                parent = psutil.Process(pid)
                parent.kill()
            except Exception:
                pass

    run_record.status = "STOPPED"
    run_record.finished_at = datetime.utcnow()
    if run_record.started_at:
        run_record.duration_seconds = int((run_record.finished_at - run_record.started_at).total_seconds())
    db.session.commit()

    # Update status.json
    status_file = os.path.join(run_dir, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                cur = json.load(f)
            cur["status"] = "STOPPED"
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(cur, f, indent=2)
        except Exception:
            pass

    # Lanjutkan run berikutnya jika ada di antrean
    check_and_start_next_run()

    return True, "Training berhasil dihentikan"

def sync_run_state(run_record):
    """
    Sinkronisasi status antara file status.json di disk dengan database.
    """
    run_dir = get_run_dir(run_record.id)
    status_file = os.path.join(run_dir, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("current_epoch") is not None:
                run_record.current_epoch = int(data["current_epoch"])
            if data.get("total_epochs") is not None:
                run_record.total_epochs = int(data["total_epochs"])

            disk_status = data.get("status")
            if disk_status in ["COMPLETED", "FAILED", "STOPPED"] and run_record.status != disk_status:
                run_record.status = disk_status
                if run_record.started_at and not run_record.finished_at:
                    run_record.finished_at = datetime.utcnow()
                    run_record.duration_seconds = int((run_record.finished_at - run_record.started_at).total_seconds())
                if data.get("error"):
                    run_record.error_message = data["error"]
                db.session.commit()
                # Cek antrean untuk run berikutnya
                check_and_start_next_run()
        except Exception:
            pass

def generate_sse_logs(app, run_id):
    """
    Generator log realtime menggunakan Server-Sent Events (PRD Section 29).
    Membaca langsung dari training.log dan status.json tanpa ketergantungan DB session per loop.
    """
    run_dir = get_run_dir(run_id)
    log_path = os.path.join(run_dir, "training.log")
    status_file = os.path.join(run_dir, "status.json")

    if not os.path.exists(log_path):
        yield "data: Menunggu log dimulai...\n\n"

    last_pos = 0
    idle_cycles = 0

    while True:
        # 1. Baca baris baru dari training.log jika ada
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                lines = f.readlines()
                last_pos = f.tell()

                for line in lines:
                    clean = line.rstrip('\r\n')
                    if clean:
                        yield f"data: {clean}\n\n"
                        idle_cycles = 0

        # 2. Cek status training langsung dari status.json di disk (aman tanpa app context)
        is_finished = False
        finish_status = "UNKNOWN"
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                    st = sdata.get("status", "")
                    if st in ["COMPLETED", "FAILED", "STOPPED", "INTERRUPTED"]:
                        is_finished = True
                        finish_status = st
            except Exception:
                pass

        # 3. Jika selesai, sinkronkan ke database dengan app context lalu akhiri stream
        if is_finished:
            if app:
                try:
                    with app.app_context():
                        rec = TrainingRun.query.get(run_id)
                        if rec:
                            sync_run_state(rec)
                except Exception:
                    pass
            time.sleep(0.5)
            yield f"data: [EOF] Training telah berstatus: {finish_status}\n\n"
            break

        time.sleep(1.0)
        idle_cycles += 1
        # Kirim heartbeat ping setiap 15 detik agar browser tidak memutus koneksi
        if idle_cycles >= 15:
            yield ": ping\n\n"
            idle_cycles = 0
