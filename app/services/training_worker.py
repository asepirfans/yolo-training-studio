import os
import sys
import json
import time
import traceback

class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def update_status(run_dir, updates):
    status_file = os.path.join(run_dir, 'status.json')
    current = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                current = json.load(f)
        except Exception:
            pass
    current.update(updates)
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass

def main():
    if len(sys.argv) < 2:
        print("Penggunaan: python training_worker.py <run_dir>")
        sys.exit(1)

    run_dir = os.path.abspath(sys.argv[1])
    log_file = os.path.join(run_dir, 'training.log')
    config_file = os.path.join(run_dir, 'config.json')

    # Setup Logging
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 70)
    print(f"[Worker] Memulai Background Training Worker di: {run_dir}")
    print(f"[Worker] Waktu Mulai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Worker] PID: {os.getpid()}")
    print("=" * 70)

    if not os.path.exists(config_file):
        print(f"[Error] File konfigurasi {config_file} tidak ditemukan!")
        update_status(run_dir, {"status": "FAILED", "error": "config.json tidak ditemukan", "exit_code": 1})
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    update_status(run_dir, {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "current_epoch": 0,
        "total_epochs": config.get("training", {}).get("epochs", 100)
    })

    try:
        from ultralytics import YOLO
        import torch

        # Parameter model & dataset
        model_name = config.get("model", "yolov8s.pt")
        dataset_yaml = config.get("dataset_yaml")
        training_cfg = config.get("training", {})
        optim_cfg = config.get("optimization", {})
        aug_cfg = config.get("augmentation", {})

        # Auto-fix path di dalam data.yaml jika hardcoded ke direktori yang tidak ditemukan
        if dataset_yaml and os.path.exists(dataset_yaml):
            try:
                import yaml
                with open(dataset_yaml, 'r', encoding='utf-8') as yf:
                    ydata = yaml.safe_load(yf)
                if isinstance(ydata, dict):
                    cpath = ydata.get('path')
                    ydir = os.path.dirname(os.path.abspath(dataset_yaml))
                    if not cpath or not os.path.exists(cpath):
                        ydata['path'] = ydir.replace('\\', '/')
                        with open(dataset_yaml, 'w', encoding='utf-8') as yf:
                            yaml.dump(ydata, yf, sort_keys=False)
                        print(f"[Worker] Auto-fixed dataset YAML path ke: {ydata['path']}")
            except Exception as ye:
                print(f"[Worker Notice] YAML path check: {ye}")

        print(f"[Worker] Loading Model: {model_name}...")
        model = YOLO(model_name)

        # Siapkan argumen training YOLO
        epochs = int(training_cfg.get("epochs", 100))
        batch = int(training_cfg.get("batch", 16)) if str(training_cfg.get("batch")).isdigit() else training_cfg.get("batch", 16)
        imgsz = int(training_cfg.get("imgsz", 640))
        workers = int(training_cfg.get("workers", 0))  # 0 aman di Windows
        patience = int(training_cfg.get("patience", 50))
        seed = int(training_cfg.get("seed", 0))
        cache = bool(training_cfg.get("cache", False))
        pretrained = bool(training_cfg.get("pretrained", True))
        save_period = int(training_cfg.get("save_period", -1))

        # Device
        device_raw = str(training_cfg.get("device", "0"))
        if device_raw.lower() == "cpu" or not torch.cuda.is_available():
            device = 'cpu'
        else:
            device = int(device_raw) if device_raw.isdigit() else device_raw

        # Optimizer
        optimizer = optim_cfg.get("optimizer", "auto")
        lr0 = float(optim_cfg.get("lr0", 0.01))
        lrf = float(optim_cfg.get("lrf", 0.01))
        momentum = float(optim_cfg.get("momentum", 0.937))
        weight_decay = float(optim_cfg.get("weight_decay", 0.0005))
        warmup_epochs = float(optim_cfg.get("warmup_epochs", 3.0))
        warmup_momentum = float(optim_cfg.get("warmup_momentum", 0.8))
        cos_lr = bool(optim_cfg.get("cos_lr", False))
        close_mosaic = int(optim_cfg.get("close_mosaic", 10))

        # Augmentation kwargs
        aug_kwargs = {}
        if aug_cfg.get("enabled", True):
            for key in ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste"]:
                if key in aug_cfg:
                    aug_kwargs[key] = float(aug_cfg[key])
        else:
            # Matikan seluruh augmentasi
            aug_kwargs = {
                "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0,
                "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
                "flipud": 0.0, "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0
            }

        outputs_dir = os.path.join(run_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # Callbacks untuk progress epoch
        def on_fit_epoch_end(trainer):
            ep = trainer.epoch + 1
            tot = trainer.epochs
            update_status(run_dir, {
                "current_epoch": ep,
                "total_epochs": tot
            })

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        print(f"[Worker] Menjalankan training: Epochs={epochs}, Batch={batch}, ImgSz={imgsz}, Device={device}...")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=workers,
            patience=patience,
            seed=seed,
            cache=cache,
            pretrained=pretrained,
            save_period=save_period,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            project=outputs_dir,
            name="train",
            exist_ok=True,
            plots=True,
            verbose=True,
            **aug_kwargs
        )

        print("=" * 70)
        print("[Worker] TRAINING SELESAI DENGAN SUKSES!")
        print(f"[Worker] Waktu Selesai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        update_status(run_dir, {
            "status": "COMPLETED",
            "finished_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exit_code": 0
        })

    except Exception as e:
        err_msg = str(e)
        print("=" * 70)
        print(f"[Worker Error] Terjadi kegagalan saat training: {err_msg}")
        traceback.print_exc()
        print("=" * 70)

        update_status(run_dir, {
            "status": "FAILED",
            "error": err_msg,
            "finished_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exit_code": 1
        })
        sys.exit(1)

if __name__ == '__main__':
    main()
