import os
import subprocess
import psutil

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

def get_gpu_info():
    """
    Mengambil status hardware GPU NVIDIA.
    Prioritas: 1. pynvml (sangat cepat & akurat) -> 2. nvidia-smi -> 3. torch.cuda
    """
    info = {
        "available": False,
        "device_name": "N/A",
        "utilization_gpu": 0,
        "temperature": 0,
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "vram_pct": 0.0,
        "power_draw_w": "N/A",
        "driver_version": "N/A",
        "cuda_version": "N/A",
        "compute_capability": "N/A",
        "fan_speed_pct": "N/A",
        "clock_graphics_mhz": "N/A",
        "clock_mem_mhz": "N/A"
    }

    # Cek CUDA info via PyTorch jika tersedia
    if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
        info["cuda_version"] = getattr(torch.version, 'cuda', 'N/A')
        try:
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
        except Exception:
            pass

    # 1. Coba pynvml
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["available"] = True
            info["device_name"] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(info["device_name"], bytes):
                info["device_name"] = info["device_name"].decode('utf-8')
            
            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["vram_total_mb"] = int(mem.total / (1024 * 1024))
            info["vram_used_mb"] = int(mem.used / (1024 * 1024))
            info["vram_free_mb"] = max(0, info["vram_total_mb"] - info["vram_used_mb"])
            if info["vram_total_mb"] > 0:
                info["vram_pct"] = round((info["vram_used_mb"] / info["vram_total_mb"]) * 100, 1)

            # Utilization & Temp & Power
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["utilization_gpu"] = int(util.gpu)
            info["temperature"] = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                info["power_draw_w"] = f"{round(power_mw / 1000.0, 1)} W"
            except Exception:
                pass

            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                info["driver_version"] = str(drv)
            except Exception:
                pass

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                info["fan_speed_pct"] = int(fan)
            except Exception:
                pass

            try:
                info["clock_graphics_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                info["clock_mem_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            except Exception:
                pass

            return info
        except Exception:
            pass

    # 2. Fallback nvidia-smi
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,memory.free,power.draw", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL, timeout=2).strip()
        lines = output.splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(',')]
            if len(parts) >= 6:
                info["available"] = True
                info["device_name"] = parts[0]
                def _to_num(v, is_float=False):
                    clean = v.replace('[N/A]', '').replace('N/A', '').strip()
                    try:
                        return float(clean) if is_float else int(float(clean))
                    except Exception:
                        return 0
                info["utilization_gpu"] = _to_num(parts[1])
                info["temperature"] = _to_num(parts[2])
                info["vram_used_mb"] = _to_num(parts[3])
                info["vram_total_mb"] = _to_num(parts[4])
                info["vram_free_mb"] = _to_num(parts[5])
                if len(parts) >= 7 and parts[6] != '[N/A]':
                    info["power_draw_w"] = f"{parts[6]} W"
                if info["vram_total_mb"] > 0:
                    info["vram_pct"] = round((info["vram_used_mb"] / info["vram_total_mb"]) * 100, 1)
                return info
    except Exception:
        pass

    # 3. Fallback torch.cuda
    if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
        try:
            info["available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            total_b = torch.cuda.get_device_properties(0).total_memory
            res_b = torch.cuda.memory_reserved(0)
            info["vram_total_mb"] = int(total_b / (1024 * 1024))
            info["vram_used_mb"] = int(res_b / (1024 * 1024))
            info["vram_free_mb"] = max(0, info["vram_total_mb"] - info["vram_used_mb"])
            if info["vram_total_mb"] > 0:
                info["vram_pct"] = round((info["vram_used_mb"] / info["vram_total_mb"]) * 100, 1)
        except Exception:
            pass

    return info

def get_system_resources():
    """
    Mengambil penggunaan CPU dan RAM sistem via psutil.
    """
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()

    return {
        "cpu_pct": round(cpu_pct, 1),
        "cpu_count": cpu_count,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_pct": round(mem.percent, 1)
    }

def get_available_devices():
    """
    Menghasilkan daftar opsi device untuk form training.
    """
    options = [{"id": "cpu", "name": "CPU"}]
    if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            dname = torch.cuda.get_device_name(i)
            options.append({
                "id": str(i),
                "name": f"GPU {i} — {dname}"
            })
    return options
