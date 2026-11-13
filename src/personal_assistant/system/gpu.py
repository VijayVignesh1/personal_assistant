import glob
import json
import os
import platform
import shutil
import subprocess
from typing import Any

from personal_assistant.system.schema import GPUInfo, GPUVendor


def _get_nvidia_smi() -> list[dict[str, Any]]:
  """Queries nvidia-smi on Linux for exact NVIDIA VRAM and driver version."""
  if not shutil.which("nvidia-smi"):
    return []
  try:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    results = []
    for line in out.strip().splitlines():
      parts = [p.strip() for p in line.split(",")]
      if len(parts) == 4:
        results.append({
            "name": str(parts[0]),
            "total_gb": round(float(parts[1]) / 1024, 2),
            "avail_gb": round(float(parts[2]) / 1024, 2),
            "driver": str(parts[3]),
        })
    return results
  except subprocess.CalledProcessError as e: 
    print(f"Error querying nvidia-smi: {e}")
    return []


def _get_cuda_version() -> str | None:
    """Returns CUDA version supported by the installed NVIDIA driver."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
        )

        for line in out.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].strip().split()[0]

    except subprocess.CalledProcessError:
        pass

    return None

def _get_linux_gpus() -> tuple[GPUInfo,...]:
  gpus: tuple[GPUInfo,...] = ()
  nv_gpus = _get_nvidia_smi()
  nv_idx = 0

  vendor_map = {
      "0x10de": "NVIDIA",
      "0x1002": "AMD",
      "0x8086": "Intel",
  }

  for card_path in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
    dev_path = os.path.join(card_path, "device")
    vendor_file = os.path.join(dev_path, "vendor")
    is_cuda = False
    if not os.path.exists(vendor_file):
      continue

    with open(vendor_file, "r", encoding="utf-8") as f:
      hex_id = f.read().strip().lower()
    vendor = vendor_map.get(hex_id, hex_id)

    # Resolve GPU model name via lspci
    slot = os.path.basename(os.readlink(dev_path))
    name = f"{vendor} GPU"
    if shutil.which("lspci"):
      try:
        lspci_out = subprocess.check_output(
            ["lspci", "-s", slot, "-mm"], text=True
        )
        parts = lspci_out.split('"')
        if len(parts) >= 6:
          name = parts[5]
      except subprocess.CalledProcessError as e:
        print(f"Error querying lspci for slot {slot}: {e}")

    total_gb: float | None = None
    avail_gb: float | None = None
    driver: str | None = None

    # 1. NVIDIA via nvidia-smi
    if vendor == "NVIDIA" and nv_idx < len(nv_gpus):
      nv = nv_gpus[nv_idx]
      name = nv["name"]
      total_gb = nv["total_gb"]
      avail_gb = nv["avail_gb"]
      driver = nv["driver"]
      is_cuda = True
      cuda_version = _get_cuda_version()
      nv_idx += 1

    # 2. AMD (and newer Intel Xe) via sysfs VRAM counters
    else:
      vram_total_file = os.path.join(dev_path, "mem_info_vram_total")
      vram_used_file = os.path.join(dev_path, "mem_info_vram_used")
      if os.path.exists(vram_total_file):
        with open(vram_total_file, "r", encoding="utf-8") as f:
          total_bytes = int(f.read().strip())
          total_gb = round(total_bytes / (1024**3), 2)
        if os.path.exists(vram_used_file):
          with open(vram_used_file, "r", encoding="utf-8") as f:
            used_bytes = int(f.read().strip())
            avail_gb = round((total_bytes - used_bytes) / (1024**3), 2)

      # Kernel driver module version/name
      driver_link = os.path.join(dev_path, "driver")
      if os.path.exists(driver_link):
        drv_name = os.path.basename(os.readlink(driver_link))
        ver_file = f"/sys/module/{drv_name}/version"
        if os.path.exists(ver_file):
          with open(ver_file, "r", encoding="utf-8") as f:
            driver = f.read().strip()
        else:
          driver = drv_name

    gpus += (GPUInfo(name, GPUVendor[vendor.upper()] if vendor.upper() in GPUVendor.__members__ else GPUVendor.UNKNOWN, total_gb, avail_gb, driver, is_cuda, cuda_version if is_cuda else None),)

  return gpus


def _get_macos_gpus() -> tuple[GPUInfo,...]:
  gpus: tuple[GPUInfo,...] = ()
  cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
  out = subprocess.check_output(cmd, text=True)
  data = json.loads(out).get("SPDisplaysDataType", [])

  os_ver = platform.mac_ver()[0]

  for item in data:
    name = item.get("sppci_model", "Unknown")
    raw_vendor = item.get("spdisplays_vendor", "")
    vendor = raw_vendor.replace("sppci_vendor_", "").capitalize() or "Apple"

    vram_str = item.get("spdisplays_vram", "")
    total_gb: float | None = None
    if "GB" in vram_str:
      total_gb = float(vram_str.replace("GB", "").strip())
    elif "MB" in vram_str:
      total_gb = round(float(vram_str.replace("MB", "").strip()) / 1024, 2)
    elif vendor == "Apple":
      # Apple Silicon uses Unified Memory
      mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
      total_gb = round(mem_bytes / (1024**3), 2)

    driver = item.get("spdisplays_gmux-version") or f"macOS {os_ver} (Metal)"
    gpus += (GPUInfo(name, GPUVendor[vendor.upper()] if vendor.upper() in GPUVendor.__members__ else GPUVendor.UNKNOWN, total_gb, None, driver),)

  return gpus


def get_gpus() -> tuple[GPUInfo,...]:
  system = platform.system()
  if system == "Linux":
    return _get_linux_gpus()
  if system == "Darwin":
    return _get_macos_gpus()
  return ()


if __name__ == "__main__":
  for gpu in get_gpus():
    print(gpu)
