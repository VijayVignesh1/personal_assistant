import asyncio

import psutil
import pynvml  # type: ignore[import-untyped]


class SystemMonitor:
    """System monitor for GPU, CPU, and RAM utilization."""
    def __init__(self, gpu_id: int = 0, interval: float = 0.1) -> None:
        """Initialize the system monitor.

        Args:
            gpu_id (int): The ID of the GPU to monitor.
            interval (float): The interval in seconds between measurements.
        """
        self.gpu_id = gpu_id
        self.interval = interval

        self.samples: list[dict[str, float]] = []
        self.running = False
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        pynvml.nvmlInit()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.gpu_id
        )

        # Initialize CPU measurement
        psutil.cpu_percent(interval=None)

        self.running = True
        self.task = asyncio.create_task(self._monitor())

    async def _monitor(self) -> None:

        while self.running:

            # GPU utilization
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(
                self.handle
            )

            # GPU memory
            memory = pynvml.nvmlDeviceGetMemoryInfo(
                self.handle
            )

            # CPU and RAM
            cpu_percent = psutil.cpu_percent(interval=None)

            ram_percent = psutil.virtual_memory().percent

            self.samples.append({
                "gpu_util": gpu_util.gpu,
                "gpu_memory_util": gpu_util.memory,
                "gpu_memory_used": memory.used / (1024 ** 2),
                "gpu_memory_total": memory.total / (1024 ** 2),
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent
            })

            await asyncio.sleep(self.interval)

    async def stop(self) -> dict[str, float]:

        self.running = False

        if self.task:
            await self.task

        pynvml.nvmlShutdown()

        if not self.samples:
            return {}

        def avg(key: str) -> float:
            return sum(s[key] for s in self.samples) / len(self.samples)

        def maximum(key: str) -> float:
            return max(s[key] for s in self.samples)

        return {
            "avg_gpu_util": avg("gpu_util"),
            "max_gpu_util": maximum("gpu_util"),

            "avg_gpu_memory_mb": avg("gpu_memory_used"),
            "max_gpu_memory_mb": maximum("gpu_memory_used"),

            "avg_gpu_memory_util": avg("gpu_memory_util"),

            "avg_cpu_util": avg("cpu_percent"),
            "max_cpu_util": maximum("cpu_percent"),

            "avg_ram_util": avg("ram_percent"),
            "max_ram_util": maximum("ram_percent")
        }