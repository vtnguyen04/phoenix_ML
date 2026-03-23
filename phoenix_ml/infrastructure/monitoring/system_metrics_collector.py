"""Background collector for system resource metrics.

Registers Prometheus gauges for CPU, memory, disk, process RSS, open FDs,
and GPU (utilization, memory, temperature). Gauges are updated on a
configurable interval via an asyncio background task.

Dependencies:
    - ``psutil`` (required): CPU, memory, disk, process metrics.
    - ``pynvml`` (optional): NVIDIA GPU metrics; skipped if unavailable.

Lifecycle:
    Instantiate ``SystemMetricsCollector``, call ``start()`` during app
    startup and ``stop()`` during shutdown. Managed by ``lifespan.py``.
"""

import asyncio
import logging
import os

import psutil
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# ── System Resource Gauges ───────────────────────────────────────

SYSTEM_CPU_PERCENT = Gauge(
    "system_cpu_usage_percent",
    "CPU usage percentage (all cores)",
)
SYSTEM_CPU_PER_CORE = Gauge(
    "system_cpu_per_core_percent",
    "CPU usage percentage per core",
    ["core"],
)
SYSTEM_MEMORY_USED_BYTES = Gauge(
    "system_memory_used_bytes",
    "Physical memory used in bytes",
)
SYSTEM_MEMORY_TOTAL_BYTES = Gauge(
    "system_memory_total_bytes",
    "Total physical memory in bytes",
)
SYSTEM_MEMORY_PERCENT = Gauge(
    "system_memory_usage_percent",
    "Physical memory usage percentage",
)
SYSTEM_DISK_PERCENT = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage for root partition",
)
SYSTEM_DISK_USED_BYTES = Gauge(
    "system_disk_used_bytes",
    "Disk space used in bytes",
)
SYSTEM_OPEN_FDS = Gauge(
    "system_open_file_descriptors",
    "Number of open file descriptors for this process",
)
SYSTEM_PROCESS_MEMORY_BYTES = Gauge(
    "process_memory_rss_bytes",
    "RSS memory of the Phoenix ML process",
)

# GPU gauges (only populated if pynvml available)
SYSTEM_GPU_UTILIZATION = Gauge(
    "system_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_index", "gpu_name"],
)
SYSTEM_GPU_MEMORY_USED = Gauge(
    "system_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_index", "gpu_name"],
)
SYSTEM_GPU_MEMORY_TOTAL = Gauge(
    "system_gpu_memory_total_bytes",
    "GPU total memory in bytes",
    ["gpu_index", "gpu_name"],
)
SYSTEM_GPU_TEMPERATURE = Gauge(
    "system_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_index", "gpu_name"],
)

# ── GPU Support Detection ────────────────────────────────────────

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
    _HAS_GPU = _GPU_COUNT > 0
    logger.info("🎮 GPU detected: %d device(s)", _GPU_COUNT)
except Exception:
    _HAS_GPU = False
    _GPU_COUNT = 0
    logger.info("No NVIDIA GPU detected, GPU metrics disabled")


def _collect_gpu_metrics() -> None:
    """Collect GPU utilization, memory, and temperature."""
    if not _HAS_GPU:
        return
    try:
        for i in range(_GPU_COUNT):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            SYSTEM_GPU_UTILIZATION.labels(
                gpu_index=str(i), gpu_name=name
            ).set(util.gpu)

            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            SYSTEM_GPU_MEMORY_USED.labels(
                gpu_index=str(i), gpu_name=name
            ).set(mem.used)
            SYSTEM_GPU_MEMORY_TOTAL.labels(
                gpu_index=str(i), gpu_name=name
            ).set(mem.total)

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                SYSTEM_GPU_TEMPERATURE.labels(
                    gpu_index=str(i), gpu_name=name
                ).set(temp)
            except Exception:
                pass
    except Exception as e:
        logger.debug("GPU metrics collection error: %s", e)


def _collect_system_metrics() -> None:
    """Collect CPU, memory, disk, and process metrics."""
    # CPU
    cpu_total = psutil.cpu_percent(interval=None)
    SYSTEM_CPU_PERCENT.set(cpu_total)

    per_core = psutil.cpu_percent(interval=None, percpu=True)
    for idx, pct in enumerate(per_core):
        SYSTEM_CPU_PER_CORE.labels(core=str(idx)).set(pct)

    # Memory
    mem = psutil.virtual_memory()
    SYSTEM_MEMORY_USED_BYTES.set(mem.used)
    SYSTEM_MEMORY_TOTAL_BYTES.set(mem.total)
    SYSTEM_MEMORY_PERCENT.set(mem.percent)

    # Disk
    disk = psutil.disk_usage("/")
    SYSTEM_DISK_PERCENT.set(disk.percent)
    SYSTEM_DISK_USED_BYTES.set(disk.used)

    # Process-level
    proc = psutil.Process(os.getpid())
    SYSTEM_PROCESS_MEMORY_BYTES.set(proc.memory_info().rss)
    try:
        SYSTEM_OPEN_FDS.set(proc.num_fds())
    except AttributeError:
        pass  # Windows doesn't support num_fds

    # GPU
    _collect_gpu_metrics()


class SystemMetricsCollector:
    """Background task that periodically collects system metrics.

    Usage::

        collector = SystemMetricsCollector(interval_seconds=5)
        collector.start()     # Starts background asyncio task
        ...
        collector.stop()      # Stops on shutdown
    """

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Start the background collection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "📊 System metrics collector started (interval=%.1fs, gpu=%s)",
            self._interval,
            _HAS_GPU,
        )

    def stop(self) -> None:
        """Stop the background collection loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("System metrics collector stopped")

    async def _loop(self) -> None:
        """Periodic collection loop."""
        # Warm up CPU percent (first call always returns 0)
        psutil.cpu_percent(interval=None)

        while self._running:
            try:
                _collect_system_metrics()
            except Exception as e:
                logger.warning("System metrics error: %s", e)
            await asyncio.sleep(self._interval)
