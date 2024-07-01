"""IMPORTANT: This module only works for NVIDIA gpus with nvidia-smi installed."""

import subprocess


def query_gpu_info(name: str, units: bool = False) -> str:
    """Queries information from the gpu, like the gpu utilization, gpu memory usage,...

    name: name of the gpu information to query. Please run 'nvidia-smi --help-query-gpu' in the terminal for more info
    units (optional): True if the output should contain the unit of the data, False if no

    Returns a string containing the required information."""
    return subprocess.run([
        "nvidia-smi", f"--query-gpu={name}",
        f"--format=csv,noheader{',nounits' if not units else ''}"
    ],
                          capture_output=True,
                          text=True).stdout.strip()


def query_gpu_total_mem(units: bool = True) -> str:
    """Queries the total memory available on the GPU.

    units (optional): True if the output should contain the unit, False if not

    Returns the total memory, measured in MB, as a string."""
    return query_gpu_info('memory.total', units)


def query_gpu_used_mem(units: bool = True) -> str:
    """Queries the memory used on the GPU.

    units (optional): True if the output should contain the unit, False if not

    Returns the used memory, measured in MB, as a string."""

    return query_gpu_info('memory.used', units)


def query_gpu_utilization() -> str:
    """Queries the utilization of the GPU.

    units (optional): True if the output should contain the unit, False if not

    Returns the utilization of the GPU, measured in %, as a string."""

    return query_gpu_info('utilization.gpu')
