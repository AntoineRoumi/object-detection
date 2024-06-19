import subprocess

def query_gpu_info(name: str, units: bool = False) -> str:
    return subprocess.run(["nvidia-smi",f"--query-gpu={name}",f"--format=csv,noheader{',nounits' if not units else ''}"], capture_output=True, text=True).stdout.strip()

def query_gpu_total_mem(units: bool = True) -> str:
    return query_gpu_info('memory.total', units)

def query_gpu_used_mem(units: bool = True) ->  str:
    return query_gpu_info('memory.used', units)

def query_gpu_utilization() -> str:
    return query_gpu_info('utilization.gpu')
