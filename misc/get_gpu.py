import subprocess
import numpy as np


def get_gpu_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    gpu_usages = [
        list(map(int, x.split(", ")))
        for x in result.stdout.decode("utf-8").split("\n")[:-1]
    ]
    return gpu_usages


def get_least_used_gpu():
    # GPUメモリ使用比率と利用率を取得
    gpu_usages = get_gpu_usage()
    print("GPU usages (memory used, memory total, utilization):", gpu_usages)

    # GPUメモリ使用比率と利用率を計算
    usage_ratios = [(x[0] / x[1], x[2] / 100) for x in gpu_usages]
    print("Usage ratios (memory, utilization):", usage_ratios)

    # GPUメモリ使用比率と利用率の合計が最小のGPU番号を取得
    least_used_gpu = np.argmin([sum(x) for x in usage_ratios])
    print("Least used GPU:", least_used_gpu)
    return least_used_gpu
