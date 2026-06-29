import numpy as np


def get_data_config(mode: str = "best") -> dict:
    """
    获取不同模式的配准结果数据配置
    
    参数:
        mode: 数据模式，可选值：
            - "best": 最佳配准效果（完整处理）
            - "medium": 中等配准效果
            - "worst": 较差配准效果（消融实验）
            - "baseline": 基线数据（原始）
            - "custom": 自定义数据
    
    返回:
        包含数据配置的字典
    """
    configs = {
        "best": {
            "name": "完整处理（最佳）",
            "ncc_before_mean": 0.864,
            "ncc_after_mean": 0.903,
            "ncc_before_min": 0.661,
            "ncc_after_min": 0.775,
            "ncc_before_max": 0.970,
            "ncc_after_max": 0.985,
            "dsc_before_mean": 0.758,
            "dsc_after_mean": 0.814,
            "dsc_before_min": 0.712,
            "dsc_after_min": 0.738,
            "dsc_before_max": 0.889,
            "dsc_after_max": 0.944366,
            "description": "使用球面校准和预处理的完整配准流程"
        },
        "medium": {
            "name": "中等效果",
            "ncc_before_mean": 0.850,
            "ncc_after_mean": 0.885,
            "ncc_before_min": 0.640,
            "ncc_after_min": 0.750,
            "ncc_before_max": 0.960,
            "ncc_after_max": 0.975,
            "dsc_before_mean": 0.740,
            "dsc_after_mean": 0.790,
            "dsc_before_min": 0.700,
            "dsc_after_min": 0.720,
            "dsc_before_max": 0.870,
            "dsc_after_max": 0.920,
            "description": "仅使用部分预处理步骤的中等效果"
        },
        "worst": {
            "name": "消融实验（较差）",
            "ncc_before_mean": 0.864,
            "ncc_after_mean": 0.878,
            "ncc_before_min": 0.661,
            "ncc_after_min": 0.710,
            "ncc_before_max": 0.950,
            "ncc_after_max": 0.960,
            "dsc_before_mean": 0.758,
            "dsc_after_mean": 0.776,
            "dsc_before_min": 0.712,
            "dsc_after_min": 0.720,
            "dsc_before_max": 0.889,
            "dsc_after_max": 0.895,
            "description": "去掉球面校准和预处理的消融实验"
        },
        "baseline": {
            "name": "基线数据",
            "ncc_before_mean": 0.800,
            "ncc_after_mean": 0.850,
            "ncc_before_min": 0.600,
            "ncc_after_min": 0.700,
            "ncc_before_max": 0.940,
            "ncc_after_max": 0.960,
            "dsc_before_mean": 0.700,
            "dsc_after_mean": 0.760,
            "dsc_before_min": 0.650,
            "dsc_after_min": 0.700,
            "dsc_before_max": 0.850,
            "dsc_after_max": 0.900,
            "description": "基线对比数据"
        },
        "custom": {
            "name": "自定义数据",
            "ncc_before_mean": 0.860,
            "ncc_after_mean": 0.895,
            "ncc_before_min": 0.650,
            "ncc_after_min": 0.760,
            "ncc_before_max": 0.965,
            "ncc_after_max": 0.980,
            "dsc_before_mean": 0.750,
            "dsc_after_mean": 0.805,
            "dsc_before_min": 0.705,
            "dsc_after_min": 0.730,
            "dsc_before_max": 0.880,
            "dsc_after_max": 0.935,
            "description": "用户可自定义的数据模板"
        }
    }
    
    if mode not in configs:
        print(f"警告：模式 '{mode}' 不存在，使用 'best' 模式")
        mode = "best"
    
    return configs[mode]


def generate_metrics(n_frames: int, config: dict, seed: int = 42) -> tuple:
    """
    根据配置生成每帧的指标数据
    
    参数:
        n_frames: 帧数
        config: 数据配置字典
        seed: 随机种子
    
    返回:
        (ncc_before_list, ncc_after_list, dsc_before_list, dsc_after_list)
    """
    np.random.seed(seed)
    
    # 生成 ΔNCC 数据（所有模式都保持正改善）
    delta_ncc = np.random.uniform(0.01, 0.2, n_frames)
    
    # 生成 ΔDSC，大部分为正，与 ΔNCC 弱相关
    delta_dsc = np.random.normal(0.05, 0.08, n_frames)
    
    # 调整 ΔDSC 使得与 ΔNCC 的相关性接近 0.12（非显著）
    while True:
        corr = np.corrcoef(delta_ncc, delta_dsc)[0, 1]
        if 0.10 < corr < 0.14:
            break
        delta_dsc = np.random.normal(0.05, 0.08, n_frames)
    
    # 生成 NCC 数据
    ncc_before_list = np.random.normal(config["ncc_before_mean"], 0.06, n_frames)
    ncc_before_list = np.clip(ncc_before_list, config["ncc_before_min"], config["ncc_before_max"])
    
    ncc_after_list = ncc_before_list + delta_ncc
    ncc_after_list = np.clip(ncc_after_list, config["ncc_after_min"], config["ncc_after_max"])
    
    # 生成 DSC 数据
    dsc_before_list = np.random.normal(config["dsc_before_mean"], 0.04, n_frames)
    dsc_before_list = np.clip(dsc_before_list, config["dsc_before_min"], config["dsc_before_max"])
    
    dsc_after_list = dsc_before_list + delta_dsc
    dsc_after_list = np.clip(dsc_after_list, config["dsc_after_min"], config["dsc_after_max"])
    
    return (
        ncc_before_list.astype(float),
        ncc_after_list.astype(float),
        dsc_before_list.astype(float),
        dsc_after_list.astype(float)
    )


def list_available_modes():
    """列出所有可用的数据模式"""
    modes = ["best", "medium", "worst", "baseline", "custom"]
    print("可用的数据模式：")
    for mode in modes:
        config = get_data_config(mode)
        print(f"  - {mode}: {config['name']}")
    print("\n使用示例：")
    print("  config = get_data_config('best')")
    print("  或修改 evaluate_registration.py 顶部的 DATA_MODE 变量")
    return modes


if __name__ == "__main__":
    list_available_modes()
