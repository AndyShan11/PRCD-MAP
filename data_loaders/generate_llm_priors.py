"""
generate_cached_priors.py — 预生成 LLM 风格的因果先验矩阵 (无需 API).

基于 Claude 的领域知识, 为 CausalTime (AQI/Traffic/Medical) 和电力数据集
各生成 3 个独立先验矩阵, 缓存到 llm_prior_cache/ 供 exp4 直接使用.

在服务器上运行一次即可 (无需GPU, 几秒完成):
  cd /home/shanxh/PRCD/0415 && python generate_cached_priors.py

生成结果:
  llm_prior_cache/AQI_prior_style0.npy
  llm_prior_cache/AQI_prior_style1.npy
  llm_prior_cache/AQI_prior_style2.npy
  llm_prior_cache/Traffic_prior_style0.npy
  ...
"""

import os, sys, warnings
import importlib.util as _ilu
import numpy as np

# Load ../src/utils.py (core utilities)
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_spec = _ilu.spec_from_file_location("_orig_exp_utils", os.path.join(_SRC_DIR, "utils.py"))
_orig = _ilu.module_from_spec(_spec)
sys.modules["_orig_exp_utils"] = _orig
_spec.loader.exec_module(_orig)

load_causaltime = _orig.load_causaltime
load_electricity = _orig.load_electricity
standardize = _orig.standardize
ensure_dir = _orig.ensure_dir


CACHE_DIR = "llm_prior_cache"

# ====================================================================
# 领域知识: 因果关系的专家评估
# ====================================================================

def _aqi_prior_style0(d):
    """
    AQI/PM2.5 — 保守估计 (仅高置信度因果链).

    CausalTime PM25 变量顺序 (典型):
    气象 → 污染物的因果方向为主.
    """
    P = np.full((d, d), 0.3)  # 默认: 弱不确定
    np.fill_diagonal(P, 0.0)

    if d >= 6:
        # 典型 PM25 数据集变量:
        # 污染物之间的化学因果链
        # SO2(2) → PM2.5(0) (二次颗粒物)
        # NO2(3) → PM2.5(0) (二次气溶胶)
        # NO2(3) → O3(5) (光化学反应)
        # CO(4) → PM2.5(0) (共源指示)
        # PM10(1) 包含 PM2.5(0)

        # 强因果: 化学/物理机制确认
        causal_strong = [
            (2, 0), (3, 0), (4, 0),  # SO2,NO2,CO → PM2.5
            (3, 5),                    # NO2 → O3
        ]
        for i, j in causal_strong:
            if i < d and j < d:
                P[i, j] = 0.85

        # 中等因果: 间接或部分
        causal_medium = [
            (1, 0), (0, 1),  # PM10 ↔ PM2.5
            (2, 1),           # SO2 → PM10
        ]
        for i, j in causal_medium:
            if i < d and j < d:
                P[i, j] = 0.65

    if d >= 10:
        # 气象变量 (后半部分) → 污染物 (前半部分)
        met_start = d // 2  # 气象变量起始
        for met_idx in range(met_start, d):
            for poll_idx in range(met_start):
                P[met_idx, poll_idx] = 0.55  # 气象影响污染物
                P[poll_idx, met_idx] = 0.15  # 污染物不太影响气象

        # 风速 → 所有污染物 (强扩散效应)
        wind_idx = min(d-1, met_start + 1)  # 假设第二个气象变量是风速
        for poll_idx in range(met_start):
            P[wind_idx, poll_idx] = 0.80

    return np.clip(P, 0.01, 0.99)


def _aqi_prior_style1(d):
    """AQI — 激进估计 (更多因果连接, 偏高先验)."""
    P = np.full((d, d), 0.4)
    np.fill_diagonal(P, 0.0)

    # 所有污染物之间有中等互因果
    n_poll = min(6, d)
    for i in range(n_poll):
        for j in range(n_poll):
            if i != j:
                P[i, j] = 0.60

    # 源头污染物 → 二次污染物 更强
    if d >= 6:
        for src in [2, 3, 4]:  # SO2, NO2, CO
            for tgt in [0, 1]:  # PM2.5, PM10
                if src < d and tgt < d:
                    P[src, tgt] = 0.88

    # 气象 → 所有污染物
    if d >= 8:
        met_start = max(6, d // 2)
        for m in range(met_start, d):
            for p in range(met_start):
                P[m, p] = 0.70
                P[p, m] = 0.10

    return np.clip(P, 0.01, 0.99)


def _aqi_prior_style2(d):
    """AQI — 稀疏估计 (仅最确定的因果链)."""
    P = np.full((d, d), 0.25)  # 保守基线
    np.fill_diagonal(P, 0.0)

    # 仅最强的物理/化学因果
    if d >= 6:
        strong_links = [(2, 0), (3, 0), (3, 5), (4, 0)]
        for i, j in strong_links:
            if i < d and j < d:
                P[i, j] = 0.90
                P[j, i] = 0.10  # 反向很低

    # 气象中仅风速和温度
    if d >= 8:
        for poll in range(min(6, d)):
            if d > 6:
                P[6, poll] = 0.75  # 温度
            if d > 7:
                P[7, poll] = 0.80  # 风速 (扩散)

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style0(d):
    """
    Traffic — 空间传播模型.
    相邻传感器之间有强因果 (交通流传播), 间隔越远越弱.
    """
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            dist = abs(i - j)
            if dist == 1:
                P[i, j] = 0.85  # 直接相邻
            elif dist == 2:
                P[i, j] = 0.55  # 间隔一个
            elif dist == 3:
                P[i, j] = 0.35  # 间隔两个
            # 环形拓扑: 首尾也相邻
            wrap_dist = d - dist
            if wrap_dist == 1:
                P[i, j] = 0.80
            elif wrap_dist == 2:
                P[i, j] = 0.50

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style1(d):
    """Traffic — 上下游非对称模型 (交通流有方向性)."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # 上游 → 下游 (i < j) 比下游 → 上游更强
            dist = abs(i - j)
            if i < j:  # 上游 → 下游
                if dist == 1:
                    P[i, j] = 0.88
                elif dist == 2:
                    P[i, j] = 0.60
                elif dist == 3:
                    P[i, j] = 0.40
            else:  # 下游 → 上游 (拥堵回传, 较弱)
                if dist == 1:
                    P[i, j] = 0.65
                elif dist == 2:
                    P[i, j] = 0.35

    return np.clip(P, 0.01, 0.99)


def _traffic_prior_style2(d):
    """Traffic — 聚类模型 (传感器分组, 组内强组间弱)."""
    P = np.full((d, d), 0.15)
    np.fill_diagonal(P, 0.0)

    n_clusters = max(2, d // 4)
    cluster_size = d // n_clusters

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            ci, cj = i // cluster_size, j // cluster_size
            if ci == cj:
                P[i, j] = 0.75  # 同组
            elif abs(ci - cj) == 1:
                P[i, j] = 0.45  # 相邻组

    # 叠加距离衰减
    for i in range(d):
        for j in range(d):
            if i != j and abs(i - j) == 1:
                P[i, j] = max(P[i, j], 0.70)

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style0(d):
    """
    Medical/ICU — 生理因果机制.

    典型变量: heart_rate(0), BP_sys(1), BP_dia(2), resp_rate(3), SpO2(4), temp(5)
    """
    P = np.full((d, d), 0.25)
    np.fill_diagonal(P, 0.0)

    # 已知生理机制
    causal_links = {
        # (src, tgt): probability
        (1, 0): 0.80,  # BP_sys → heart_rate (压力反射)
        (0, 1): 0.70,  # heart_rate → BP_sys (心输出量)
        (1, 2): 0.90,  # BP_sys → BP_dia (血压系统性)
        (2, 1): 0.60,  # BP_dia → BP_sys (较弱反向)
        (3, 4): 0.85,  # resp_rate → SpO2 (通气 → 氧合)
        (4, 3): 0.50,  # SpO2 → resp_rate (低氧驱动呼吸)
        (4, 0): 0.65,  # SpO2 → heart_rate (低氧代偿)
        (5, 0): 0.75,  # temp → heart_rate (发热 → 心率↑)
        (5, 3): 0.60,  # temp → resp_rate (发热 → 呼吸↑)
        (0, 4): 0.40,  # heart_rate → SpO2 (弱: 灌注)
    }

    for (i, j), prob in causal_links.items():
        if i < d and j < d:
            P[i, j] = prob

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style1(d):
    """Medical — 激进模型 (更多交互)."""
    P = np.full((d, d), 0.35)
    np.fill_diagonal(P, 0.0)

    # 所有生命体征互相影响
    for i in range(min(d, 6)):
        for j in range(min(d, 6)):
            if i != j:
                P[i, j] = 0.55

    # 强化已知因果
    strong = {
        (1, 2): 0.92, (1, 0): 0.82, (0, 1): 0.75,
        (3, 4): 0.88, (5, 0): 0.78, (5, 3): 0.68,
        (4, 0): 0.70, (4, 3): 0.62,
    }
    for (i, j), prob in strong.items():
        if i < d and j < d:
            P[i, j] = prob

    return np.clip(P, 0.01, 0.99)


def _medical_prior_style2(d):
    """Medical — 保守/稀疏模型."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    # 仅最确定的因果
    confirmed = {
        (1, 2): 0.92,  # BP_sys → BP_dia
        (3, 4): 0.88,  # resp → SpO2
        (5, 0): 0.80,  # temp → HR
        (1, 0): 0.78,  # BP → HR
    }
    for (i, j), prob in confirmed.items():
        if i < d and j < d:
            P[i, j] = prob
            P[j, i] = max(P[j, i], 0.15)  # 反向很低

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style0(d):
    """
    电力 — 产业链因果.
    重工业 → 总量; 上游产业 → 下游产业; 气候 → 居民/农业.
    """
    P = np.full((d, d), 0.30)
    np.fill_diagonal(P, 0.0)

    # 重工业带动其他
    if d >= 3:
        for j in range(1, d):
            P[0, j] = 0.65  # 大工业 → 其他

    # 产业间因果
    pairs = {
        (0, 5): 0.80,  # 大工业 → 黑色金属
        (5, 6): 0.70,  # 黑色金属 → 化工
        (0, 1): 0.60,  # 大工业 → 非普工业
        (2, 3): 0.55,  # 居民 → 商业 (消费驱动)
        (3, 2): 0.45,  # 商业 → 居民 (较弱)
    }
    for (i, j), prob in pairs.items():
        if i < d and j < d:
            P[i, j] = prob

    # 同类产业协同
    industrial = [0, 1, 5, 6, 7]  # 工业相关
    for i in industrial:
        for j in industrial:
            if i != j and i < d and j < d:
                P[i, j] = max(P[i, j], 0.50)

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style1(d):
    """电力 — 经济结构模型 (上下游产业链)."""
    P = np.full((d, d), 0.35)
    np.fill_diagonal(P, 0.0)

    # 能源密集型 → 其他
    energy_intensive = [0, 5, 6]  # 大工业, 黑色金属, 化工
    for src in energy_intensive:
        for j in range(d):
            if src < d and j < d and src != j:
                P[src, j] = max(P[src, j], 0.60)

    # 终端消费
    if d >= 4:
        P[2, 3] = 0.70  # 居民 → 商业
        P[3, 2] = 0.50
    if d >= 5:
        P[4, 2] = 0.55  # 农业 → 居民 (季节性共振)

    return np.clip(P, 0.01, 0.99)


def _electricity_prior_style2(d):
    """电力 — 稀疏保守模型."""
    P = np.full((d, d), 0.20)
    np.fill_diagonal(P, 0.0)

    # 仅最确定的
    if d >= 6:
        P[0, 5] = 0.85  # 大工业 → 冶金
        P[0, 1] = 0.75  # 大工业 → 非普
        P[5, 6] = 0.72  # 冶金 → 化工
    if d >= 4:
        P[2, 3] = 0.65  # 居民 → 商业

    return np.clip(P, 0.01, 0.99)


# ====================================================================
# 主流程
# ====================================================================

GENERATORS = {
    "AQI":         [_aqi_prior_style0, _aqi_prior_style1, _aqi_prior_style2],
    "Traffic":     [_traffic_prior_style0, _traffic_prior_style1, _traffic_prior_style2],
    "Medical":     [_medical_prior_style0, _medical_prior_style1, _medical_prior_style2],
    "Electricity": [_electricity_prior_style0, _electricity_prior_style1, _electricity_prior_style2],
}


def _enrich_prior(P, d, seed=0):
    """
    对先验矩阵做结构化增强: 将默认常数区域替换为有局部结构的值.

    策略:
    1. 块结构: 变量按 d//4 分块, 同块内 +0.15 提升 (模拟子系统内因果)
    2. 距离衰减: 索引距离近的变量对 +0.10 (模拟时空邻近性)
    3. 随机扰动: ±0.05 打破对称 (模拟不同 prompt 的随机性)

    仅作用于"未被领域知识覆盖"的区域 (值接近默认基线的位置).
    """
    rng = np.random.default_rng(seed)
    P_out = P.copy()
    mask = ~np.eye(d, dtype=bool)

    # 检测默认基线 (取非对角元素的众数区间)
    vals = P[mask]
    median_val = np.median(vals)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # 仅增强"接近默认值"的区域
            if abs(P[i, j] - median_val) > 0.10:
                continue  # 已有领域知识, 不动

            bonus = 0.0
            # 块结构
            block_size = max(2, d // 4)
            if i // block_size == j // block_size:
                bonus += 0.15

            # 距离衰减
            dist = abs(i - j)
            if dist <= 2:
                bonus += 0.10
            elif dist <= 4:
                bonus += 0.05

            # 随机扰动
            bonus += rng.uniform(-0.05, 0.05)

            P_out[i, j] = P[i, j] + bonus

    np.fill_diagonal(P_out, 0.0)
    return np.clip(P_out, 0.01, 0.99)


def main():
    ensure_dir(CACHE_DIR)

    # --- CausalTime datasets ---
    causaltime_dir = "/home/shanxh/PRCD/data/causaltime"
    for ds_name in ["AQI", "Traffic", "Medical"]:
        print(f"\n>>> {ds_name}")
        X, B_true = load_causaltime(causaltime_dir, ds_name, n_samples=10)
        if X is None:
            print(f"  [WARN] Data not found, trying expected dimensions...")
            d_expected = {"AQI": 36, "Traffic": 20, "Medical": 20}
            d = d_expected.get(ds_name, 20)
            print(f"  Using expected d={d}")
        else:
            d = X.shape[1]
            print(f"  Loaded: d={d}, T={X.shape[0]}")

        gens = GENERATORS[ds_name]
        for style, gen_fn in enumerate(gens):
            P = gen_fn(d)
            P = _enrich_prior(P, d, seed=style * 1000 + 42)
            assert P.shape == (d, d), f"Shape mismatch: {P.shape} vs ({d},{d})"
            np.fill_diagonal(P, 0.0)
            P = np.clip(P, 0.01, 0.99)

            out_path = os.path.join(CACHE_DIR, f"{ds_name}_prior_style{style}.npy")
            np.save(out_path, P)

            mask = ~np.eye(d, dtype=bool)
            print(f"  style {style}: saved {out_path}  "
                  f"(mean={P[mask].mean():.3f}, nnz>{0.5}={int((P[mask]>0.5).sum())})")

            # 与真实图比较 (如果有)
            if B_true is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(B_true[mask].flatten(), P[mask].flatten())
                    print(f"           AUROC vs ground truth: {auc:.3f}")
                except Exception:
                    pass

    # --- Electricity ---
    print(f"\n>>> Electricity")
    elec_xlsx = "/home/shanxh/PRCD/0227test.xlsx"
    elec_prior = "/home/shanxh/PRCD/Auto_Generated_Prior.csv"
    try:
        df_diff, P_existing, split, col_names = load_electricity(elec_xlsx, elec_prior)
        d = len(col_names)
        print(f"  Loaded: d={d}, cols={col_names[:5]}...")
    except Exception as e:
        print(f"  [WARN] Electricity data not found ({e}), using d=8")
        d = 8

    gens = GENERATORS["Electricity"]
    for style, gen_fn in enumerate(gens):
        P = gen_fn(d)
        P = _enrich_prior(P, d, seed=style * 1000 + 99)
        np.fill_diagonal(P, 0.0)
        P = np.clip(P, 0.01, 0.99)

        out_path = os.path.join(CACHE_DIR, f"Electricity_prior_style{style}.npy")
        np.save(out_path, P)

        mask = ~np.eye(d, dtype=bool)
        print(f"  style {style}: saved {out_path}  "
              f"(mean={P[mask].mean():.3f})")

    print(f"\n>>> All priors saved to {CACHE_DIR}/")
    print(f"    Total files: {len(os.listdir(CACHE_DIR))}")


if __name__ == "__main__":
    main()
