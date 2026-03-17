#!/bin/bash
# ============================================================
# NeurIPS 实验全流程运行脚本 (3 seeds 快速验证版) — Linux版
# 预计总时间: 13-20 小时 (单卡串行)
# ============================================================
# 使用方法:
#   1. 将此脚本放到你的项目目录下 (即各 exp*.py 所在目录)
#      例如: /home/shanxh/electricity/electricity/
#   2. 赋予执行权限:
#        chmod +x run_all_experiments.sh
#   3. 指定GPU并运行 (推荐用 nohup 防止断线):
#        CUDA_VISIBLE_DEVICES=2 nohup bash run_all_experiments.sh > run_all.log 2>&1 &
#      或直接前台运行:
#        CUDA_VISIBLE_DEVICES=2 bash run_all_experiments.sh
# ============================================================

# ---------- 基础配置 ----------
# 如果你用 conda 环境, 取消注释下一行并改成你的环境名:
# source activate your_env_name

export PYTHONWARNINGS="ignore::FutureWarning"

# 脚本所在目录即为工作目录 (确保 exp*.py 都在这里)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="python"   # 如需指定完整路径, 改为: PY="/path/to/python"

# ---------- 工具函数 ----------
log() {
    echo ""
    echo "============================================================"
    echo " $*"
    echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
}

run() {
    # 运行命令; 如果失败则打印错误但继续执行后续实验
    echo ""
    echo ">>> 运行: $*"
    "$@"
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "!!! 警告: 命令退出码 $ret — $*" >&2
    fi
    return $ret
}

# ============================================================
log "NeurIPS Full Experiment Pipeline — 3 Seeds Quick Validation"
# ============================================================


# ============================================================
# 实验一: 合成数据基准测试 (5个子实验)
# 预计总时间: 4-7 小时
# ============================================================

echo ""
echo ">>> [EXP1] Synthetic Benchmark — Starting..."

# Sub 1: Prior Degradation — 核心实验, 验证先验质量对性能的影响
# 设置: d=20, T=500, 11档prior_acc × 3种corruption × 3 seeds = 99 settings × 6 methods
# 预计: 1.5-2.5 小时
echo ""
echo "  [EXP1-PRIOR] Prior degradation (est. 1.5-2.5h)..."
run $PY exp1_full_benchmark.py --sub prior --seeds 0 1 2

# Sub 2: Noise Robustness — 4种噪声分布下的鲁棒性
# 设置: d=20, T=500, 4种噪声 × 6档prior_acc × 3 seeds = 72 settings × 6 methods
# 预计: 1-1.5 小时
echo ""
echo "  [EXP1-NOISE] Noise robustness (est. 1-1.5h)..."
run $PY exp1_full_benchmark.py --sub noise --seeds 0 1 2

# Sub 3: Graph Structure — ER vs Scale-Free
# 设置: d=20, T=500, 2种图 × 6档prior_acc × 3 seeds = 36 settings × 6 methods
# 预计: 30-45 分钟
echo ""
echo "  [EXP1-GRAPH] Graph structure (est. 30-45min)..."
run $PY exp1_full_benchmark.py --sub graph --seeds 0 1 2

# Sub 4: Nonlinearity — 非线性瞬时效应下的表现
# 设置: d=[10,20], T=[500,1000], 3档prior_acc × 3 seeds = 36 settings × 6 methods
# 预计: 30-45 分钟
echo ""
echo "  [EXP1-NONLINEAR] Nonlinearity (est. 30-45min)..."
run $PY exp1_full_benchmark.py --sub nonlinear --seeds 0 1 2

# Sub 5: Scalability — 运行时间 vs 维度 (d=10到200)
# 设置: d=[10,20,50,100,200] × 3 seeds × ~3 methods
# 预计: 1-2 小时 (d=200很慢)
echo ""
echo "  [EXP1-SCALE] Scalability (est. 1-2h)..."
run $PY exp1_full_benchmark.py --sub scale --scalability --seeds 0 1 2

echo ""
echo ">>> [EXP1] Done!"


# ============================================================
# 实验二: 公开基准数据集 (Lorenz-96 + Electricity)
# 预计总时间: 1.5-2.5 小时
# ============================================================

echo ""
echo ">>> [EXP2] Real-World Benchmarks — Starting..."

# Lorenz-96 基准 (自动生成, 无需外部数据)
echo ""
echo "  [EXP2-LORENZ] Lorenz-96 benchmark (est. 1-2h)..."
run $PY exp2_real_benchmarks.py --bench lorenz96 --seeds 0 1 2

# Electricity Case Study (需要 0227test.xlsx 和 Auto_Generated_Prior.csv)
echo ""
echo "  [EXP2-ELEC] Electricity case study (est. 15-30min)..."
run $PY exp2_real_benchmarks.py --bench electricity --seeds 0 1 2

echo ""
echo ">>> [EXP2] Done!"


# ============================================================
# 实验三: 下游预测 (因果特征选择 → LSTM/VAR/Transformer)
# 预计总时间: 2-4 小时
# ============================================================

echo ""
echo ">>> [EXP3] Downstream Prediction — Starting..."

echo ""
echo "  [EXP3-ELEC] Electricity prediction (est. 1.5-3h)..."
run $PY exp3_downstream_prediction.py --bench electricity --seeds 0 1 2

echo ""
echo "  [EXP3-LORENZ] Lorenz-96 prediction (est. 30-60min)..."
run $PY exp3_downstream_prediction.py --bench lorenz96 --seeds 0 1 2

echo ""
echo ">>> [EXP3] Done!"


# ============================================================
# 实验四: 消融实验 (A-F 6种变体)
# 预计总时间: 2.5-4 小时
# ============================================================

echo ""
echo ">>> [EXP4] Ablation Study — Starting..."

echo ""
echo "  [EXP4-SYNTH] Synthetic ablation (est. 1-1.5h)..."
run $PY exp4_ablation.py --sub synthetic --seeds 0 1 2

echo ""
echo "  [EXP4-LORENZ] Lorenz-96 ablation (est. 30-45min)..."
run $PY exp4_ablation.py --sub lorenz --seeds 0 1 2

echo ""
echo "  [EXP4-REAL] Real data ablation (est. 30-60min)..."
run $PY exp4_ablation.py --sub real --seeds 0 1 2

echo ""
echo "  [EXP4-HARDMASK] Soft vs Hard mask (est. 15-30min)..."
run $PY exp4_ablation.py --sub hard_mask --seeds 0 1 2

echo ""
echo ">>> [EXP4] Done!"


# ============================================================
# 实验五: 可扩展性 & 超参敏感性 & τ 深入分析
# 预计总时间: 3-5 小时
# ============================================================

echo ""
echo ">>> [EXP5] Scalability & Sensitivity — Starting..."

echo ""
echo "  [EXP5-A] Scalability timing (est. 1-2h)..."
run $PY exp5_scalability_sensitivity.py --part A --seeds 0 1 2

echo ""
echo "  [EXP5-B] Hyperparameter sensitivity (est. 30-45min)..."
run $PY exp5_scalability_sensitivity.py --part B --seeds 0 1 2

echo ""
echo "  [EXP5-C] Temperature tau deep-dive (est. 1-1.5h)..."
run $PY exp5_scalability_sensitivity.py --part C --seeds 0 1 2

echo ""
echo "  [EXP5-D] Convergence analysis (est. 15-20min)..."
run $PY exp5_scalability_sensitivity.py --part D --seeds 0 1 2

echo ""
echo ">>> [EXP5] Done!"


# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETED"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 结果目录:"
echo "   exp1_prior/     exp1_noise/    exp1_graph/"
echo "   exp1_nonlinear/ exp1_scale/"
echo "   exp2_results/"
echo "   exp3_results/"
echo "   exp4_results/"
echo "   exp5_results/"
echo "============================================================"