# ============================================================
# NeurIPS 实验全流程运行脚本 (3 seeds 快速验证版)
# 预计总时间: 13-20 小时 (CPU-only, 单机串行)
# ============================================================
# 使用方法:
#   1. 在 PowerShell 中 cd 到你的项目目录 (E:\electricity)
#   2. 运行: .\run_all_experiments.ps1
#   或者复制下面每个实验的命令块, 分别粘贴执行
# ============================================================

$PY = "E:\pythonn3.10\python.exe"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

Write-Host "============================================================"
Write-Host " NeurIPS Full Experiment Pipeline — 3 Seeds Quick Validation"
Write-Host " Start Time: $(Get-Date)"
Write-Host "============================================================"

# ============================================================
# 实验一: 合成数据基准测试 (5个子实验)
# 预计总时间: 4-7 小时
# ============================================================

Write-Host "`n>>> [EXP1] Synthetic Benchmark — Starting..."

# Sub 1: Prior Degradation — 核心实验, 验证先验质量对性能的影响
# 设置: d=20, T=500, 11档prior_acc × 3种corruption × 3 seeds = 99 settings × 6 methods
# 预计: 1.5-2.5 小时
Write-Host "`n  [EXP1-PRIOR] Prior degradation (est. 1.5-2.5h)..."
& $PY exp1_full_benchmark.py --sub prior --seeds 0 1 2

# Sub 2: Noise Robustness — 4种噪声分布下的鲁棒性
# 设置: d=20, T=500, 4种噪声 × 6档prior_acc × 3 seeds = 72 settings × 6 methods
# 预计: 1-1.5 小时
Write-Host "`n  [EXP1-NOISE] Noise robustness (est. 1-1.5h)..."
& $PY exp1_full_benchmark.py --sub noise --seeds 0 1 2

# Sub 3: Graph Structure — ER vs Scale-Free
# 设置: d=20, T=500, 2种图 × 6档prior_acc × 3 seeds = 36 settings × 6 methods
# 预计: 30-45 分钟
Write-Host "`n  [EXP1-GRAPH] Graph structure (est. 30-45min)..."
& $PY exp1_full_benchmark.py --sub graph --seeds 0 1 2

# Sub 4: Nonlinearity — 非线性瞬时效应下的表现
# 设置: d=[10,20], T=[500,1000], 3档prior_acc × 3 seeds = 36 settings × 6 methods
# 预计: 30-45 分钟
Write-Host "`n  [EXP1-NONLINEAR] Nonlinearity (est. 30-45min)..."
& $PY exp1_full_benchmark.py --sub nonlinear --seeds 0 1 2

# Sub 5: Scalability — 运行时间 vs 维度 (d=10到200)
# 设置: d=[10,20,50,100,200] × 3 seeds × ~3 methods
# 预计: 1-2 小时 (d=200很慢)
Write-Host "`n  [EXP1-SCALE] Scalability (est. 1-2h)..."
& $PY exp1_full_benchmark.py --sub scale --scalability --seeds 0 1 2

Write-Host "`n>>> [EXP1] Done!"


# ============================================================
# 实验二: 公开基准数据集 (Lorenz-96 + Electricity)
# 预计总时间: 1.5-2.5 小时
# ============================================================
# 注意: CausalTime, DREAM4, Netsim 需要额外下载数据
#       如果没有这些数据, 代码会跳过并提示下载方式
#       Lorenz-96 是自动生成的, 始终可用

Write-Host "`n>>> [EXP2] Real-World Benchmarks — Starting..."

# Lorenz-96 基准 (自动生成, 无需外部数据)
# 设置: d=[10,20], T=[1000,2000], 6档prior_acc × 3 seeds = 72 settings × ~5 methods
# 预计: 1-2 小时
Write-Host "`n  [EXP2-LORENZ] Lorenz-96 benchmark (est. 1-2h)..."
& $PY exp2_real_benchmarks.py --bench lorenz96 --seeds 0 1 2

# Electricity Case Study (需要你的 0227test.xlsx 和 Auto_Generated_Prior.csv)
# 预计: 15-30 分钟
Write-Host "`n  [EXP2-ELEC] Electricity case study (est. 15-30min)..."
& $PY exp2_real_benchmarks.py --bench electricity --seeds 0 1 2

Write-Host "`n>>> [EXP2] Done!"


# ============================================================
# 实验三: 下游预测 (因果特征选择 → LSTM/VAR/Transformer)
# 预计总时间: 2-4 小时
# ============================================================

Write-Host "`n>>> [EXP3] Downstream Prediction — Starting..."

# Electricity 下游预测
# 设置: 3个目标变量 × ~7种特征选择方法 × 3种预测器 × 2种top_m × 3 seeds
# 预计: 1.5-3 小时 (LSTM+Transformer训练较慢)
Write-Host "`n  [EXP3-ELEC] Electricity prediction (est. 1.5-3h)..."
& $PY exp3_downstream_prediction.py --bench electricity --seeds 0 1 2

# Lorenz-96 下游预测
# 预计: 30-60 分钟
Write-Host "`n  [EXP3-LORENZ] Lorenz-96 prediction (est. 30-60min)..."
& $PY exp3_downstream_prediction.py --bench lorenz96 --seeds 0 1 2

Write-Host "`n>>> [EXP3] Done!"


# ============================================================
# 实验四: 消融实验 (A-F 6种变体)
# 预计总时间: 2.5-4 小时
# ============================================================

Write-Host "`n>>> [EXP4] Ablation Study — Starting..."

# Synthetic ablation (图质量指标)
# 设置: d=[10,20], T=[500,1000], 4档prior_acc × 3 seeds × 6 variants = ~288 runs
# 预计: 1-1.5 小时
Write-Host "`n  [EXP4-SYNTH] Synthetic ablation (est. 1-1.5h)..."
& $PY exp4_ablation.py --sub synthetic --seeds 0 1 2

# Lorenz-96 ablation
# 预计: 30-45 分钟
Write-Host "`n  [EXP4-LORENZ] Lorenz-96 ablation (est. 30-45min)..."
& $PY exp4_ablation.py --sub lorenz --seeds 0 1 2

# Real data ablation (下游RMSE)
# 预计: 30-60 分钟
Write-Host "`n  [EXP4-REAL] Real data ablation (est. 30-60min)..."
& $PY exp4_ablation.py --sub real --seeds 0 1 2

# Soft vs Hard mask 对比 (核心ablation)
# 预计: 15-30 分钟
Write-Host "`n  [EXP4-HARDMASK] Soft vs Hard mask (est. 15-30min)..."
& $PY exp4_ablation.py --sub hard_mask --seeds 0 1 2

Write-Host "`n>>> [EXP4] Done!"


# ============================================================
# 实验五: 可扩展性 & 超参敏感性 & τ 深入分析
# 预计总时间: 3-5 小时
# ============================================================

Write-Host "`n>>> [EXP5] Scalability & Sensitivity — Starting..."

# Part A: 可扩展性 (Runtime vs Dimension)
# 设置: d=[10,20,50,100,200] × 3 seeds × ~4 methods
# 预计: 1-2 小时 (d=200 很慢)
Write-Host "`n  [EXP5-A] Scalability timing (est. 1-2h)..."
& $PY exp5_scalability_sensitivity.py --part A --seeds 0 1 2

# Part B: 超参数敏感性 (λ1 × λ2 网格搜索)
# 设置: 4×4 grid × 3 seeds = 48 runs
# 预计: 30-45 分钟
Write-Host "`n  [EXP5-B] Hyperparameter sensitivity (est. 30-45min)..."
& $PY exp5_scalability_sensitivity.py --part B --seeds 0 1 2

# Part C: Temperature τ 深入分析 (核心贡献)
# 设置: 11档prior_acc × 2种噪声 × 3 seeds + 训练轨迹 + 真实数据
# 预计: 1-1.5 小时
Write-Host "`n  [EXP5-C] Temperature tau deep-dive (est. 1-1.5h)..."
& $PY exp5_scalability_sensitivity.py --part C --seeds 0 1 2

# Part D: 收敛分析 (ALM收敛性)
# 设置: 3档prior_acc × 3 seeds = 9 runs (带完整训练日志)
# 预计: 15-20 分钟
Write-Host "`n  [EXP5-D] Convergence analysis (est. 15-20min)..."
& $PY exp5_scalability_sensitivity.py --part D --seeds 0 1 2

Write-Host "`n>>> [EXP5] Done!"


# ============================================================
# 完成
# ============================================================
Write-Host "`n============================================================"
Write-Host " ALL EXPERIMENTS COMPLETED"
Write-Host " End Time: $(Get-Date)"
Write-Host " Results are in:"
Write-Host "   exp1_prior/     exp1_noise/    exp1_graph/"
Write-Host "   exp1_nonlinear/ exp1_scale/"
Write-Host "   exp2_results/"
Write-Host "   exp3_results/"
Write-Host "   exp4_results/"
Write-Host "   exp5_results/"
Write-Host "============================================================"