# `tools/evolution_db.py` 细化设计（M1，接口 + Schema，**不含实现**）

> 配套：[full-agent-refactor-plan.md](full-agent-refactor-plan.md)（A1 方案）。
> 本文只定义**落盘结构、JSON Schema、CLI 接口签名、选择/准入算法规格**，供 review；**不写任何实现代码**。
> **算法与参数复用参考(原)仓库 LoongFlow 的做法**（`agentsdk/memory/evolution/{in_memory.py, boltzmann.py, base_memory.py}`）：
> score = evaluator 单 float（越大越好）、novelty = 代码距离 + MAP-Elites、准入 = MAP-Elites 岛内替换 + 精英存档 + 迁移、选择 = 自适应温度 Boltzmann。详见 §4。
> 解记录字段对齐原仓库 `Solution` dataclass，并保留 atrex 既有 `performance/correctness/profile_evidence` 约定。

---

## 1. 定位与边界

- **定位**：PES 进化循环的**状态中枢与唯一真源**。承载"多候选 + 多岛种群(island) + MAP-Elites 特征网格 + 精英存档 + 血缘(`parent_id`)"，是 `memory_manager.py`（线性 `v<N>.json`）装不下的那一层。
- **做什么**：候选准入(add，含 MAP-Elites/岛/精英/迁移)、父代选择(select-parents，自适应温度 Boltzmann)、每代落 checkpoint、取最优(best)、回溯血缘(lineage)、进度汇总(summary)。
- **不做什么**：不编译、不跑 kernel、不测时、不 profile（这些是 evaluator 子流程的事，结果通过 `add` 写进来）；不调 LLM；不管 git commit（沿用现有流程）。
- **可配置**：种群/岛/精英/迁移/温度/预算/收敛全部参数化，默认值=原仓库默认（§7）。

---

## 2. 落盘结构

```text
kernel_opt_<name>/
├── database/
│   ├── config.json                       # 进化配置（§3.3，默认=原仓库默认）
│   ├── solutions/                        # 全量解池（内容寻址；含评测失败的 score=0 解）
│   │   └── <solution_id>/
│   │       ├── solution.json             # 解记录（schema §3.1）
│   │       └── kernel.py                 # 该候选的代码快照
│   └── checkpoints/
│       └── iter-<K>/
│           ├── best_solution.json        # 本代最优（schema §3.1）
│           ├── metadata.json             # 本代种群/岛/精英/特征网格状态（schema §3.2）
│           └── solutions/                # 本代准入候选的 solution.json 副本
├── iteration/<K>/                        # 由编排 skill 写：planner/ executor/<M_N>/ summarizer/
├── profiles/<K>/<child>/                 # 复用现有 profile 产物布局
└── memory/v<N>.json                      # 兼容映射（memory_manager.py 继续可用；非真源）
```

- 真源 = `database/`。`solutions/<id>/` 内容寻址（`solution_id` = 短哈希），`checkpoints/iter-<K>/` 按代快照，互为索引。
- **失败/低分候选不另立"拒绝池"**（对齐原仓库）：所有候选都进 `populations`，评测失败记 `score=0`，由 MAP-Elites 网格竞争 + 超出 `population_size` 时按分数剪枝自然淘汰。负样本证据保留在 `evaluator/` 日志与 `summarizer/` 里。

---

## 3. JSON Schema

### 3.1 解记录 `solution.json` / `best_solution.json`

字段 = 原仓库 `Solution` dataclass + atrex 评测细节（放进 `metadata`，不破坏原结构）。

```jsonc
{
  // —— 对齐原仓库 Solution dataclass ——
  "solution_id": "<8-hex>",
  "parent_id": "<8-hex | null>",        // seed 为 null；可回溯到 baseline
  "generate_plan": "<planner 给该候选的策略文本>",
  "island_id": 0,                        // 所属岛
  "iteration": 0,                        // 产生该解的代号 K
  "generation": 0,                       // 进化代数（语义同 iteration，对齐原仓库双字段）
  "timestamp": 0.0,                      // epoch 秒（原仓库用 float）
  "sample_cnt": 0,                       // 作为父代被采样次数
  "sample_weight": 0.0,                  // 采样权重（Boltzmann 用，见 §4.4）
  "score": 0.0,                          // 适应度，越大越好；定义见 §4.1。失败=0
  "evaluation": "<evaluator 原始结论字符串>",
  "summary": "<summarizer 蒸馏文本>",
  "solution": "database/solutions/<id>/kernel.py",  // 代码：路径或内联（原仓库内联字符串；这里默认存路径）

  // —— atrex 扩展，放进 metadata（不影响原仓库字段语义）——
  "metadata": {
    "MAP_Elite_feature": "<bin-bin-bin>",   // MAP-Elites 特征坐标键（complexity-diversity-score）
    "lang": "CuteDSL | FlyDSL | triton | cuda",
    "code_sha256": "<hex>",
    "correctness": { "rel_err": null, "status": "PASS | FAIL | TIMEOUT_FAIL" },
    "performance": {
      "latency_us": null, "tflops": null, "bandwidth_gbps": null,
      "tflops_peak_utilization_pct": null, "bandwidth_peak_utilization_pct": null
    },
    "optimization": {
      "action_category": "<vectorized_load | swizzle | double_buffering | k_split | ...>",
      "action_description": "<what changed vs parent>"
    },
    "profile_evidence": {
      "tool_used": "<ncu | profile_kernel.sh | do_bench>",
      "evidence_summary": "<key metrics>",
      "bottleneck_type": "<compute_bound | memory_bound>",
      "evidence_chain": "<evidence -> inference -> action>"
    },
    "iteration_ref": "iteration/<K>/executor/<M_N>/"
  }
}
```

> 说明：`action_category` 不再用于 novelty（§4.3 改用代码距离），但仍保留作 planner 可读标签与人审用途。

### 3.2 代级元数据 `metadata.json`（种群/岛/精英/特征网格状态）

```jsonc
{
  "generation": 0,
  "timestamp": 0.0,
  "baseline": { "solution_id": "<8-hex>", "latency_us": null },  // score 归一化基准（来自 seed=v0）
  "best_solution_id": "<8-hex>",
  "best_score": 0.0,
  "population_size_current": 0,           // 当前 populations 中解数
  "elites": ["<solution_id>", "..."],     // 精英存档（上限 elite_archive_size）
  "islands_state": [
    {
      "island_id": 0,
      "members": ["<solution_id>", "..."],
      "best_solution_id": "<8-hex>",
      "feature_map": { "<bin-bin-bin>": "<solution_id>" }  // MAP-Elites：每个特征格当前占位解
    }
    // ... num_islands 个
  ],
  "feature_stats": {                      // 特征缩放统计（minmax），原仓库 _update_feature_stats
    "complexity": { "min": 0.0, "max": 0.0 },
    "diversity":  { "min": 0.0, "max": 0.0 },
    "score":      { "min": 0.0, "max": 0.0 }
  },
  "migration": { "last_migration_generation": 0 },
  "history": [ { "generation": 0, "best_score": 0.0, "best_solution_id": "<8-hex>" } ],
  "convergence": { "no_improve_streak": 0, "stopped": false, "stop_reason": null }
  // stop_reason: target_met | no_improve | budget_exhausted | null
}
```

### 3.3 配置 `config.json`（默认值=原仓库默认）

```jsonc
{
  // —— 种群/岛/精英/迁移：原仓库 InMemoryEvolveMemory 默认 ——
  "population_size": 100,
  "num_islands": 3,
  "elite_archive_size": 50,
  "migration_interval": 10,
  "migration_rate": 0.2,
  "feature_dimensions": ["complexity", "diversity", "score"],
  "feature_bins": null,                   // null=自动: ceil(elite_archive_size^(1/dims))，默认→4

  // —— Boltzmann 选择：原仓库 boltzmann.py 默认 ——
  "selection": {
    "strategy": "boltzmann",
    "initial_temperature": 1.0,
    "min_temperature": 0.5,
    "max_temperature": 2.0,
    "exploration_rate": 0.2,
    "use_sampling_weight": true,
    "sampling_weight_power": 1.0
  },

  // —— atrex 任务侧 ——
  "n_candidates": 3,                      // 每代 child 数（executor fan-out）
  "budget": { "max_generations": 30 },
  "convergence": { "no_improve_patience": 5, "min_rel_improve": 0.01 },
  "score_metric": "speedup_vs_baseline"
}
```

---

## 4. 算法规格（复用原仓库）

### 4.1 适应度 `score`（原仓库：单 float，越大越好）

- evaluator 返回一个 `score: float`，**越大越好**（原仓库 `EvaluationResult.score`、`_is_better` 直接比 `score >`）。
- kernel 任务把它定义为**相对 baseline 的加速比**：`score = baseline.latency_us / candidate.latency_us`（即问题 1 的"加速比"口径，与参考实现 score≈speedup 一致）。
- 硬门：评测失败 / `correctness != PASS` → `score = 0.0`（原仓库失败即 0），仍写盘进 populations，由网格竞争/剪枝淘汰。
- baseline 取 seed（`memory/v0.json` 导入）的 `latency_us`。

### 4.2 准入 `add`（原仓库：MAP-Elites 岛内替换 + 精英存档 + 迁移）

`add_solution` 流程（对齐 `in_memory.py`）：

1. **算 MAP-Elites 特征坐标**：对 `feature_dimensions=[complexity, diversity, score]` 各算 bin（§4.3），得坐标键写入 `metadata.MAP_Elite_feature`。
2. **入岛 / 网格替换**：按 `island_id` 落岛；该岛 `feature_map[坐标]` 为空则占位，否则**与占位解比 `score`**，更优则替换（MAP-Elites：每格只留最优）。
3. **更新精英存档**：维护全局 `elites`（上限 `elite_archive_size=50`），更优解入档、超额淘汰最差。
4. **更新岛最优**。
5. **迁移检查**：每 `migration_interval=10` 代，按 `migration_rate=0.2` 在岛间迁移解（环形拓扑），促进跨岛交流。
6. **种群剪枝**：`len(populations) > population_size(100)` 时按分数移除最低者。

> 不再有我先前自创的"6 步准入 + 岛容量 8"；改为原仓库的 MAP-Elites+岛+精英+迁移。

### 4.3 novelty / diversity（原仓库：代码文本距离 + MAP-Elites 特征）

- **diversity（代码距离）**：`_fast_code_diversity(c1,c2) = 0.1·|Δlen| + 10·|Δlines| + 0.5·|charset 对称差|`；对"多样性参考子集"求均值得到该解的 diversity 值。参考子集用贪心选最分散的若干解（`diversity_reference_size=20`）。
- **MAP-Elites 三特征**（每维 minmax 缩放后分箱，默认 `feature_bins=4`）：
  - `complexity` = 代码长度 `len(code)`
  - `diversity` = 上面的代码距离均值
  - `score` = 适应度
- 坐标键形如 `"2-1-3"`，决定解落在哪个网格格子（§4.2 第 2 步）。
- **不用** action_category 0/1（那是我先前的简化，现废弃）。

### 4.4 父代选择 `select-parents`（原仓库：自适应温度 Boltzmann）

`select_parents_with_dynamic_temperature`（对齐 `boltzmann.py`）：

1. **算种群 diversity** → **自适应温度**：`T` 在 `[min=0.5, max=2.0]` 间按多样性调（越散温度越高、越偏探索），与当前温度 0.8/0.2 混合稳态。
2. **exploration**：以 `exploration_rate=0.2` 概率直接随机选一个（纯探索）。
3. **精英/非精英分组采样**：从 elites 取 3 个、non-elites 取 2 个，凑 5 个候选。
4. **Boltzmann 概率**：`P(i) ∝ exp((score_i − max_score)/T)`（减 max 保数值稳定）；若 `use_sampling_weight`，再乘 `sample_weight^power` 归一。
5. 兜底：softmax / 取最高分。
6. 选 N 次得 N 个父代（被选解 `sample_cnt++`）。

---

## 5. CLI 接口签名

通用：所有命令含 `--workspace <path>`（必填，沿用 `memory_manager.py` 风格）；输出默认人读表格，加 `--json` 输出机读 JSON。退出码：`0` 成功 / `2` 用法错误 / `3` 状态错误（db 未初始化）/ `4` 未找到。

```text
python tools/evolution_db.py init        --workspace <ws> [--config <config.json>]
python tools/evolution_db.py import-seed  --workspace <ws> [--from memory/v0.json] [--code kernel.py]
python tools/evolution_db.py add          --workspace <ws> --generation <K> --parent <id|null>
                                          --code <path> --lang <CuteDSL|FlyDSL|triton|cuda>
                                          --score <f>                  # evaluator 给的适应度（失败传 0）
                                          --correctness <PASS|FAIL|TIMEOUT_FAIL> [--rel-err <f>]
                                          [--latency-us <f>] [--tflops <f>] [--bandwidth-gbps <f>]
                                          [--island <id>]              # 缺省按 round-robin 分岛
                                          [--generate-plan <str>] [--action-category <str>]
                                          [--evaluation <str>] [--evidence-file <json>]
                                          [--iteration-ref <path>] [--json]
                                          # MAP-Elites 坐标、精英、迁移、剪枝由 add 内部按 §4.2 自动处理
python tools/evolution_db.py select-parents --workspace <ws> [--n 3] [--json]
                                          # 自适应温度 Boltzmann（§4.4），温度/exploration 读 config
python tools/evolution_db.py checkpoint   --workspace <ws> --generation <K> [--json]
python tools/evolution_db.py best         --workspace <ws> [--island <id>] [--json]
python tools/evolution_db.py lineage      --workspace <ws> --solution <id> [--json]
python tools/evolution_db.py summary      --workspace <ws> [--json]
python tools/evolution_db.py list         --workspace <ws> [--generation <K>] [--island <id>] [--json]
python tools/evolution_db.py config       --workspace <ws> [--get <key>] [--set <key=value> ...]
```

### 命令语义与 I/O

| 命令 | 作用 | 关键输出 | 副作用 |
|------|------|---------|--------|
| `init` | 建 `database/`、写 `config.json`（默认=原仓库默认） | db 路径 | 创建目录/配置 |
| `import-seed` | 把 baseline(`memory/v0.json`+`kernel.py`)登记为 seed(gen0, parent=null) | `solution_id` | 写 seed + 首个 checkpoint |
| `add` | 登记一个已评测候选，跑 §4.2（MAP-Elites/岛/精英/迁移/剪枝） | `solution_id` + 落格/替换/入精英情况 | 写 `solutions/<id>/` |
| `select-parents` | 按 §4.4 自适应温度 Boltzmann 采样 N 个父代 | 父代精简记录列表 | `sample_cnt++` |
| `checkpoint` | 固化第 K 代：算 best、写 metadata、复制本代候选、判收敛 | checkpoint 路径 + best + 是否停止 | 写 `checkpoints/iter-<K>/` |
| `best` | 当前全局/指定岛最优 | best 解记录 | 无 |
| `lineage` | 回溯父链到 seed | id 链 + 每环 score/plan | 无 |
| `summary` | 逐代进度（分数曲线、种群/岛/精英规模、收敛状态） | 表格/JSON | 无 |
| `list` | 列解（可筛代/岛） | 解清单 | 无 |
| `config` | 读/改进化配置 | 配置项 | 改 `config.json` |

### 典型一代调用序列（由 `gpu-kernel-evolve` 编排）

```bash
# 一次性
python tools/evolution_db.py init        --workspace $WS
python tools/evolution_db.py import-seed  --workspace $WS               # 从 v0 建 seed

# 第 K 代
python tools/evolution_db.py select-parents --workspace $WS --n 3 --json   # → 喂 planner
# planner 出 3 策略 → executor fan-out 3 候选 → evaluator 各自出 score/correctness
python tools/evolution_db.py add --workspace $WS --generation $K --parent <pid> \
    --code <child>/kernel.py --lang triton --score 11.4 --correctness PASS \
    --latency-us 1448 --generate-plan "k_split GEMM1" --evidence-file <child>/evidence.json   # ×3
python tools/evolution_db.py checkpoint   --workspace $WS --generation $K --json
# 读 checkpoint.convergence.stopped 决定是否继续
```

---

## 6. 与 `memory_manager.py` / git 的关系

- **扩展而非替换**：复用其 JSON 读写/schema helper；`evolution_db.py` 加"种群/岛/精英/MAP-Elites/选择/血缘"层。
- **单一真源**：`database/` 为真源；`memory/v<N>.json` 退为兼容映射（可选地把每代 best 同步过去），明确标注"非真源"。
- **seed 桥接**：`import-seed` 读 `memory/v0.json` 的 `performance.latency_us` 作 baseline 归一化基准。
- **git**：commit 仍由编排流程负责；建议每代 `checkpoint` 后提交一次，message 带 `gen=K best=<id> score=<x>`。

---

## 7. 默认参数（=原仓库默认；可被核心文档覆盖）

| 参数 | 默认 | 来源 |
|------|------|------|
| `population_size` | 100 | 原仓库 |
| `num_islands` | 3 | 原仓库 |
| `elite_archive_size` | 50 | 原仓库 |
| `migration_interval` / `migration_rate` | 10 / 0.2 | 原仓库 |
| `feature_dimensions` | complexity / diversity / score | 原仓库 |
| `feature_bins` | 自动→4（`ceil(50^(1/3))`） | 原仓库 |
| `selection` 温度 | init 1.0 / min 0.5 / max 2.0 | 原仓库 boltzmann |
| `exploration_rate` | 0.2 | 原仓库 |
| `sampling_weight` / `power` | on / 1.0 | 原仓库 |
| `n_candidates` (N) | 3 | atrex 方案 |
| `budget.max_generations` | 30 | atrex 方案 |
| `convergence` | patience 5 / min_rel_improve 0.01 | atrex 方案 |
| `score_metric` | speedup_vs_baseline | atrex 方案 |

> ⚠️ 容量类参数（population 100 / elite 50）是**上限**。atrex 本地评测下 N=3×≤30 代 ≈ 最多 90 候选，种群不会填满、剪枝很少触发、迁移约 3 次——属正常，参数全可配。若要更激进可调大 N 或代数预算。

---

## 8. M1 交付物（确认后再写实现）

1. `reference/solution.schema.json`、`reference/evolve_metadata.schema.json`、`reference/evolve_config.schema.json`（§3 三份 schema 落成文件）。
2. `tools/evolution_db.py`（实现 §5 全部子命令 + §4 原仓库算法：MAP-Elites/岛/精英/迁移/剪枝 + 自适应温度 Boltzmann）。
3. 单元测试：MAP-Elites 落格与替换、代码距离 diversity、岛迁移、精英存档淘汰、自适应温度 Boltzmann 采样分布、血缘回溯、seed 导入、收敛判据触发、种群剪枝。
4. 与 `memory_manager.py` 的兼容映射（seed 导入 + 每代 best 同步）。

## 9. 设计选择确认状态

| # | 问题 | 结论 |
|---|------|------|
| 1 | score 口径 | ✅ **复用原仓库**：单 float、越大越好 = 加速比（`baseline/candidate` latency），失败=0 |
| 2 | 准入策略 | ✅ **复用原仓库**：MAP-Elites 岛内替换 + 精英存档(50) + 迁移(每10代/20%) + 超 100 剪枝（替代先前自创 6 步/岛容量 8） |
| 3 | novelty | ✅ **复用原仓库**：代码文本距离 + MAP-Elites 特征(complexity/diversity/score)（替代先前 action_category 0/1） |
| 4 | 父代选择 | ✅ **复用原仓库**：自适应温度 Boltzmann + 精英/非精英分组 + sample_weight + exploration 0.2（替代先前定温不放回采样） |
| 5 | 失败/低分候选 | ✅ **复用原仓库**：不另立拒绝池，统一进 populations 记 score=0，由网格竞争+剪枝淘汰；负样本证据留在 evaluator/summarizer |

| 6 | 容量参数 | ✅ **保留原仓库默认**：population 100 / elite 50 / num_islands 3（上限，本地小预算下不会填满，属正常） |

> §9 全部敲定。M1 实现按本文 §3（schema）/§4（原仓库算法）/§5（CLI）/§7（默认参数）落地。
