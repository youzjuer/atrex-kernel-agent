# Atrex Kernel Agent · Full-Agent 进化式改造方案（草案）

> 状态：**定稿（默认参数版）**。决策 A/B/C 已全部拍板，剩余开放项已按合理默认值锁定（见 §4.4 默认参数表）。
> 本方案基于两份来源：(1) 参考实现 `syhya/mlsys26-flashinfer-contest` 的 **full-agent / LoongFlow PES** 方法；(2) 当前 atrex 仓库 `dev_ycb` 分支的现有工作流。
> ⚠️ 钉钉核心文档（`alidocs.dingtalk.com/i/nodes/YMyQA2dXW7gYo6MzcZQMKELAWzlwrZgb`）暂未读到（登录态限制）。本稿采用的默认值若与该文档冲突，**以文档为准并据此修订**；§4.4 标注了哪些值属于"可被文档覆盖"。

---

## 1. 改造目标（一句话）

把 atrex 当前**单轨迹、profile 驱动的线性优化循环**，升级为**进化式 Plan–Execute–Summary（PES）+ 种群（island-model）**的搜索循环：每轮并行产出多个候选 kernel，由统一的 evaluator 作为唯一晋级门，按分数/新颖度维护种群并做带血缘（`parent_id`）的父代选择，从而把"一条路走到黑"变成"多路并行进化、择优汇聚"。

**保留** atrex 的核心资产不变：`gpu-wiki` 硬件规格溯源、Roofline 分析、`ncu`/`rocprofv3` profile 证据链、工具集 `tools/`。这些恰好成为新循环里 planner 的输入证据与 evaluator 的打分信号。

---

## 2. 现状 vs 目标（差异分析）

| 维度 | 现状（atrex 单轨迹） | 目标（full-agent / PES 进化） |
|---|---|---|
| 循环形态 | 线性：profile→证据→查知识→单类别改动→gate→commit→重复 | 进化：Plan→Execute(多候选)→Evaluate→Summary→种群更新→下一代 |
| 每轮候选数 | 1（强制"只改一个优化类别"） | N（executor 并行 fan-out 出多个 child） |
| 选择机制 | git 线性 commit / 失败 revert | island-model 种群 + 按分数&新颖度准入 + Boltzmann 父代选择 |
| 记忆载体 | `memory/v<N>.json` + `masked` 标志 | `database/checkpoints/.../{best_solution.json, metadata.json, solutions/}` |
| 血缘 | 线性 v0→v1→…（无显式父子） | DAG，每个解带 `parent_id`，可回溯到 seed |
| 停滞处理 | `partial-restart`：mask 一半记忆后重启 | 种群多样性 + 岛屿迁移/重置（partial-restart 可保留为兜底） |
| 知识来源 | gpu-wiki → reference → web | **不变**，作为 planner 上下文证据 |
| profile 证据 | 决策前置硬约束 | **不变**，既喂 planner 又作为 evaluator/summary 信号 |
| 晋级门 | 本地 `test_kernel.py` + `do_bench` | evaluator 返回 `(correctness, score/latency)`，是唯一晋级门 |
| 停止条件 | README 的 Stop Conditions（peak×90%） | **不变** + 迭代预算 / 收敛判据 |
| 全量留痕 | 每次 commit + per-version 文件 | per-iteration 完整 trace（planner/executor/summarizer/evaluator I/O） |

**核心 gap**：atrex 缺三样东西 —— ①一个能装"多候选+种群+血缘"的 **进化数据库**；②executor 的 **多候选 fan-out** 能力；③planner 从"读最新一版记忆"升级为"读整个打分种群+历史 summary 出策略"。

---

## 3. 目标架构

### 3.1 PES 进化循环（每一代 K）

```text
            ┌──────────────────────────────────────────────────────────┐
            │                  Evolutionary Database                    │
            │   islands × solutions（score, correctness, parent_id）     │
            └───────────▲───────────────────────────────┬──────────────┘
                        │ 父代选择(Boltzmann)            │ 准入(score & novelty)
                        │                                │
   ┌────────────┐  plan │   ┌────────────┐  N candidates ┌──────────────┐
   │  PLANNER   │───────┼──▶│  EXECUTOR  │──────────────▶│  EVALUATOR   │
   │ 读种群+证据 │       │   │ fan-out N  │  compile/run  │ 唯一晋级门    │
   │ +历史summary│       │   │ 每个child= │  + profile    │ (corr,score) │
   └────────────┘       │   │ 一个候选   │               └──────┬───────┘
         ▲              │   └────────────┘                      │
         │ 喂回下一代    │                                       │ 结果回灌
   ┌─────┴──────┐       │                                       │
   │  SUMMARY   │◀──────┴───────────────────────────────────────┘
   │ 蒸馏成败经验 │
   └────────────┘
```

- **Planner**：消费"当前种群（带分数/岛状态）+ 上轮 summary + profile 证据 + gpu-wiki/reference 知识"，产出**策略级**改进计划（例："把 FP8 dequant 融进 GEMM"、"GEMM1 做 K-split"），而非 token 级 diff。可一次产出多条策略分支。
- **Executor**：把计划落成代码，**fan-out N 个 child**，每个 child 独立产出一个候选 kernel + 完整 history。child 间互不可见，保证多样性。
- **Evaluator**：对每个候选编译+跑 `test_kernel.py` 正确性 +（`do_bench`/profile）测时，返回 `(correctness, score, latency)`。**唯一晋级门**；失败候选不丢弃，转为负样本证据。
- **Summary**：蒸馏本代"什么有效/什么失败"，喂回下一代 planner 上下文。
- **Database**：island-model 种群；按分数&新颖度准入；父代选择从岛中采样；每代落 checkpoint。

### 3.2 与 atrex 现有阶段的衔接

```text
解析输入 → Step0(gpu-wiki硬件规格 + Roofline + Stop Conditions)   ← 不变
        → Stage1 baseline(产出 v0 → 作为进化 seed)               ← 基本不变
        → Stage2: PES 进化循环（替换原线性循环）                   ← 本次改造核心
        → 收敛/达标 → output-contract 打包 generated_kernel.py    ← 不变
```

---

## 4. 关键决策点（需你拍板，文档可覆盖）

### 决策 A：目标形态 —— "融入" vs "移植" ✅ 已确认 = A1

> **结论（已拍板）：采用 A1 —— 在 atrex 内融入进化循环，不整体移植 LoongFlow 框架。**

| 方案 | 做法 | 优点 | 代价 | 结论 |
|---|---|---|---|---|
| **A1 在 atrex 内融入进化循环** | 沿用 atrex 的 SKILL.md + `agents/*.md` + `tools/*.py` 体例，新增一个 PES 编排 skill + 一个进化数据库工具，把"单候选"扩成"多候选+种群" | 改动可控、与现有 profile/gpu-wiki 体系无缝、符合 atrex 既有"skill/agent/tool"风格 | 需要自己实现轻量种群/选择逻辑 | ✅ **采用** |
| **A2 移植 LoongFlow Python 框架** | 把参考仓库 `src/loongflow/framework/pes/*` 引入 atrex，适配其 task/eval | 直接拿到成熟的 planner/executor/evaluator/database/Boltzmann 选择 | 引入大体量 Python 依赖与运行时，和 atrex 的 markdown-skill 体例割裂；评测器需对接 atrex 本地 profile 而非 Modal | ❌ 不采用 |

理由：atrex 本质是"由 Skill/Agent 文档驱动 + 一组 Python 工具"的系统，进化循环的"种群/选择/血缘"用一个新的 `tools/evolution_db.py`（基于现有 `memory_manager.py` 扩展）即可承载，无需引入整套框架运行时。后续所有组件设计（§5）与里程碑（§6）均按 A1 推进。

### 决策 B：evaluator 跑在哪 ✅ 已定 = 本地

- 参考实现用 **Modal 远程 B200**；atrex 现状是**本地 GPU + `test_kernel.py` / `do_bench` / `ncu`**。
- **结论：沿用 atrex 本地评测**，evaluator 直接复用现有正确性测试 + `do_bench` 测时 + `ncu`/`rocprofv3` profile。远程/批量并行评测**不在本期范围**（接口预留：evaluator 抽象成可替换后端，未来要接 Modal 只换实现，不动编排）。

### 决策 C：种群与并行规模 ✅ 已定（保守默认，可配置）

- **每代候选数 N = 3**（executor fan-out 3 个 child）。
- **种群/岛/精英/选择 = 复用原仓库 LoongFlow 默认**（`num_islands=3`、`population_size=100`、`elite_archive_size=50`、迁移每 10 代/20%、MAP-Elites 特征网格、自适应温度 Boltzmann）。详见 [evolution-db-design.md](evolution-db-design.md) §4/§7。
- **迭代预算 = 最多 30 代**（参考实现 MoE 用了 40 代；本地评测更慢，先设 30 上限，可配置）。
- **收敛/停止判据**（满足任一即停）：① README `Stop Conditions` 达标；② 连续 **5** 代 best 无提升（相对提升 < 1%）；③ 迭代预算耗尽。
- 容量类参数（population 100 / elite 50）是上限；本地 N=3×≤30 代下种群不会填满，属正常，全部可配置。

### 4.4 默认参数表（本稿锁定值）

| 参数 | 默认值 | 可被核心文档覆盖 | 落点 |
|---|---|---|---|
| 目标形态 | A1 融入（不移植 LoongFlow） | 否（已确认） | §5 / §6 |
| evaluator 运行环境 | 本地（do_bench + ncu/rocprofv3） | 是 | §5.3 evaluator |
| 每代候选数 N | 3 | 是 | `gpu-kernel-evolve` |
| 岛数量 | **3（原仓库默认）** | 是 | `evolution_db` |
| 迭代预算 | ≤30 代 | 是 | workspace README `Evolve Config` |
| 收敛判据 | 达标 / 连续5代无提升(<1%) / 预算耗尽 | 是 | `gpu-kernel-evolve` |
| score 口径 | **加速比(原仓库:单float越大越好)** | 是 | evaluator |
| novelty / 准入 | **代码距离 + MAP-Elites + 精英存档 + 迁移(原仓库)** | 是 | `evolution_db add` |
| 父代选择 | **自适应温度 Boltzmann(原仓库)** | 是 | `evolution_db select-parents` |
| 复用 LoongFlow 代码 | 否（仅借鉴概念） | 否（A1 已确认） | 全局 |
| partial-restart | 保留，退化为停滞兜底/岛重置 | 是 | `agents/gpu-kernel-partial-restart.md` |
| 首验证算子 / 平台 | moe 方向 / **按实际可用 GPU 硬件选择**（运行时由 atrex `platform` 输入确定，规格从 gpu-wiki 取；atrex 支持 H20/H100/H200/MI300X/MI308X/MI355X 等） | 是 | M6 验证 |
| trace/checkpoint 格式 | 对齐参考 schema（database/iteration/evaluator） | 是 | §5.1 |

---

## 5. 组件级设计（A1 方案）

### 5.1 进化数据库 `tools/evolution_db.py`（新增，扩展自 `memory_manager.py`）
承载种群与血缘。落盘结构对齐参考实现的 trace schema：

```text
kernel_opt_<name>/
├── database/
│   └── checkpoints/iter-<K>/
│       ├── best_solution.json     # 本代最优：{solution_id, code/path, score, correctness, parent_id, evidence}
│       ├── metadata.json          # 种群/岛状态：每个解的 score、island、admit 时间
│       └── solutions/             # 本代准入的全部候选
├── iteration/<K>/
│   ├── planner/                   # 计划 prompt + LLM 输出
│   ├── executor/<M_N>/            # 每个 child 的代码 + history
│   └── summarizer/                # 蒸馏反馈
└── profiles/<K>/<child>/          # 复用现有 profile 产物布局
```

接口（CLI，沿用 memory_manager 风格）：
- `admit`：按分数&新颖度把候选并入种群，写 `parent_id`。
- `select-parents`：从岛中采样父代（Boltzmann/温度，对齐参考的 `agentsdk/memory/evolution/boltzmann.py` 思路）。
- `checkpoint`：落一代 checkpoint（best_solution + metadata + solutions）。
- `best` / `lineage` / `summary`：取最优、回溯血缘、打印进度表。
- 兼容旧 `memory/v<N>.json`（v0 作为 seed 导入）。

### 5.2 PES 编排 skill `skills/gpu-kernel-evolve/SKILL.md`（新增，替换 Stage 2 的线性循环）
负责一代的编排：select-parents → 调 planner → 调 executor(fan-out N) → 调 evaluator → admit → summary → checkpoint → 停止判据。
复用 `gpu-kernel-bottleneck-analysis` 做证据提取、`gpu-wiki`/reference 做知识检索。

### 5.3 子 agent（新增/改造 `agents/*.md`）
- `agents/gpu-kernel-planner.md`（新）：读种群+证据+summary → 出 1..N 条策略计划。
- `agents/gpu-kernel-executor.md`（新）：按单条策略实现一个候选 kernel（一个 child = 一个 subagent 实例，主 agent 并行 fan-out）。
- `agents/gpu-kernel-evaluator.md`（新或由 skill 直接编排）：编译/正确性/测时/profile → 回 `(correctness, score, latency, evidence)`。
- `agents/gpu-kernel-summarizer.md`（新）：蒸馏本代经验。
- `agents/gpu-kernel-baseline.md`（改）：产物 v0 同时登记为进化 seed。
- `agents/gpu-kernel-partial-restart.md`（保留）：退化为"种群停滞兜底/岛重置"。

### 5.4 顶层路由 `SKILL.md`（改）
Stage 2 由"profile-optimizer 线性循环"改为"gpu-kernel-evolve 进化循环"；Step0 / 硬件规格溯源 / Stop Conditions / output-contract 不变。

### 5.5 模板与 schema（`reference/`）
- 新增 `reference/solution.schema.json`（候选/best_solution 结构，含 `parent_id`、`score`、`correctness`、`evidence`）。
- 新增 `reference/planner_prompt.md` / `executor_prompt.md` / `summary_prompt.md`（对齐参考的 evolve_plan / evolve_execute / evolve_summary）。
- 复用 `reference/v_iteration.schema.json`、`plan.md`、`profile_guide.md`。

---

## 6. 落地步骤（里程碑，A1）

> 注：本节是"代码阶段"的步骤，当前阶段**不执行**，待方案 + 核心文档确认后再开工。

1. **M0 对齐**：拿到钉钉核心文档，核对决策 A/B/C 与本方案，定稿设计。
2. **M1 数据库**：实现 `tools/evolution_db.py`（admit/select-parents/checkpoint/lineage），单测覆盖准入与父代选择；提供 `memory/v0.json → seed` 导入。
3. **M2 编排骨架**：新增 `skills/gpu-kernel-evolve/SKILL.md` + 四个子 agent 文档，先跑通"N=1、单岛"等价于现状的退化模式，确保不回归。
4. **M3 多候选 fan-out**：executor 并行产出 N 个 child，evaluator 批量评测、按门准入，落第一代 checkpoint。
5. **M4 种群与选择**：接入 Boltzmann 父代选择 + 新颖度准入 + 岛屿，验证血缘 `parent_id` 完整可回溯。
6. **M5 收尾**：summary 反馈闭环、停止/收敛判据、与 output-contract 串接；更新 `README.md` / `docs/design.md` / 架构图。
7. **M6 验证**：在一个真实 kernel（如 moe 方向）上跑 N 代，对比单轨迹基线的收敛速度与最终分数。

---

## 7. 风险与缓解

- **多候选并行的算力/时间成本**：N 越大越贵 → N 可配置，默认小值；evaluator 做编译失败早停。
- **进化数据库与现有 memory/git 双轨**：以 `database/` 为新真源，`memory/v<N>.json` 作兼容映射，避免两套真源打架。
- **profile 证据在多候选下的开销**：仅对"准入候选/每代 best"做完整 ncu，child 初筛用 `do_bench` 快测。
- **与 atrex 现有硬约束冲突**：gpu-wiki 溯源、"profile 驱动"等硬规则在新循环里**保持不变**，planner/evaluator 必须继承。

---

## 8. 待核对清单（拿到核心文档后逐条确认）

- [x] 目标形态：**A1 融入**（已确认，不移植 LoongFlow）
- [x] evaluator 运行环境：**本地**（do_bench + ncu/rocprofv3；远程评测预留接口、不在本期）
- [x] 种群参数：**N=3、单岛起步(≤2)、≤30 代、收敛=达标/连续5代无提升(<1%)/预算耗尽**
- [x] 是否复用参考实现代码/命名：**否**（仅借鉴概念；trace/checkpoint 目录命名对齐参考 schema）
- [x] 适用算子范围与平台：**通用**；**首验证 = moe 方向 @ H20**
- [x] 是否保留 partial-restart：**保留**，退化为种群停滞兜底 / 岛重置
- [x] trace/checkpoint 格式：对齐参考 schema（`database/` + `iteration/` + `evaluator/`，见 §5.1）

> 以上为本稿锁定的默认值。若后续拿到钉钉核心文档且某项与默认冲突，按该文档修订本方案对应条目。

---

*定稿（默认参数版）；当前不涉及任何代码改动。下一步进入 M1 时再开始写代码。*
