# Privacy GateKeeper — Design Spec

> 状态：设计稿 v1.2（2026-05-12）。本文聚焦**机制可证明性**与**集成边界**。
> 范围：在 `AgentLoop` 入口处对用户消息做隐私检测、路径选择、（可选）用户确认、脱敏、出云、还原；并维护可审计的隐私会计。
> v1.1 → v1.2 变更：M1 / M1.5 / M3 实现已发货；本文档保留作为规格基线。
> 用户向使用指南见 [`docs/privacy-gatekeeper-m1.md`](../docs/privacy-gatekeeper-m1.md)。

---

## 0. 目标与非目标

**目标**
1. 阻止用户消息中的隐私实体以可识别形式进入云端 LLM。
2. 在数学上可证明的范围内提供可量化的隐私保证；超出范围的部分明确标注为**计算性可否认 (plausible deniability)**，不冒充无条件保证。
3. 在效用（任务可完成度）与隐私之间提供分级路径。
4. 提供可审计、可追溯、可销毁的处理记录。

**非目标**
- 不抵御已经获得端侧密钥/内存的本地攻击者。
- 不抵御与端侧串通的恶意小模型 supply chain（假定端侧模型可信）。
- 不替代 TLS / 网络层防护。
- 不保证 LDP 路径还原后的语义保真度（有损）。

---

## 1. 威胁模型 (Threat Model)

| 角色 | 信任级别 | 假设 |
|---|---|---|
| 端侧运行时（nanobot 进程） | trusted | 持密钥、能解密会话密钥 |
| 端侧小模型（local LM） | trusted | 模型权重不被攻击者反向还原 |
| 云端大模型 provider | **honest-but-curious** | 协议正确执行，但会尝试从输入推断隐私 |
| 网络中间人 | adversarial | 由 TLS 防御，本方案不重复处理 |
| 审计日志读者 | semi-trusted | 能看元数据/统计，原文受信封加密 |

**攻击者能力上限**（影响 k-decoy 的下界）：多项式时间，持有公开语言先验、可以观察到本方案对外暴露的所有路由决策与请求模式。

**攻击者能力下限**（影响 metric-DP 的 ε 选取）：拥有独立同分布的辅助信息但**不**持有目标用户的端侧 secrets。

---

## 2. 输入 / 输出契约

**输入**
- `raw_message: str`
- `session_ctx: SessionContext`（最近 N 轮消息、累计隐私会计、会话密钥句柄）
- `task_difficulty_hint: float ∈ [0,1]`（可选；由上游或简易分类器给出）
- `user_path_preference: ExecutionPath | None`（**新增**；用户预先指定的路径偏好，仍受 §4.5 安全底线约束）
- `channel_capabilities: ChannelCaps`（**新增**；声明该 channel 是否支持交互式确认、超时上限等）

**输出**
- `ExecutionPath ∈ {simple, normal, k-decoy, metric-dp, blocked}`（最终路径，可能来自系统推荐、用户确认或用户预选）
- `RecommendedPath: ExecutionPath`（**新增**；系统的初始推荐，用于审计）
- `AllowedPathSet: Set[ExecutionPath]`（**新增**；安全底线允许用户切换的候选集）
- `PathSource ∈ {system_auto, user_preselected, user_confirmed, fallback_timeout}`（**新增**；记录最终路径如何确定）
- `PrivacyMessage: str | List[str]`（k-decoy 路径输出 k 条）
- `DetectedEntities: List[Entity]`（type, span, risk_class, linkability, confidence）
- `RestorationPlan`（用于把云端响应映射回真实实体）
- `LLMResponse: str`（已还原；或在 blocked 时为安全拒绝消息）
- `AuditRecord`（元数据，原文加密；包含 RecommendedPath、用户选择、是否超时等）

---

## 3. 核心组件

### 3.1 PrivacyEntityDetector
两层级联：
1. **正则层**：邮箱、E.164 电话、中国身份证、银行卡（Luhn）、IPv4/IPv6、AWS/GCP/Azure 凭证模式、SSH key 头、JWT、PEM block、高熵字符串（长度 ≥ 20 且 Shannon 熵 ≥ 4.0 bit/char）。
2. **小模型语义层**：对正则未命中的 span 做命名实体补检；同时对正则命中做去伪（如电话号码上下文实为版本号）。

每个 `Entity` 必须带：
- `type` ∈ {EMAIL, PHONE, ID_NUMBER, BANK_CARD, ADDRESS, NAME, IP, GEO, MEDICAL, CREDENTIAL, KEY_MATERIAL, INTERNAL_INSTRUCTION, OTHER}
- `risk_class` ∈ {CATASTROPHIC, HIGH, MEDIUM, LOW}（见 §4.1）
- `domain_entropy_bits` 估计（用于决定是否能生成可信诱饵）
- `linkability` ∈ {SINGLE_USE, RECURRENT_IN_SESSION, RECURRENT_CROSS_SESSION}
- `confidence` ∈ [0,1]

**Fail-closed**：若 `confidence < τ_uncertain` 但语义提示存在，按 `MEDIUM` 处理而非放行。

### 3.2 ExecutionDecider（重画的决策树）

ExecutionDecider 现在产出**推荐路径**和**允许的候选集**，而非最终决定。最终决定可能由 ConfirmationGate（§3.7）根据用户输入或预设偏好覆盖，但任何覆盖都受 §4.5 安全底线约束。

输入：`{entities, task_difficulty, accountant.remaining_eps, session_ctx}`

```
# 第一阶段：硬底线（不可被任何 user choice 覆盖）
if any entity.risk_class == CATASTROPHIC:
    return Recommendation(path=BLOCKED, allowed={BLOCKED}, reason="catastrophic_risk")
if any entity.type ∈ {KEY_MATERIAL, CREDENTIAL, INTERNAL_INSTRUCTION}:
    return Recommendation(path=BLOCKED, allowed={BLOCKED}, reason="hard_secret")

# 第二阶段：系统推荐（可在 allowed 内由用户切换）
if task_difficulty < θ_simple AND端侧模型能力满足:
    rec = SIMPLE
elif entities is empty:
    rec = NORMAL                       # 受路由侧信道模式 §4.4 影响
elif all entities.risk_class ≤ MEDIUM
   AND entities.count ≤ k_max
   AND not any entity.linkability == RECURRENT_CROSS_SESSION
   AND entities have synthesizable decoys (domain_entropy_bits ≤ θ_synth):
    rec = K_DECOY
elif accountant.remaining_eps_session ≥ ε_required:
    rec = METRIC_DP
else:
    rec = BLOCKED                      # 预算耗尽 fail-closed

# 候选集 = {rec} ∪ 所有"≥ rec 严格性"的路径，且满足各自前置条件
allowed = {rec} ∪ stricter_paths_satisfying_preconditions(rec, entities, accountant)
return Recommendation(path=rec, allowed=allowed, reason=...)
```

参数（建议默认）：`τ_uncertain=0.4`, `θ_simple=0.3`, `k_max=3`, `θ_synth=12 bits`, ε 见 §4.3。

"严格性偏序"的定义（用于构造 allowed 集合）：`BLOCKED ≻ SIMPLE ≻ METRIC_DP ≻ K_DECOY ≻ NORMAL`。该偏序仅用于"用户可以单方面选择更严"的判断；实际的隐私强度比较见 §6。

### 3.3 PrivacyTransformer

#### 3.3.1 K-Decoy（重命名以避免与表格型 k-匿名混淆）

**M2 v1 简化版（已实现，2026-05）**：原方案的"端侧小模型生成 k-1 条诱饵消息 + 多轮发送"工程代价过高，且与 §5.1 的 AgentRunner 单消息约束冲突（cron / tool call 不能并发 K 次）。落地版采用"单消息 pseudonym 替换 + typed pool 作为隐式匿名集"：

- 对每个检测到的 entity，从 `EntityType -> List[str]` 的 typed pool 中按 `idx = HMAC_SHA256(deployment_key, session_key‖type‖canonical(value)) mod len(pool)` 选取一个 pseudonym 替换。
- **K_effective = pool size**（减去原值后剩余条数）。云端只见到单个 pseudonym；从云端视角看，该 pseudonym 等概率对应 pool 中任意一条同类型的真实值。
- pool 的来源：默认与 `MetricDPTransform` 共享 `_DEFAULT_POOLS`；用户可在 config 中覆盖以引入更大、更贴合自身领域的诱饵集合。
- 同 (session_key, value) → 同 pseudonym（多轮对话连贯）；不同 session → 不同 pseudonym（HMAC 跨会话不可链接）。
- **保证强度**：对计算受限、无先验的半诚实云端，区分器优势 ≤ 1/K_effective − 1/2 + adv(prior)；持有上下文语言先验时优势随轮次增长而退化。**不是无条件 k-匿名**。
- **与 METRIC_DP 的关系**：K_DECOY 不消耗 ε、不调用 embedding 模型；适合 ε 预算耗尽或 backend 不可用时的 fallback。提供的隐私保证严格弱于 METRIC_DP（无形式化 dχ-privacy），但好于 NORMAL（云端永远见不到原值）。
- **使用约束**：HARD_BLOCK_TYPES（KEY_MATERIAL / CREDENTIAL / JWT / INTERNAL_INSTRUCTION）由 decider 优先拦截到 BLOCKED；KDecoyTransform 自身对未注册 pool 或 pool 大小 < 2 的 entity 类型 fall back 到 `[<TYPE>_REDACTED]` placeholder，防御性退化。

**原方案的多消息 K-decoy（保留作为 v2 候选）**：真正发出 K 条并行查询、丢弃 K-1 条响应、拦截 decoy 路径的 tool call。工程代价 3-5x，需重构 AgentRunner 支持并行 cloud call + decoy 抑制。当 v1 在用户反馈中暴露"隐式匿名集太弱"时再升级。

#### 3.3.2 Metric DP on Token Embeddings（替代 "LDP"）
- 形式化：机制 M 是 ε-dχ-private（参考 Andrés et al. 2013；Feyisetan et al. WSDM 2020 MADLIB/SANTEXT）当且仅当对任意 x, x' 与任意输出集合 S：
  - `Pr[M(x) ∈ S] ≤ exp(ε · d(x, x')) · Pr[M(x') ∈ S]`
- 实现路径：
  1. 取 token embedding `e_x ∈ R^d`；
  2. 采样多元 Laplace 噪声 `η ~ Lap_d(0, b)`，其中 `b = 1/ε`，密度 `∝ exp(-ε‖η‖)`；
  3. 计算 `e' = e_x + η`；
  4. 在词表中取 `t' = argmin_t ‖e_t − e'‖`（最近邻投影）；
  5. 由后处理性质，`t'` 仍满足 ε-dχ-privacy。
- **敏感度选择**：以 embedding 矩阵的最大成对 L2 距离作为有效直径上界 `Δ`，便于解释 ε 与"语义混淆半径"。
- **保证强度**：严格 (ε)-dχ-privacy；并由组合定理给出会话累计 ε。
- **效用代价**：还原层（§3.4）有损；用户可见标注必须开启。

#### 3.3.3 伪名层（多轮一致性，所有路径共用）
- `pseudo(entity) = base32(HMAC_SHA256(session_key, type ‖ canonical(entity)))[:n]`
- session_key 从 OS keyring / 端侧 KMS 派生，session 结束销毁。
- 同会话内同一实体 → 同一伪名（保证模型推理连贯）。
- 跨会话密钥不同 → 攻击者不可链接。

### 3.4 Restorer
- **K_DECOY**：发出 k 条消息；端侧只取与真实消息对应的那条响应；其余丢弃。无需小模型还原（伪名直接反查）。
- **METRIC_DP**：端侧小模型基于 (原始上下文, 匿名映射, 云端响应) 生成还原文本。Prompt 设计应：
  - 显式给出"被替换的伪名 → 真实实体"的反查表（仅用于本地推理，不出域）；
  - 要求小模型保持云端回答的结构与论据，仅替换占位符与受噪声污染的语义片段。
- 输出标注：`response_metadata.fidelity ∈ {EXACT, RESTORED_LOSSY}`；UI 应展示。

### 3.5 PrivacyAccountant
- 维护 `(ε_session, ε_user_24h)`。
- 每次 METRIC_DP 调用扣减相应 ε（基本合成或高级合成；建议高级合成 + RDP 转换）。
- 超预算 → 决策器自动 fail-closed 到 BLOCKED 或 SIMPLE。
- 预算窗口：会话级（会话结束清零）+ 用户日级（滚动 24h）。

### 3.6 AuditLogger
- 记录字段：`timestamp, session_id_hash, path, recommended_path, allowed_set, path_source, user_choice_latency_ms, entity_types_count_by_type, risk_class_histogram, eps_consumed, decision_reason`。
- **不记录原文实体**；如需取证留存，使用信封加密（envelope encryption）：
  - 数据密钥 DEK 加密原文；
  - DEK 用 KEK 加密；
  - KEK 由独立 KMS / age recipient 管理，普通审计读者无权限。
- 日志本身写到 nanobot 的 memory 子系统外的独立目录，避免被 agent 工具读到。
- 当 `path_source == user_confirmed` 且 `path ≺ recommended_path`（严格性偏序意义下用户选了更弱的路径，目前只可能在等价安全档内横向切换，不可能突破底线），需在日志中显式标记 `user_downgrade=true`，便于事后审计。

### 3.7 ConfirmationGate（v1.1 新增）

负责在 ExecutionDecider 产出推荐之后、Transformer 执行之前，按配置决定是否与用户做一次确认交互。

**输入**：`Recommendation`（来自 §3.2）、`user_path_preference`（来自 §2 输入）、`channel_capabilities`、`config.privacy.confirmation`。

**核心流程**：

```
mode = config.confirmation.mode  # always | risk_threshold | never
threshold_met = (recommendation.path ∈ {METRIC_DP, K_DECOY, BLOCKED})
                OR (any entity.risk_class ≥ HIGH)

# 第一步：如有用户预选，先做底线校验
if user_path_preference is not None:
    if user_path_preference ∉ recommendation.allowed:
        # 用户预选违反安全底线（典型：CATASTROPHIC 时尝试选 NORMAL）
        # 默默拒绝降级，仍按推荐执行；审计标记 violation_attempt
        chosen = recommendation.path
        path_source = system_auto
        audit.violation_attempt = user_path_preference
    else:
        chosen = user_path_preference
        path_source = user_preselected
        if mode == always:
            # 即便有预选，always 模式仍需二次确认
            chosen, path_source = await ask_user(recommendation, default=chosen)
    return Decision(chosen, path_source)

# 第二步：无预选，按 mode 决定是否问
if mode == never:
    return Decision(recommendation.path, system_auto)
if mode == risk_threshold and not threshold_met:
    return Decision(recommendation.path, system_auto)

# 第三步：交互式确认（mode==always 或满足 threshold）
if not channel_capabilities.supports_interactive_confirm:
    return apply_channel_fallback(recommendation, config)   # 见 §5.4
chosen = await ask_user_with_timeout(recommendation, config.timeout_seconds)
if chosen is TIMEOUT:
    return Decision(BLOCKED, fallback_timeout)              # fail-closed
return Decision(chosen, user_confirmed)
```

**确认消息内容**（提供给用户）：
1. 检测到的实体清单（`type`, `span` 高亮，`risk_class`，`linkability`，`confidence`）。原文对用户可见——这是用户自己的数据。
2. 系统推荐的路径 + `decision_reason`（自然语言解释为何如此推荐）。
3. `AllowedPathSet` 中每条路径的简短描述与影响：
   - 该路径下"云端能看到什么"的诚实陈述（例：`METRIC_DP`→"云端将看到经噪声扰动的语义近邻替换"，`K_DECOY`→"云端将看到 k 条等价查询"，`BLOCKED`→"消息不发送"）。
   - 预计 ε 消耗（仅 METRIC_DP 路径）和会话剩余预算。
   - 预计响应保真度（`EXACT` / `RESTORED_LOSSY`）。
4. **取消选项**：用户可以选择"不发送此消息"。等价于切到 `BLOCKED` 但语义上是"撤回"，审计区分。

**安全属性**：
- ConfirmationGate 永远不能扩大 `AllowedPathSet`——该集合由 ExecutionDecider 按底线规则确定，是不可突破的上界。
- ConfirmationGate 可以让用户**升级**保护到任意 ≻ 推荐的路径（前提是该路径前置条件满足，例如 METRIC_DP 仍需有预算）。
- 用户的预选若违反底线 → 静默回落到推荐，不报错揭示底线规则细节（防止枚举试探）；但审计记录 `violation_attempt`。

**用户响应的关联**：
- 每次确认请求生成 `confirmation_id`（128-bit 随机），通过 channel 异步发出。
- 用户回复携带该 id（按钮回调 / 命令参数）。
- session 在等待期间挂起本 turn；同会话其他消息进队列，不抢先处理。
- 超时后该 id 失效，迟到的回复被忽略并审计。

---

## 4. 风险与参数

### 4.1 风险类映射（建议默认；可在 config 覆盖）

| risk_class | 实体示例 | 允许的路径 |
|---|---|---|
| CATASTROPHIC | API Key, SSH/PEM, JWT, 信用卡 PAN+CVV, 钱包私钥, 内部 system prompt | BLOCKED |
| HIGH | 身份证、银行卡号、医疗诊断、精确地理位置、未脱敏 IP+时间戳 | METRIC_DP only（且需用户明示授权） |
| MEDIUM | 邮箱、电话、姓名、组织、住址城市级 | K_DECOY 或 METRIC_DP |
| LOW | 一般地名、常见公司名、公开 URL | NORMAL（可走伪名层） |

### 4.2 K-decoy 的明确局限（写入用户可见文档）
- 仅对**单条消息**且**实体不跨轮重复**时有意义。
- 攻击者有上下文 → 优势退化；建议 `k ≥ 5`，但 k 越大 token 成本和延迟越高。
- 高熵实体（API Key、UUID、长哈希）**禁用**。

### 4.3 Metric-DP 参数建议
- 单次查询 ε 默认 `ε_query = 8`（语义对应"约 80% 概率落在原 token 的语义近邻 k=10 内"，可配置）。
- 会话预算 `ε_session ≤ 32`（≈4 次查询的基本合成上限；用 RDP/高级合成可放宽）。
- 用户日预算 `ε_user_24h ≤ 64`。
- 这些数字是**工程默认**，应允许 config 覆盖；不是普适最优。

### 4.4 路由侧信道缓解（可选 §5）
- **保守模式**（默认）：所有"含 entity 但不需 BLOCKED"的查询都走 METRIC_DP 或 K_DECOY；不向云端透露"无隐私"信号。
- **均衡模式**：对 NORMAL 路径以概率 `p` 注入 cover noise（即额外的诱饵 transform），对脱敏路径以概率 `q` 跳过——形成 randomized response，使得云端从单次观测无法可靠推断"该查询是否原本含隐私"。这需要把 `(p, q)` 纳入 ε 会计。
- **激进模式**：完全 honest routing，承担侧信道风险，仅用于内部测试。

### 4.5 用户覆盖的安全底线矩阵（v1.1 新增）

下表定义在不同检测结果下，用户可以覆盖到的最弱路径（"地板"）。**任何弱于地板的用户选择都被静默回落为系统推荐**。

| 检测结果 | 地板（最弱可选） | 候选集 |
|---|---|---|
| 任一实体 = CATASTROPHIC | BLOCKED | {BLOCKED} |
| 任一类型 ∈ {KEY_MATERIAL, CREDENTIAL, INTERNAL_INSTRUCTION} | BLOCKED | {BLOCKED} |
| 任一实体 = HIGH，预算充足 | METRIC_DP | {METRIC_DP, BLOCKED, SIMPLE*} |
| 任一实体 = HIGH，预算不足 | BLOCKED | {BLOCKED, SIMPLE*} |
| 全部实体 ≤ MEDIUM，K_DECOY 前置满足 | K_DECOY | {K_DECOY, METRIC_DP**, BLOCKED, SIMPLE*} |
| 全部实体 ≤ MEDIUM，仅 METRIC_DP 可行 | METRIC_DP | {METRIC_DP, BLOCKED, SIMPLE*} |
| 全部实体 = LOW 或无实体 | NORMAL | {NORMAL, METRIC_DP**, K_DECOY**, BLOCKED, SIMPLE*} |

注：
- `SIMPLE*`：仅当端侧模型能力满足任务时出现。
- `METRIC_DP**` / `K_DECOY**`：若用户主动升级到这些路径，仍需消耗 ε 预算或满足 K_DECOY 前置。
- "地板永不可被覆盖"是一条强不变量：实现时由 `AllowedPathSet` 约束 + ConfirmationGate 静默回落两层防御，不通过弹错给攻击者枚举机会。

---

## 5. 与 AgentLoop 的集成

### 5.1 集成点
- 位置：`nanobot/agent/loop.py`，在构造 `ProcessingContext` 之后、调用 `AgentRunner` 之前。
- 接口（草案，v1.1 含确认流）：
  ```
  gate = GateKeeper.from_config(config.privacy)
  recommendation = await gate.detect_and_recommend(
      raw_message,
      session_ctx,
      user_path_preference=ctx.user_path_preference,   # 来自 SDK 字段或 channel 解析
  )
  decision = await gate.confirm(
      recommendation,
      channel_capabilities=channel.capabilities,
      send_confirm=channel.send_confirmation,           # 由 channel 提供
      await_response=channel.await_confirmation_reply,  # 由 channel 提供
  )
  if decision.path == BLOCKED:
      publish_outbound(decision.refusal_message, metadata=decision.audit_view)
      return
  outcome = await gate.transform(decision)
  ctx.message = outcome.privacy_message      # 可能是 List[str]
  ctx.metadata.privacy = decision.audit_view  # path, recommended_path, source, fidelity
  response = await runner.run(ctx)
  final = await gate.restore(response, outcome)
  publish_outbound(final, metadata=decision.audit_view)
  ```
- **遵循 `.agent/design.md` 的 "core stays small" 约束**：GateKeeper 实现作为独立模块 `nanobot/privacy/`；`loop.py` 只调用门面。
- ConfirmationGate 的等待期间，本 turn 在 `AgentLoop` 中挂起；同 session 后续 inbound 消息进入队列按 FIFO 处理，避免抢先污染上下文。

### 5.2 配置（`nanobot/config/schema.py` 新增 PrivacyConfig）
按"explicit over magical"约束声明：
- `enabled: bool`
- `local_model: ModelRef`
- `risk_class_overrides: Dict[EntityType, RiskClass]`
- `regex_extensions: List[Pattern]`
- `k_decoy: { k_max: int, allowed_risk: List[RiskClass] }`
- `metric_dp: { eps_query: float, eps_session_max: float, eps_user_24h_max: float }`
- `routing_mode: 'conservative' | 'balanced' | 'honest'`
- `audit: { enabled: bool, kms_recipient: str | None, log_dir: Path }`
- `pseudonym: { hmac_key_source: 'keyring'|'env'|'kms' }`
- **`confirmation`**（v1.1 新增）：
  - `mode: 'always' | 'risk_threshold' | 'never'`（默认 `risk_threshold`）
  - `risk_threshold: RiskClass`（仅 `mode==risk_threshold` 生效，默认 `HIGH`）
  - `timeout_seconds: int`（默认 60）
  - `on_timeout: 'block' | 'recommended'`（默认 `block`，即 fail-closed）
  - `channel_fallback_default: 'forced_conservative' | 'use_recommended' | 'reject'`（默认 `forced_conservative`，见 §5.4）
  - `channel_fallback_overrides: Dict[ChannelName, FallbackPolicy]`（按 channel 覆盖默认）

### 5.3 与 Tools 子系统的边界
- 工具产出的内容（如 `read_file` 的返回）也可能包含隐私。**本期不把 GateKeeper 套到工具输出**，避免影响 agent 推理能力；改为在 `runner` 内对**送往云端**的 tool result 做二次轻量检查（仅 CATASTROPHIC 阻断）。完整覆盖留给 v2。

### 5.4 Channel 能力与 Fallback 策略（v1.1 新增）

**Channel 能力声明**（在 `nanobot/channels/base.py` 增加）：
- `supports_interactive_confirm: bool`
- `confirmation_max_latency_seconds: int`（channel 自己能保证的回执时延上限；ConfirmationGate 取 `min(配置 timeout, channel max_latency)`）
- `send_confirmation(...)` / `await_confirmation_reply(...)` 接口（仅在支持时实现）

**Fallback 策略**（当 channel 不支持或回执上限低于必要值时触发）：
- `forced_conservative`（默认）：直接执行 `AllowedPathSet` 中**最严**的路径（地板上方第一档常常就是 BLOCKED 或 SIMPLE，等价于"宁可不发也不冒险"）。审计标记 `path_source=fallback_no_confirm`。
- `use_recommended`：直接按系统推荐执行，等价于该 channel 关闭确认。透明但放弃用户主权。
- `reject`：拒收消息并向用户返回错误。最严但用户体验差，适合合规重场景。

**配置形式**：
```
confirmation:
  channel_fallback_default: forced_conservative
  channel_fallback_overrides:
    webhook: use_recommended    # 内部系统对接，信任 schema
    email_bridge: reject        # 高合规要求
    telegram: forced_conservative
```

**典型 channel 分类**（实施时填写默认值表，可被用户 config 覆盖）：

| Channel | 默认 supports_interactive | 默认 fallback |
|---|---|---|
| WebSocket / WebUI | ✅ | n/a |
| Telegram / Discord / Slack / Feishu | ✅（按钮回调） | n/a |
| Matrix / WhatsApp / QQ / WeChat | ✅（限制更多，需 channel 自测） | forced_conservative |
| Webhook（无回执） | ❌ | forced_conservative（可改） |
| Email bridge | ❌ | forced_conservative |
| 批处理/CLI 单次模式 | ❌ | use_recommended（CLI 用户已隐含确认） |

---

## 6. 数学保证总结（诚实陈述）

| 路径 | 形式化保证 | 实现状态 | 假设 |
|---|---|---|---|
| BLOCKED | 信息不出域 | ✅ shipped (M1) | 端侧不被攻陷 |
| SIMPLE | 信息不出域 | ⚠️ 接口存在，路径默认禁用（`local_model_available=False`） | 同上 |
| METRIC_DP | (ε)-dχ-privacy on transformed tokens；(ε_session, ε_user_24h) 由 PrivacyAccountant 强制；超预算 fail-closed | ✅ shipped (M3) — 已经实测真实 OpenAI embedding | 攻击者无端侧密钥；候选池 embedding 与原 token 共享同一 embedding model；Laplace 采样器实现正确（经验证：与理论 Gamma(d, 1/ε) 矩匹配，dχ-privacy 经验边界通过）；最近邻投影是后处理 |
| K_DECOY | 计算性可否认：单条消息上区分器优势 ≤ 1/K_effective − 1/2 + adv(prior) | ✅ shipped (M2 v1) — pseudonym + typed pool 隐式匿名集 | 攻击者多项式时间且无强先验；HMAC 密钥未泄漏；K_effective = pool size − 1 |
| NORMAL | **无保护**（按定义） | ✅ shipped (M1) | 已判定无隐私实体；接受路由侧信道 |

**M3 实测验证（2026-05-12）**：
- 真实 OpenAI `text-embedding-3-small` 1536D 向量经 OpenRouter 拿到，cosine 排序符合直觉。
- `alice@x.com` 端到端被替换为候选池中的某个邮箱，云端 echo 后 Restorer 正确还原。
- ε=8.0 真实记录到 accountant；预算耗尽时决策器正确 fail-closed 到 BLOCKED。
- 7 个端到端集成测试覆盖完整 pipeline 状态机。

**已知未解决问题（M4+ 候选）**：
1. 工具输出（filesystem / MCP / web fetch）含隐私时的统一处理 — agent 当前可读取本地文件并直接送云端。
2. 流式响应下的还原 — 当前设计假设全量响应；流式 token-by-token 还原会让攻击者通过响应到达时序部分推断。
3. 多语言 NER 的 recall 长尾 — `LLMSemanticDetector` 依赖 prompt 内容里的 category 列表，小模型对低资源语言可能漏报。
4. LM-assisted Restorer rewriting — 云端把 "Bob" 改写成 "Robert" 会让字符串级 restore 漏过；hook 已留接口（`backend=`, `use_llm_rewrite=`）但未实现。
5. 与 `MyTool` 自修改能力的交互 — agent 改自身 prompt 时如何不绕过 GateKeeper。

---

## 7. 实施分期

- **M1** ✅ shipped — 检测器（正则）+ 决策器（推荐 + AllowedSet）+ ConfirmationGate（三种 mode、超时、channel fallback）+ BLOCKED + NORMAL + 审计记录。
- **M1.5** ✅ shipped — SIMPLE stub 收口；`LocalModelBackend` 抽象 + `LLMProviderBackend` 适配器；CLI + WebSocket 交互确认；`local_model` 通过现有 `providers.*` 注册表；`LLMSemanticDetector` 自动 wire；`OpenAICompatProvider.embed` + 非规范 JSON httpx fallback。
- **M3** ✅ shipped — Multivariate Laplace 采样器 + token-level dχ-privacy `MetricDPTransform` + `PrivacyAccountant`（(ε_session, ε_user_24h) + 滑动窗口 + 持久化）+ `Restorer`（两段 sentinel 替换）+ GateKeeper / AgentLoop 端到端集成（含决策器 metric_dp_supported 动态判断）。已对真实 OpenAI embedding（经 OpenRouter）端到端验证。
- **M2** ✅ shipped (v1) — `KDecoyTransform` 实现 HMAC-pseudonym + typed pool 隐式匿名集；HMAC 密钥从 `NANOBOT_PRIVACY_KEY` env / `~/.nanobot/privacy_audit/pseudo_key` 文件解析（缺失时自动生成 32 字节随机密钥并 0600 持久化）；K_DECOY 与 METRIC_DP 共享 typed pool；K_DECOY 不消耗 ε、不需 embedding，自然作为 METRIC_DP 预算耗尽时的 fallback。多消息 K-decoy（真正并行 K 条云端调用）保留作为 v2 候选。
- **M4** 未开始 — 路由侧信道缓解（`routing_mode: balanced` + cover-traffic randomisation）；信封加密审计（KMS / age recipient 已留 `audit.kms_recipient` 配置位）；工具输出 PII 扫描（agent 读取的文件/MCP/web fetch 当前都直接出云）；MyTool 自修改防护。
- **WebUI** 未开始 — 渲染 `privacy_confirmation` envelope 弹窗。协议侧已就绪 + 测试覆盖；纯前端 PR。

每期独立 PR，符合 `.agent/design.md` 的 "minimal change" 与 "keep PRs reviewable" 约束。

---

## 8. 参考文献

- Sweeney, L. (2002). *k-anonymity: A model for protecting privacy.* IJUFKS.
- Machanavajjhala, A. et al. (2007). *l-diversity: Privacy beyond k-anonymity.* TKDD.
- Andrés, M. E. et al. (2013). *Geo-indistinguishability: Differential privacy for location-based systems.* CCS.
- Feyisetan, O. et al. (2020). *Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations.* WSDM (MADLIB).
- Yue, X. et al. (2021). *Differential privacy for text analytics via natural text sanitization.* ACL (SANTEXT).
- Mironov, I. (2017). *Rényi differential privacy.* CSF.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.*
