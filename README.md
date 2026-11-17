# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。模型推理出错时自动**故障隔离**：停止接单、
把失效模型上的未终止单结算为 `failed`，并需要一次成功的恢复探测才重新接单。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额 **429**；隔离中 **503** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果（任何状态均可查） |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409**；其他终态原样返回（幂等） |
| POST | `/recovery/probe` | 恢复探测：用 64×64 RGB 输入核查模型；成功 **200**，失败 **503**，并发探测 **409** |
| GET | `/recovery` | 隔离状态 / 是否有探测在跑 / 最近一次探测结果 |
| GET | `/stats` | 容量/状态计数（演示与观测用） |
| GET | `/health` | 存活探针 |
| GET | `/docs` | Swagger UI（OpenAPI: `/openapi.json`） |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：

```
queued → processing → completed
queued|processing → cancelled | expired        (原有规则)
queued|processing → failed                     (本批推理抛错；故障隔离)
```

`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。
订单视图新增 `failure` 字段：因推理错误被结算为 `failed` 的单为
`"inference_error"`，其余为 `null`。

## 故障隔离与恢复（本次新增）

### 隔离

- 任一批次前向计算抛错后，引擎**先**进入 `isolated`，再由 HTTP 层对外可见，
  因此新提交不会漏进第二个坏批次：`POST /predict` 立即返回 **503**
  （不读取图片、不占容量）。
- 该**故障批**中仍为 `processing` 的单 → `failed`（保留 `batch_id`）。
- **故障发生时仍在排队**的所有单 → `failed`（从未进入模型，`batch_id` 为 `null`）。
- 以上两类单一次性**释放全部占用**（`/stats.capacity_used` 归零）。
- 已 `cancelled` / `expired` 的单**保留原状态**、不被改写为 `failed`。
- 所有旧单仍可 `GET /orders/{id}` 查询；对终态单 `cancel` 返回 200 + 原状态。

### 恢复探测

- `POST /recovery/probe` 在**与原批次相同的计算通道**（同一模型、同一单线程执行器）
  上，用一张确定的 **64×64 RGB** 输入（`preprocessing.recovery_probe_input()`，
  与正常帧同一条归一化管线）做一次前向。
- 仅当返回**恰好 10 个有限 logits**（shape `[1, 10]` 且全部 `isfinite`）才算成功：
  - 成功 → 解除隔离、重新接单，返回 **200**，`probe.logits_shape == [1, 10]`；
  - 失败 → 保持隔离、继续 503，返回 **503**，`probe.reason` 为
    `exception` / `bad_shape` / `non_finite`，可**重试**；
  - 已有探测在跑 → 另一个请求立即返回 **409**（`reason: already_running`），
    不会并发跑两个探测、也不会额外跑一次模型。
- 探测不会阻塞事件循环：慢探测期间 `GET /orders/{id}`、`POST .../cancel`、
  `GET /stats`、`GET /recovery` 仍是毫秒级响应，`/stats.probe_running == true`。
- 服务健康时也可调用探测（等同一次金丝雀检查），不影响正常流量。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批
  （或一次恢复探测）在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。
- **故障隔离**：见上一节；故障批 + 当时排队单 → `failed` 并释放容量，取消/过期单原状态保留。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟 |
| `INJECT_FAILURE_ONCE` | `false` | **演示用**：一次性推理异常。`true`/`1` = 下一次推理调用（批或探测）抛错；`2` = 前两次（可演示“首次探测失败、重试成功”）；`0`/`false` 关闭。也接受别名 `INJECT_FAILURE_COUNT` |
| `PROBE_DELAY_MS` | 0 | **演示用**：恢复探测内的人为延迟，用于观察“慢探测期间查询/取消仍响应” |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |

## 演示：一次性异常 + 隔离 + 恢复

```bash
# 1) 打开一次性故障注入启动
INJECT_FAILURE_ONCE=true docker compose up --build
#    或在 docker-compose.yml 里把 INJECT_FAILURE_ONCE 改成 "true"

# 2) 第一张图触发坏批次：单号变为 failed
OID=$(curl -s -X POST localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["order_id"])')
sleep 0.5; curl -s localhost:8000/orders/$OID
# {"status":"failed","failure":"inference_error",...}

# 3) 隔离期间新提交 503
curl -i -X POST localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000" | head -1
# HTTP/1.1 503 Service Unavailable
curl -s localhost:8000/recovery
# {"isolated":true,"accepting_orders":false,"probe_running":false,...}

# 4) 恢复探测成功 -> 重新接单
curl -i -X POST localhost:8000/recovery/probe | head -1
# HTTP/1.1 200 OK    {"recovered":true,..."logits_shape":[1,10]...}
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict \
  -F "file=@samples/sample_circles.png" -F "deadline_ms=5000"
# 201
```

演示“首次探测也失败、可重试”：`INJECT_FAILURE_ONCE=2`。第一次 `POST /recovery/probe`
返回 503（`reason: exception`），再调一次返回 200。
演示“并发探测不并跑”：设 `PROBE_DELAY_MS=2000` 后并发两次探测，得到 200 + 409。
演示“慢探测不挡查询/取消”：探测进行中（`/stats.probe_running=true`）查询/取消仍毫秒级返回。

慢推理演示：把 `INFERENCE_DELAY_MS` 改为 `2000`，推理约 2 秒/批，
期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察）。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟；以及本次新增：

- 坏批次 → 批次内单 `failed`、当时排队单 `failed` 且释放容量、取消/过期单状态保留；
- 隔离期间 `/predict` 503、旧单仍可查询、终态单取消幂等；
- 探测成功 → 200 并恢复接单（且能正常完成新单）；
- 探测失败 → 503、保持隔离、可重试；模型返回 NaN 或错误 shape 时**不**恢复；
- 并发探测 → 一个 200、一个 409，实际只跑一次；
- 慢探测期间查询/取消/stats 均 < 0.5s 响应，且探测成功前 `/predict` 仍为 503；
- 探测输入确为单张 `[1, 3, 64, 64]` RGB。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
- 故障隔离只在**进程内**生效：不会自动重启容器或重新加载模型文件；
  恢复依赖模型本身在探测时已恢复正常（例如上游临时故障消失）。
- `INJECT_FAILURE_ONCE` 是进程级一次性计数，重启容器后重新计数。
