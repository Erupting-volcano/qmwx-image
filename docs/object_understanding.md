# 视觉语言模型框架与物体理解

本版在既有分割之后增加可独立关闭的物体理解模块。模型管理与分析任务分开：Backend 管理模型请求和供应商差异，任务管理提示词、结构化输出和证据校验。当前提供 Qwen3-VL-Plus 的 DashScope OpenAI 兼容接口适配；**本次只验证框架与模拟 HTTP 契约，未执行真实付费调用，也未验证真实照片识别精度**。

物体理解默认关闭，不要求 API Key 才能启动。图片采集、质量评估和独立配置的分割继续工作。启用后，经过尺寸适配的照片、可选的目标辅助图及已校验观察信息会发送到服务端配置的模型服务商，可能产生费用。更换 API Base URL 会改变接收方，必须由服务端管理员确认，不能把未经信任的地址作为上传表单输入。

## 执行流程

```text
图片整批验证与事务暂存
  → 元数据 → 质量评估 → 可选分割
  → ObjectUnderstandingProcessor
      → 安全读取原图、检查像素限制、EXIF 转正和 RGB 转换
      → 六项任务的独立提示词、Pydantic 输出模型与能力需求
      → RecognitionRouter 按 task_id 选择已注册模型
      → 将同一模型承担的单图任务组合为一次 analyze(request)
      → 校验 JSON、任务结构、来源及候选状态
      → 保留单图观察；有多个有效视角时另做语义融合
  → ProfileAssembler 校验/合并
  → 原有 LocalStorage 事务保存或回滚
```

分割成功时可以提供目标区域辅助图，但保留原图上下文。分割不可用不会阻止使用原图理解。每图的处理失败分别记录，其他图片继续执行；融合失败保留单图结果。该模块不自行写正式目录，也不覆盖原始图片、档案身份或处理历史。

## 模型管理接口

`app/modules/recognition_backend.py` 定义统一的请求、返回值、`RecognitionCapabilities` 和 Backend Protocol。Backend 声明 `model_id`、`model_version`、能力并实现 `analyze(request)`；输入可含单图、多图、文字指令、任务 ID 和结构化输出要求，返回经过规范化的字典、实际模型身份、耗时、警告和小型模型元数据。

`app/modules/recognition_factory.py` 提供注册表、配置检查和按任务路由：

```python
from app.config import Settings
from app.modules.recognition_factory import RECOGNITION_REGISTRY, RecognitionRouter

settings = Settings()
registration = RECOGNITION_REGISTRY.get("qwen3-vl-plus")
print(registration.capabilities)

router = RecognitionRouter(settings)
print(router.model_id_for_task("material"))
# get_backend() 延迟初始化；analyze() 才会产生推理请求。
backend = router.get_backend("material")
```

能力是模型及当前适配器的明确承诺，不由“OpenAI 兼容”四个字推导。首个 Qwen 适配支持图片、多图片及 JSON Object 结构化输出；没有声明未经本版核实和实现的 JSON Schema 强约束、多轮会话或视频输入。任务在调用前检查所需能力；不支持、未注册、未配置或初始化失败均返回明确说明，不尝试其他模型。

Qwen 的非思考开关、图片 Base64 编码、HTTP `response_format` 等放在 Backend 内，业务任务不直接创建供应商客户端。返回 JSON 仍必须通过任务 Pydantic 校验；JSON Object 模式只约束合法 JSON，不保证符合业务结构。异常不会带回 API Key、完整 Base64、私密服务地址或原始 HTTP 响应，不自动重试收费请求，也不通过另一次收费调用“修复”坏 JSON。

模型特有信息使用小型、命名空间隔离的元数据，例如允许的 token 用量或安全的请求编号。图片、掩码、logits、原始响应和大型状态不能作为 JSON 元数据写进档案；新供应商应明确选择允许保留的字段。

## 开通与配置

在百炼控制台取得有权限调用目标模型的 API Key，并确认地域、业务空间、API Base URL 和模型名称匹配。当前官方文档示例可能使用业务空间地址，例如 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。模板中的旧 `dashscope.aliyuncs.com` 地址只作为兼容参考，**不能保证对所有新账号、地域或业务空间通用**；以控制台和对应官方文档为准。

在运行后端的 PowerShell 进程中设置：

```powershell
$env:DIGITIZER_RECOGNITION_ENABLED = "true"
$env:DIGITIZER_RECOGNITION_BACKEND = "openai_compatible"
$env:DIGITIZER_RECOGNITION_MODEL = "qwen3-vl-plus"
$env:DIGITIZER_RECOGNITION_API_BASE = "https://你的业务空间地址/compatible-mode/v1"
$env:DIGITIZER_RECOGNITION_API_KEY = "你的服务端 API Key"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

示例中的地址和密钥为占位内容。真实密钥只由服务端环境变量提供，不填写到网页、提示词、JSON 配置、源码或聊天中。`.env.example` 是空白变量参考；程序不会自动加载 `.env` 文件。配置修改后需重启服务。

| 配置变量（均以 `DIGITIZER_RECOGNITION_` 开头） | 默认值 | 用途 |
| --- | --- | --- |
| `ENABLED` | `false` | 是否执行物体理解 |
| `COMBINE_TASKS` | `true` | 将同一模型承担的单图任务组合；设为 false 时逐任务执行 |
| `BACKEND` | `openai_compatible` | 默认适配器 |
| `MODEL` | `qwen3-vl-plus` | 默认已注册模型 ID |
| `API_BASE` | 兼容地址参考 | 服务端 HTTPS API Base URL，以控制台为准 |
| `API_KEY` | 空 | 默认模型的服务端凭证 |
| `TIMEOUT` | `60` | 单次请求超时秒数 |
| `MAX_IMAGES` | `10` | 每批最多分析的原始图片数；超出部分仍保存 |
| `MAX_IMAGE_SIDE` | `1536` | 发往模型的副本最长边限制，不改原图 |
| `MAX_CONCURRENCY` | `2` | 同一模型的并发请求上限 |
| `MIN_INTERVAL` | `0.25` | 请求启动的最小间隔秒数 |
| `MAX_OUTPUT_TOKENS` | `8192` | 费用/输出上限，过小可能截断 JSON |
| `TASK_MODELS` | `{}` | 任务 ID → 已注册模型 ID 的覆盖配置 |
| `MODEL_CONFIGS` | `{}` | 各模型独立的 API 地址、名称、凭证环境变量名、超时 |

输出 token 上限用于控制费用和资源，无法保证任何复杂物体都能在上限内完成。若响应被截断或结束原因不是正常完成，模块记录失败并保留其他结果，不自动追加收费请求；管理员可评估任务和费用后调整上限。设置并发上限不意味着每个任务都会独立并发调用：默认组合执行先减少请求数。

## 六个独立任务与提示词

提示词位于 `app/prompts/object_understanding/`，每项有独立版本；任务定义和输出校验位于 `app/modules/object_understanding/`。修改一个任务提示词不会修改其他任务或 Backend。每次观察记录实际使用的提示词版本，便于追溯。

| task_id | 职责 | 边界 |
| --- | --- | --- |
| `classification` | 大类、具体类别候选、名称、可能用途和依据 | 图片证据不足时使用宽泛类别，候选不等于确定事实 |
| `appearance` | 主/次颜色及分布、整体形状、表面和可见结构 | 不测量真实尺寸或精确几何 |
| `material` | 材料候选、依据、来源与可选自评分 | 材料状态只能为 `candidate`，不推断内部材料 |
| `components` | 可见部件、描述、依据和来源 | 仅为 `observed/candidate`，不推断隐藏部件 |
| `viewpoint` | front/back/left/right/top/bottom/close_up/perspective/unknown | 无法判断时 unknown，不计算相机姿态 |
| `multi_view_fusion` | 合并同义观察、保留独有特征、共同证据、冲突和未知项 | 语义融合，不做特征点匹配或三维重建 |

提示词明确区分直接观察、候选和未知；照片中的文字和其他输入数据不是可以改变系统规则的指令。结构化输出模型拒绝未经来源支持的图片引用，也不接受 VLM 自行声明 `confirmed`。模型返回 0.95 自评分不等于材质真实概率或用户确认。

修改提示词时同步更新该任务的版本，并重启进程加载新代码；本版没有在线提示词编辑器。记录提示词版本和正文指纹用于历史追溯与复用判断，不会因为只改一个任务就重写其他任务或供应商适配器。

模型需要专属措辞时，可在服务初始化阶段注册适配，任务输出结构保持不变。例如：

```python
from app.prompts.object_understanding import get_prompt, register_prompt_adapter

base = get_prompt("material")
register_prompt_adapter("second_model", "material", version="1.0.0-second-model",
                        text=base.text + "\n这里补充已验证的模型措辞适配要求。")
```

只有该模型的材料任务使用适配文本，其他任务和模型继续使用原提示词。实际版本与正文指纹进入观察记录和复用判断；公共证据约束仍由组合管理器统一加入，不能通过专属措辞绕过来源校验。重复注册会报错，修改后需重新启动服务。

## 默认调用次数与任务级路由

没有单独覆盖模型时，同一张图片的分类、外观、材料、部件和视角任务由组合提示词一次调用完成。各任务仍分别校验，某项无效不应把其他有效观察变成已确认事实。

- 关闭模块：0 次调用。
- 1 张需要分析的图片、默认路由：1 次综合观察，不做额外多视角融合。
- N 张需要分析的图片，且至少两张有有效观察：通常 N 次综合观察 + 1 次融合。
- 失败、图片数量限制、已有有效缓存或任务覆盖可能改变次数；系统不会为了修复失败自动重试。

若某个模型或输出上限不适合组合任务，可明确设置 `DIGITIZER_RECOGNITION_COMBINE_TASKS=false` 后重启，改为每图最多五次独立任务调用，再计算融合请求。此方式成本更高；程序不会在组合失败后自行追加五次付费调用。

给材料任务单独选择模型的配置示例：

```powershell
$env:DIGITIZER_RECOGNITION_TASK_MODELS = '{"material":"second_model","multi_view_fusion":"qwen3-vl-plus"}'
$env:DIGITIZER_RECOGNITION_MODEL_CONFIGS = '{"second_model":{"model":"已验证的模型API名称","api_base":"https://模型服务地址/v1","api_key_env":"SECOND_VLM_API_KEY","timeout":60}}'
$env:SECOND_VLM_API_KEY = "第二个模型的服务端密钥"
```

`second_model` 只是未来注册示例，并非本版内置模型；单写配置不会自动创建它，也不会继承 Qwen 能力。模型配置 JSON 只保存凭证环境变量名，不允许内嵌真实密钥。相同模型承担的任务可以继续组合；使用不同模型的任务才单独分组执行。某个覆盖模型不可用时，不静默把该任务送到默认模型。

增加第二种视觉模型时，先实现 `RecognitionBackend` 或复用经过验证的兼容适配器，声明能力，再使用 `RecognitionRegistration` 注册 provider、API model name、版本、部署方式、状态及延迟 builder。用 `RECOGNITION_REGISTRY.register()` 在创建处理器前完成注册，必要时增加独立配置检查和供应商适配；任务和提示词无需直接依赖供应商。

下面只演示注册边界；`your_adapter` 代表以后完成的真实适配器，本仓库没有这个示例模型，也不返回模拟推理结果：

```python
from your_adapter import build_verified_backend
from app.modules.recognition_backend import RecognitionCapabilities
from app.modules.recognition_factory import RecognitionRegistration, RECOGNITION_REGISTRY

RECOGNITION_REGISTRY.register(RecognitionRegistration(
    model_id="second_model", provider="your_provider", model_name="已核实的API模型名",
    model_version="已核实的版本或服务版本未声明标识",
    capabilities=RecognitionCapabilities(image_input=True, structured_output=True),
    deployment_type="cloud", availability_status="available",
    description="已完成真实适配，并有独立的能力依据",
    builder=build_verified_backend,
))
```

`builder(settings)` 应读取该模型自己的配置并返回身份/版本/能力与注册信息一致的 Backend。`resolve_model_config(settings, registration)` 可解析模型独立的 endpoint、凭证环境变量和超时。`MODEL_CONFIGS[id].model` 必须与注册的 `model_name` 一致；改成未知 API 名称必须另行核实和注册，不能让它继承 Qwen 的能力。

## 数据、来源和缓存

完整数据位于 `processing_results` 中 `module_name: "object_understanding"` 的 `data`，通过原有查询与 JSON 下载接口读取，旧档案没有此结果时仍可正常打开。物体档案版本和 HTTP 路径不变。

`ImageObservation` 保留每图视角、类别候选、外观、材料候选、可见部件、其他观察、未知项、警告、模型元数据和提示词版本。重要事实具有 `asset_ids` 与 `evidence`，每个引用都必须属于当前档案；单图任务不能引用其他图片。

`ObjectUnderstandingData` 保留综合类别、外观、材料和部件，以及单图观察、多视角共同支持和结构化冲突。共同结论必须保留多个来源，冲突分别保留各个说法及来源，不取最高置信分数强行选一个。单图整理不声明“多视角共同证据”。不覆盖用户已确认的类别、材料、部件、测量或几何信息；兼容档案字段中的 AI 材料仍为候选，可见部件也不是用户确认结果。

来源 ID 和标签校验能阻止引用别的档案或无来源的材料/部件标签，但不能证明自由文本的视觉事实与语义改写都正确。特别是融合后的外观描述和共同观察仍需人工复查，不能把“结构化校验通过”理解为真实识别精度已通过验证。

成功且未变化的同一档案图片可从处理历史复用；判断依据包括文件内容、任务模型配置、提示词版本及任务签名。变更文件、任务路由或提示词会使相应分析重新执行。失败结果不作为成功缓存。缓存只用于同一档案及图片的有效历史，不在不同物体之间推断身份。当前 HTTP `collect()` 每次创建新档案，不提供跨档案图片去重或已有档案持久化修订接口；重新上传不能承诺 0 费用。

页面展示来源、候选、逐图状态和融合冲突；所有文件名、观察和警告通过安全 DOM 文本写入，不执行模型返回的 HTML。完整模型元数据和提示词版本可在 JSON 中审查，页面不暴露 API Key 或私密配置。

## 失败排查与当前限制

| 现象 | 排查方向 |
| --- | --- |
| 未启用或缺凭证 | 检查当前服务进程的环境变量并重启，不影响基础上传 |
| 模型未注册或能力不足 | 显式注册与核实能力，不依赖兼容接口名称猜测能力 |
| 鉴权、权限或地址错误 | 核对地域、业务空间 Base URL、Key 和模型权限 |
| 超时、限流或额度不足 | 根据安全错误说明检查网络和账号；不会自动收费重试 |
| JSON/来源校验失败 | 保留其他任务或图片结果，核对提示词、输出上限和输出模型 |
| 多视角融合失败 | 查看已保留的各图观察及冲突，不把失败当作没有目标物体 |

当前没有真实识别精度评测、材料物理验证、隐藏结构推断、专用 OCR、跨物体匹配、本地 VLM 部署、训练、相机姿态、真实尺寸或三维几何。本次也没有网页模型选择器、提示词在线编辑器或后台队列；长耗时模型可在后续加外部任务调度。

自动化测试使用明确的测试后端和模拟 HTTP 响应，只验证结构、路由、能力、费用控制逻辑、来源、失败隔离、档案兼容及事务行为。不能把这些结果描述为 Qwen 识别真实物体成功。真实账号权限、费用、网络耗时、照片识别效果以及手机触屏仍需独立验收。

2026-10-30 最终全量结果为 **599 passed、1 skipped、1 warning**，包括 21 组页面 DOM 回归；跳过的是现存真实付费 API 烟测，警告是既有 Starlette TestClient/httpx 弃用提示。没有执行真实 Qwen 推理。测试覆盖组合/独立调用、任务模型覆盖、模型专属提示词、成功任务和融合复用、文件变化、辅助裁剪策略变化、部分与全部失败、来源与确认状态保护、聚合边界、档案查询下载及事务回滚。

官方依据：[视觉理解接口](https://help.aliyun.com/zh/model-studio/vision)、[结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output)、[OpenAI 兼容 Chat 接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。支持范围、地域地址与计费以当前官方文档和账号控制台为准。
