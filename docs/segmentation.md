# 多模型图像分割框架

本版在现有框选分割流程上增加统一请求、能力声明、结果契约和模型注册工厂。**当前只有百度智能云“智能抠图”Backend 已实现真实推理适配**；SAM 2.1 Hiera Tiny 只预留能力和配置，始终返回尚未实现的说明。没有下载权重、安装 PyTorch/SAM2、调用付费接口或把测试模拟输出当作模型推理结果。

默认仍选百度，分割默认关闭。选择尚未实现或未知后端时不会静默回退，基础上传、元数据和质量评估仍可继续。切换的是服务端配置；本次没有网页模型选择器，也没有重新设计原有 Pipeline 和事务。

选择并启用百度后端后，带有效框的照片会通过 HTTPS 发送至百度云，**发送整张经过 EXIF 转正与尺寸适配的照片，不是只发框内裁剪图**，可能产生调用费用；原图、掩码和 JSON 档案仍保存在本地。

## 总体结构和执行流程

```text
ImageCollectionService：整批合法性检查和事务暂存
  → ImageMetadataProcessor → ImageQualityProcessor
  → ObjectSegmentationProcessor
      → 根据配置查询注册表并延迟创建所选 Backend
      → 安全读取原图、复用验证依据、EXIF 转正、RGB 转换
      → 比例框换成转正原图像素坐标 → SegmentationRequest
      → 检查模型能力 → Backend.predict(request)
      → MaskPrediction：主要掩码、候选、分数语义和小型扩展
      → 校验所有掩码 → 原事务 save_mask()
  → ProfileAssembler / Pipeline 校验资源与产物
  → LocalStorage 保存 JSON、清理未引用掩码、提交或回滚
```

`segmentation_backend.py` 定义 `SegmentationBackend` Protocol、提示类型、`SegmentationCapabilities` 和扩展后的 `MaskPrediction`。`segmentation_factory.py` 管理注册信息、部署与配置状态、延迟构造。`baidu_backend.py` 保留真实 HTTP 实现；`sam2_backend.py` 是明确的未实现接入位置。处理器依赖统一契约，新增模型不需要复制图片校验、面积计算或事务文件操作。

## 已登记的模型和能力

| 配置 ID | 部署方式 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| `baidu` | cloud | 代码已实现，配置就绪时 `available`，缺凭证时 `unavailable` | 实际调用仍可能受网络、权限、额度及图片参数影响，状态查询不会付费探测 |
| `sam2.1_hiera_tiny` | local | `not_implemented` | 只登记元数据和配置；即使手动配置权重路径，也不能推理 |

能力是明确的布尔声明；SAM2 表中的“预留”描述未来适配目标，不代表当前已部署：

| 能力 | 百度已实现 | SAM 2.1 预留 |
| --- | --- | --- |
| `box_prompt` 矩形框 | 是 | 是 |
| `point_prompt` 前景/背景点击 | 否 | 是 |
| `mask_prompt` 现有掩码 | 否 | 是 |
| `text_prompt` 文字 | 否 | 否 |
| `automatic_mask` 自动主体提取 | 是 | 是，仅能力预留 |
| `multiple_masks` 多候选 | 否 | 是 |
| `quality_score` 模型质量分数 | 否 | 是 |
| `iterative_refinement` 迭代修正 | 否 | 是 |

调用前用 `check_request_capabilities()` 检查请求；不支持时抛出 `UnsupportedPromptError`，不忽略、补造或转换提示。能力声明、模型部署状态和服务实际执行是否成功是三件事：声明了能力的占位模型仍会被工厂拒绝创建；百度配置就绪也不保证远端请求成功。

## 统一请求、结果与扩展

`SegmentationRequest` 接收已经按 EXIF 转正的 `RGB` Pillow 图片，以及以下一种显式提示：

- `BoxPrompt(box=(x0,y0,x1,y1))`：转正原图的像素边界，左上包含、右下不包含，具有正面积且不越界。
- `PointPrompt(points=((x,y),...), labels=(1,0,...))`：相同像素坐标系，`1` 为前景、`0` 为背景；结构预留。
- `MaskPrompt(mask=...)`：同尺寸、`L` 模式、`0/255` 的提示掩码；结构预留。
- `TextPrompt(text=...)`：显式文本提示；结构预留，不擅自拿物体名称调用不支持文本的模型。
- `AutomaticPrompt()`：显式请求自动主体提取。只有支持此能力的后端可以执行；缺少上传目标框不等于自动模式。

统一 Backend 接口接收请求；百度另外保留旧 `predict(image, box)` 的兼容入口。下例展示直接使用统一请求的方式，只有实际调用 `predict()` 才会访问百度及可能计费；它不是默认运行的测试代码：

```python
from app.config import Settings
from app.modules.segmentation_backend import BoxPrompt, SegmentationRequest
from app.modules.segmentation_factory import create_backend

# oriented_rgb 需由调用方先完成安全读取、像素限制检查和 EXIF 转正。
# 此例仅展示调用边界；正常采集请继续使用 ImageCollectionService。
backend = create_backend(Settings())
request = SegmentationRequest(oriented_rgb, BoxPrompt((20, 30, 300, 400)))
prediction = backend.predict(request)  # 百度真实调用，可能计费；测试不执行此段。
# 使用完预测后，关闭 prediction.mask 及 candidates 中的 Pillow 图像。
```

Backend 内完成尺寸适配和输出归一化。`MaskPrediction.mask` 与每个 `MaskCandidate.mask` 都必须是输入转正原图大小的 `L` 二值掩码，`255` 为前景、`0` 为背景。处理器会再次检查，不信任后端自述。保留旧 `MaskPrediction(mask, score, warnings)` 的前三个位置参数，其余字段为可选关键字：

| 字段 | 缺省和含义 |
| --- | --- |
| `candidates` | 空元组；其他候选的 `mask/score/score_type/warnings` |
| `score` / `score_type` | 无可靠分数时均为空；有值时记录分数定义，例如 `predicted_iou`，不能跨类型或跨模型直接比较 |
| `model_id` / `model_version` | 实际使用的模型与版本；百度服务版本未公开，记录服务版本未声明的标识，不假装知道网络结构 |
| `processing_time` | 可空，非负有限秒数，后端的一次处理耗时 |
| `extensions` | 空字典；按模型 ID 命名空间隔离、总量不超过 16 KiB 的 JSON 信息 |

例如百度仅保留经过白名单选择的 API 模式等小型信息，放在 `extensions["baidu"]`；不复制原始 HTTP 响应、令牌、密钥、照片 Base64 或私密配置。未来模型可在自己的命名空间记录参数和状态引用。掩码、logits、交互缓存及大型二进制数据不能直接放入 JSON；候选掩码使用独立 PNG 和事务引用，未来 logits/状态文件需要单独完善产物适配后才能持久化。本次没有创建通用二进制缓存系统，也没有候选选择 UI。

## 注册、查找和增加第三种模型

注册表查询不会联网、加载权重或把缺配置变成一次收费探测。可分别查看原始注册信息和当前配置下的状态：

```python
from app.config import Settings
from app.modules.segmentation_factory import (
    SEGMENTATION_REGISTRY, describe_backend, get_backend_registration,
)

settings = Settings()
registration = get_backend_registration("baidu")
print(registration.capabilities.box_prompt)  # True，仅能力声明
current = describe_backend(settings.segmentation_backend, settings)
print(current.status, current.reason)  # 当前配置就绪与否
catalog = [item.to_public_dict() for item in SEGMENTATION_REGISTRY.list_models()]
```

`GET /api/v1/config` 暴露当前后端 ID、能力、状态和安全说明，页面据此显示可用性；不会返回密钥或本地 checkpoint 路径。`available` 只表示实现存在且必要配置已提供，不保证百度授权、网络或额度一定可用。未注册 ID 抛出 `UnknownBackendError`；未实现模型抛出 `NotImplementedBackendError`；初始化异常会转换为安全的 `BackendInitializationError`，不自动尝试第二个模型。

增加第三种模型时：

1. 新增独立 Backend 实现 `SegmentationBackend`：声明 `model_id/model_name/model_version/capabilities`，提供 `predict(request)`。Backend 完成其私有输入转换、真正推理和统一掩码归一化，不直接写档案或正式目录。
2. 在应用创建采集服务前调用 `SEGMENTATION_REGISTRY.register()`，提供元数据、能力、部署方式、状态及延迟构造函数。ID 不可重复注册或静默覆盖，实际 Backend 的身份、版本、能力必须与注册信息一致。
3. 将该 ID 配到 `DIGITIZER_SEGMENTATION_BACKEND`，重启后由同一个处理器和事务流程执行。需要新配置时只为该 Backend 增加明确的配置字段和安全检查。
4. 为真实 Backend 的请求转换、错误隔离和结果契约补测试，按模型部署方式单独管理可选依赖；当前未部署的模型保持 `not_implemented`。

注册代码示意如下；`your_package.segmenter` 是你未来实际编写的适配器，不是仓库内置模型，也没有模拟推理实现：

```python
from your_package.segmenter import YourBackend
from app.modules.segmentation_backend import SegmentationCapabilities
from app.modules.segmentation_factory import BackendRegistration, SEGMENTATION_REGISTRY

CAPABILITIES = SegmentationCapabilities(box_prompt=True)
SEGMENTATION_REGISTRY.register(BackendRegistration(
    model_id="your_model", name="你的模型", model_name="your_model",
    model_version="1.0", description="已完成真实适配的矩形框模型",
    capabilities=CAPABILITIES, deployment="local", status="available",
    builder=lambda settings: YourBackend(),
))
```

真实类必须使用相同 ID、版本和能力，只有实现完成后才标记 `available`。若需要配置检查，可传 `configuration_check(settings)` 回调：正常返回 `None`，缺配置返回适合公开显示的简短说明，不能把密钥或绝对路径拼进说明。

新增提示类型时先定义可验证的结构和明确的坐标含义，再扩展 `SegmentationCapabilities` 与 `check_request_capabilities()`，并只在真正支持的 Backend 中实现。现有点/掩码/文字结构不等于这些方式已接入网页或采集服务；需要新 HTTP 输入时还须明确增加请求解析与交互，不能把文字偷偷转成矩形框，也不能将“无框”当作自动分割。

## 不可用状态排查

| 现象 | 处理方式 |
| --- | --- |
| `disabled` | 按需设置 `DIGITIZER_SEGMENTATION_ENABLED=true` 后重启；关闭时不构造分割模型 |
| 未注册模型 | 检查 `DIGITIZER_SEGMENTATION_BACKEND` 是否与注册 ID 一致；不会自动回退 |
| SAM2 `not_implemented` | 这是本次的预期结果，安装依赖或填权重路径不能补齐未写的推理代码；要调用现有服务需明确选 `baidu` |
| 百度 `unavailable` | 检查当前进程的 API Key、Secret Key；开通智能抠图权限，重启后重试 |
| 百度调用 `failed` | 按安全错误说明与错误编号检查网络、超时、应用权限、额度和图片框大小；不会自动重试付费请求 |
| 有掩码但需确认 | 人工检查是否为目标物体以及细边界；不能用服务返回成功代替效果验收 |

可显式运行 `Sam2Backend(settings.sam2_checkpoint_path, settings.sam2_device).check_dependencies()` 查看依赖是否存在和权重路径是否配置；返回状态始终为 `not_implemented`。此诊断不导入 torch/SAM2、不验证版本或 CUDA 兼容性、不下载文件，也不证明已经有推理能力。

## 选型与官方来源

百度智能云的智能抠图支持矩形框引导，可以沿用现有框选页面与不规则物体的像素掩码流程。调用国内服务端接口，无需在本机安装 PyTorch、配置 GPU 或下载权重。本版只有这一种具备实际推理实现的后端；SAM2 预留不安装任何大型模型依赖。

官方来源：

- [智能抠图接口文档](https://cloud.baidu.com/doc/IMAGEPROCESS/s/rm8zl3koj)：矩形框 `control` 提示、请求格式、尺寸限制及掩码返回值。
- [调用方式与鉴权示例](https://cloud.baidu.com/doc/IMAGEPROCESS/s/2k3bclou0)：获取 access token 的表单 POST 示例。
- [Access Token 鉴权说明](https://ai.baidu.com/ai-doc/REFERENCE/Ck3dwjhhu)。
- [智能抠图产品与开通入口](https://ai.baidu.com/tech/imageprocess/segment)：以账号控制台显示的权限、计费和额度为准。

注册 ID 为 `baidu`，为兼容历史数据，结果中的 `model_name` 继续使用服务标识 `baidu_intelligent_segmentation`。官方接口没有公开承诺具体神经网络架构或版本，不能把该标识理解为 SAM2、SAM3 或某个已验证的模型版本。旧档案中的 `sam2.1_hiera_tiny` 标识继续保留、读取和预览。

## 开通与配置

1. 在百度智能云开通图像增强与特效中的“智能抠图”能力，并确认应用具有调用权限。
2. 在相应应用中取得 **API Key 与 Secret Key**。本接口不是千帆 OpenAI 兼容接口，不使用千帆 Bearer 密钥。
3. 只在运行后端的机器上设置环境变量，然后重启服务。

例如在 PowerShell 中设置当前终端进程的变量：

```powershell
$env:DIGITIZER_SEGMENTATION_ENABLED = "true"
$env:DIGITIZER_SEGMENTATION_BACKEND = "baidu"
$env:BAIDU_SEGMENTATION_API_KEY = "你的应用 API Key"
$env:BAIDU_SEGMENTATION_SECRET_KEY = "你的应用 Secret Key"
$env:DIGITIZER_SEGMENTATION_TIMEOUT_SECONDS = "30"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

以上是占位示例，不要将真实密钥复制到聊天、网页、源码、测试或 Git。密钥只在后端使用，不会放入浏览器代码或公开配置接口。仓库中的 `.env.example` 只有空白变量参考；应用读取进程环境变量，**不会自动加载 `.env` 文件**。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DIGITIZER_SEGMENTATION_ENABLED` | `false` | 是否启用所选分割后端 |
| `DIGITIZER_SEGMENTATION_BACKEND` | `baidu` | 模型注册 ID；当前另登记 `sam2.1_hiera_tiny` 占位 |
| `BAIDU_SEGMENTATION_API_KEY` | 空 | 百度应用 API Key，仅服务端使用 |
| `BAIDU_SEGMENTATION_SECRET_KEY` | 空 | 百度应用 Secret Key，仅服务端使用 |
| `DIGITIZER_SEGMENTATION_TIMEOUT_SECONDS` | `30` | 百度单次 HTTP 请求超时秒数，允许 1–120 |
| `SAM2_CHECKPOINT_PATH` | 空 | SAM2 未来权重路径接口，本版不加载或下载 |
| `SAM2_DEVICE` | `auto` | 预留 `auto/cpu/cuda` 设备配置，本版不执行设备探测或推理 |

只需在同一启动终端修改 `DIGITIZER_SEGMENTATION_BACKEND` 并重启服务；默认值 `baidu` 保持原有行为。选择 `sam2.1_hiera_tiny` 会得到明确的未实现说明，不能通过填写 checkpoint 直接获得推理能力；不会回退到百度。未选中的模型不加载依赖或权重，分割关闭时不要求任何凭证。SAM2 的诊断接口只查看配置和依赖是否存在，不安装、不导入大型模型库；它不改变 `not_implemented` 状态。

百度凭证只供所选百度适配器使用，SAM2 占位不读取它们来调用服务。缺少所选后端配置、模型未实现或初始化失败都通过安全说明记录，不会阻止基础图片功能启动。当前不提供 `requirements-sam2.txt` 安装流程，因为本次没有真实 SAM2 集成；未来实施时再按[官方仓库](https://github.com/facebookresearch/sam2)核实 Python、PyTorch 和 Windows/WSL 条件。关闭分割可将 `DIGITIZER_SEGMENTATION_ENABLED` 改回 `false` 并重启。

照片有云端处理需求时才应开启分割。每张带有效目标框的图片会触发一次分割请求，价格及额度请查[官方产品页](https://ai.baidu.com/tech/imageprocess/segment)和账号控制台；本项目不承诺固定价格或免费额度。

## 页面操作

1. 选择照片，页面显示所选后端和是否可用；逐图展开“框选分割目标”。百度可用时提示照片外发和可能计费，SAM2 占位会显示尚未实现。
2. 在预览图上用鼠标或触屏拖出矩形，包住一个需要分割的物体。可以反向拖动、重画或清除。
3. 点击“建立数字档案”。选择百度、启用且配置有效时，后端调用百度云；没有框的图片记录 `needs_input`，不自动切换到 `auto`。
4. 在“物体定位与前景分割”结果区查看原图、独立掩码、状态、面积、边界框及说明。并排预览中的白色是前景，黑色是背景。

矩形只是接口的目标提示，程序不会直接填充矩形充当生产掩码。百度接口不返回可靠的掩码质量分数，`confidence` 为 `null`；有有效掩码时仍记录需要人工确认，用户应检查目标和轮廓。目标不清楚时重新框选或补拍，不能根据服务返回了掩码就认定结果正确。

本版在创建新档案时提交目标框，不提供已有档案的掩码编辑平台。重新框选后再次创建会生成新的采集记录，不会覆盖旧档案。

## 实际调用与图片适配

访问令牌通过固定地址的表单 POST 获取：

```text
POST https://aip.baidubce.com/oauth/2.0/token
grant_type=client_credentials&client_id=<API Key>&client_secret=<Secret Key>
```

后端缓存有效令牌并按过期时间刷新；不会每张图片重复获取令牌。框提示使用 `method: "control"` 与 `position`；显式 `AutomaticPrompt` 使用 `method: "auto"` 并省略 `position`，不会把自动模式伪装成矩形框。当前网页仅提交框提示。分割使用官方接口：

```text
POST https://aip.baidubce.com/rest/2.0/image-process/v1/segment?access_token=<token>
Content-Type: application/json
```

```json
{
  "image": "<纯 Base64 图片内容>",
  "method": "control",
  "position": [[[100, 100], [500, 500]]],
  "return_form": "mask",
  "refine_mask": "false"
}
```

框坐标示例由后端按实际输入图计算，不能照抄为所有图片的提示。响应 `image` 是整张输入图大小的掩码 PNG Base64；项目不会下载接口返回的任意远程 URL，也不会用启发式填框作为 API 失败的替代。请求超时或服务失败不会自动重复计费调用，重试需重新发起采集。

官方尺寸条件为最短边至少 128、最长边最多 3000，Base64 图片内容不超过 10 MB。后端在现有图片验证与像素限制之内，受控缩小过大的照片，必要时为极端长宽比图片增加边缘留白，并同步映射提示框；框需位于图片边缘以内，过小的框应重新框选。返回掩码去掉适配留白后，按最近邻映回 EXIF 转正原图尺寸。处理过程不修改原始图片，发生尺寸适配时会给出说明。

服务端只访问固定百度 HTTPS 地址，限制响应大小，拒绝重定向。异常对外使用安全的中文说明；不会将密钥、access token、完整请求地址或照片 Base64 写入档案、页面和错误日志。

## 提示与坐标约定

HTTP 创建接口的可选 multipart 文本字段 `segmentation_boxes` 为 JSON 数组，长度必须与 `images` 一致；每项是 `null` 或 `[x0,y0,x1,y1]`。按实际图片上传顺序关联，不按文件名关联。

输入坐标是 **EXIF 转正后显示图的比例坐标**：每个数在 `[0,1]`，左上小于右下。前端使用图像实际显示矩形换算比例，因此 CSS 缩放不改变提示意义。后端先按 EXIF 转正，再将比例框映射到实际送入 API 的尺寸。

输出 `width/height`、掩码和 `bbox` 同属 `coordinate_system: "exif_oriented_pixels"`。`bbox` 为整数 `[x0,y0,x1,y1]`：左上包含，右下不包含，原点在左上角。`ObjectProfile.images` 既有宽高仍保留编码方向，不改变其含义。旋转不改变总像素数或面积比例。

通过 Python 调用时，在原有 `ImageInput` 中可选提供：

```python
from app.models.segmentation import NormalizedBox
from app.services.image_collection import ImageInput

# stream 的所有权和关闭方式与原有接口相同。
image = ImageInput("photo.jpg", stream, segmentation_box=NormalizedBox(
    x0=0.1, y0=0.1, x1=0.9, y1=0.9,
))
```

## 数据与文件约定和历史兼容

新增字段 `model_id/model_version/score/score_type/processing_time/extensions/candidates` 均有空值或空容器缺省值。原有 `model_name`、`confidence`、`mask_path`、状态与坐标含义不变；旧百度和旧 SAM2 JSON 不需批量改写。`score` 保留模型原始有限数值，`score_type` 说明定义；旧 `confidence` 仍保留原有 0–1 范围，超出该范围的模型分数只放 `score`，不截断或假装是概率。历史没有分数类型的值显示“历史记录未声明”，不能与新模型指标混比。

新记录只在 `predicted_iou` 为 0–1 或兼容未声明类型的旧质量分数时填充 `confidence`；文字匹配等其他评分即使在 0–1 内也只放 `score/score_type`。当前仅对明确的 `predicted_iou` 使用 0.7 的初始复查阈值，其余分数保留原值并要求人工确认；这不是经过效果校准的跨模型评分标准。

结果进入 `processing_results` 中的 `object_segmentation`，不增加物体档案顶层字段，不改变版本 `0.1.0`。每图记录 `asset_id`、文件名、服务标识、状态、实际 `prompt_type/prompt_box`、转正宽高、掩码路径、bbox、面积比例、候选分数、警告及必要的错误编号。只有有效生成掩码的状态表示已经得到结果，不能从服务标识推断请求已经执行。

| 逐图状态 | 含义 |
| --- | --- |
| `success` | 存量或已接入后端的有效掩码，仍应人工检查目标与边界 |
| `needs_confirmation` | 有掩码，需要人工确认；当前百度接口不提供可靠质量分数，使用此状态 |
| `not_found` | 接口返回空前景，没有保存伪造掩码 |
| `failed` | 本图分析、云请求或掩码保存失败，其他图片继续处理 |
| `disabled` | 未启用，没有发送分割请求 |
| `unavailable` | 未登记、未实现、初始化失败、缺少所选配置/权限或安全事务存储，没有执行成功 |
| `needs_input` | 缺少用户框选，等待新一次带框的采集 |

批量 `success_count/review_count/failed_count/skipped_count` 为互斥数量，总和等于 `input_count` 和逐图记录数。`failed_count` 包含 `failed/not_found`，`skipped_count` 包含 `disabled/unavailable/needs_input`。未执行或失败的记录没有掩码路径、边界框或伪造面积、分数。当前面积复查阈值为 `<=0.001` 或 `>=0.98`；只是初始合理性筛查规则，未经真实拍摄样本校准。全图掩码不能标为普通成功。

掩码文件为单帧 8 位 `L` PNG，像素值只允许 `0/255`，尺寸与转正原图一致；不把像素矩阵写进 JSON。文件名为：

```text
objects/{object_id}/masks/{asset_id}_{artifact_id}.png
```

主要掩码和非空候选掩码都仅调用本次事务的 `save_mask()`；候选记录使用安全相对 `mask_path` 与 bbox、面积、分数类型，不把 PIL 对象或像素写进 JSON。页面和现有 `/mask` 路径仍预览主要掩码。文件先写到 `.staging/<transaction>/objects/.../masks`；JSON 验证通过后随整物体目录提交。模块或适配器失败产生的未引用掩码会清理；临时写入碎片清理失败会阻止发布并回滚，不会混入正式档案。流水线和存储层检查掩码归属、真实 PNG 内容、像素值、尺寸、实际 bbox 和面积。

预览接口仍只接受档案与图片 UUID：

```text
GET /api/v1/objects/{object_id}/images/{asset_id}/original
GET /api/v1/objects/{object_id}/images/{asset_id}/mask
```

接口从档案中查找资源并验证目录边界，不接受任意文件路径作为参数。没有有效掩码时返回安全的 404；最新分割记录失败时，不会静默返回历史掩码充当当前结果。已有 Windows 长路径支持、图片验证复用与事务安全检查继续保留。

## 验证范围与限制

2026-10-30 本次框架版本全量测试为 **453 passed、1 skipped、1 warning**，包含 15 组页面 DOM 回归；跳过真实付费 API 烟测，唯一警告是既有 Starlette TestClient/httpx 弃用提示。没有执行真实模型精度验收。

自动化测试使用明确的测试后端及模拟 HTTP 响应，验证注册/查找、默认配置、未登记和未实现模型、能力拒绝、标准请求与结果、不同分数类型、延迟加载、历史 JSON，以及原有鉴权、请求格式、尺寸与坐标映射、状态、坏响应、单图失败隔离、掩码保存、查询预览和事务回滚；测试不需要真实密钥，不会外发照片或产生 API 调用费用。生产默认后端只调用百度智能抠图接口；SAM2 没有推理实现，不返回空白掩码充当结果，也没有假分割兜底。

本次没有配置实际账号密钥，没有执行真实云端推理；自动化测试通过不等于已经验证账号权限、额度、真实照片分割效果或网络耗时。使用者开通服务后仍需用非敏感样例验收。DOM 自动化不等于已经完成全部手机触屏和 EXIF 实机验收。

分割只能给出图像中的前景区域，不提供物体类别、真实尺寸、材料属性或三维结构。带透明色的 16 位灰度图片仍不支持，需转换为 8 位 PNG 或 JPEG；普通透明图片在发送副本上与白底合成，原图保持不变。
