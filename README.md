# 其命维新——物体信息输入与数字化模块 V0.1

这是旧物改造辅助项目「其命维新」的物体信息输入与数字化模块，为后续改造方案生成、几何约束验证和操作规划提供标准化物体档案。V0.1 负责图片采集、合法性检查、基础元数据、图像质量筛查和统一数据输出，并提供可选的 SAM 2.1 Tiny 框选前景分割。基础环境使用 Python、Pydantic 和 Pillow；模型依赖单独安装，FastAPI 与中文页面是接入适配层和调试入口。

处理流程：**多图输入 → 限量写入与验证 → 元数据模块 → 图像质量筛查 → 可选框选分割 → 统一档案校验 → 原子保存**。默认模块顺序为 `ImageMetadataProcessor → ImageQualityProcessor → ObjectSegmentationProcessor`。

当前能力包括基础图片信息、质量筛查，以及配置好可选依赖和本地权重后的模型分割。分割默认未启用；需要用户在照片上框选目标，不会按物体名称自动定位。没有接入物体类别识别、材料识别、尺寸估计、三维重建或 CAD 建模；这些结果仍为未知。没有标尺或相机标定时，不会虚构物体尺寸。“高精度”是后续目标，不是 V0.1 的精度承诺。

## 接入主项目

支持两种接入方式，均使用同一套数据模型、校验、Pipeline 和存储逻辑：

- **通过 HTTP 接入**：独立运行下面的服务，主项目使用 `POST /api/v1/objects` 上传，使用返回的 `object_id` 查询和下载档案。
- **在 Python 中直接调用**：主项目能导入本模块代码时，调用 `ImageCollectionService`，无需启动 Web 服务或使用调试页面。服务层不依赖 FastAPI。

从当前项目根目录运行的直接调用示例：

```python
from pathlib import Path

from app.config import Settings
from app.services.image_collection import ImageCollectionService, ImageInput, CollectionError

collector = ImageCollectionService(Settings(data_dir=Path("data")))
try:
    with open("front.jpg", "rb") as front, open("side.png", "rb") as side:
        profile = collector.collect("主项目中的物体名称", [
            ImageInput(filename="front.jpg", stream=front),
            ImageInput(filename="side.png", stream=side),
        ])
    payload = profile.model_dump(mode="json")
    # 将 payload 交给主项目；图片和 profile.json 已经保存。
except CollectionError as error:
    print(error.code, str(error))  # 安全的输入错误；文件系统异常由调用方处理。
```

`ImageInput.stream` 是调用方提供的二进制文件流，从当前位置读取，由调用方负责关闭，也可以使用 `io.BytesIO`。`collect()` 返回已经落盘的 `ObjectProfile`；`collector.get_profile(object_id)` 可重新读取。每次调用创建一份新的收集记录，V0.1 不负责向主项目已有档案追加图片、管理用户权限或操作主项目数据库。

为兼容原始数据契约，返回模型仍称为 `ObjectProfile`，HTTP 路径仍为 `/api/v1/objects`；其中类别、材料、测量、几何和部件是留给主项目后续处理的明确空值或默认状态。主项目可保存 `object_id` 与自身业务 ID 的对应关系。`storage_path` 始终相对于本模块的数据目录，主项目解析路径时需要使用同一个数据根目录。

## 环境与安装

- Python 3.11 或更高兼容版本。
- Windows PowerShell；其他系统使用对应的虚拟环境 Python 路径即可。
- 不需要 Node.js、数据库、GPU、Docker 或外部 AI 服务。

以上为基础上传与质量评估环境。可选模型分割另外需要 PyTorch、SAM 2 与手动取得的 Tiny 权重；安装与配置见[分割模块说明](docs/segmentation.md)，不会随基础安装自动下载。

在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

下列命令直接调用虚拟环境的 Python，不需要调整 PowerShell 执行策略或激活脚本。

本次在 Windows、Python 3.13.9 上完成验证，`requirements-lock.txt` 记录了验证时的精确依赖版本；需要复现该依赖组合时，可将安装命令的文件名替换为 `requirements-lock.txt`。

## 启动与使用

仅在本机使用：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在电脑浏览器打开调试入口 [http://127.0.0.1:8000](http://127.0.0.1:8000)，输入物体名称，选择同一物体的多张 JPEG、PNG 或 WebP 照片，点击“建立数字档案”。页面显示文件信息、逐图质量问题和拍摄建议、未知信息以及 JSON 查看和下载入口。API 交互文档在 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

开发时可以在启动命令后追加 `--reload`。

### 手机局域网访问

1. 将电脑和手机连接到同一个可互通的局域网。
2. 使用以下命令监听电脑的网络接口：

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. 执行 `ipconfig`，找到当前无线或有线网卡的局域网 IPv4 地址，例如 `192.168.1.20`。
4. 在手机浏览器打开 `http://192.168.1.20:8000`，完成多图上传、结果查看和 JSON 下载。
5. 若无法连接，检查 Windows 防火墙是否允许该 Python 服务通过当前专用网络，路由器是否开启访客网络或客户端隔离，以及手机是否仍使用蜂窝网络。`0.0.0.0` 是监听地址，不能作为手机访问地址。

手机上传 HEIC/HEIF 等格式会被拒绝，请先转换为 JPEG、PNG 或 WebP。本版本未包含登录或公网部署配置，局域网访问范围由监听地址和网络环境决定。

**手机实机验收需要在你的网络中执行，自动化测试不能代替手机与防火墙的连通性验证。**

## API

| 方法和路径 | 用途 | 成功响应 |
| --- | --- | --- |
| `GET /` | 中文网页 | HTML |
| `GET /health` | 服务状态 | `200`，含 `status: "ok"` |
| `GET /api/v1/config` | 获取当前图片大小、数量、总量与像素限制 | `200`，不包含服务器目录 |
| `POST /api/v1/objects` | 上传图片、执行模块并创建档案 | `201`，含 `object_id`、`status`、`profile` |
| `GET /api/v1/objects/{object_id}` | 读取完整档案 | `200`，ObjectProfile JSON |
| `GET /api/v1/objects/{object_id}/profile.json` | 下载档案 | `200`，JSON 附件 |
| `GET /api/v1/objects/{object_id}/images/{asset_id}/original` | 按档案和图片 ID 安全预览原图 | `200`，原图片 |
| `GET /api/v1/objects/{object_id}/images/{asset_id}/mask` | 预览最新有效分割掩码 | `200`，二值 PNG；不可用时 `404` |

上传使用 `multipart/form-data`，名称字段为 `name`，每张图片使用同名的 `images` 字段：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/objects" `
  -F "name=示例物体" `
  -F "images=@D:\photos\front.jpg" `
  -F "images=@D:\photos\side.png"
```

将返回的 `object_id` 替换到以下命令中：

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/objects/替换为object_id"
curl.exe -o profile.json "http://127.0.0.1:8000/api/v1/objects/替换为object_id/profile.json"
```

创建响应中的 `profile`、查询响应、下载内容和磁盘 `profile.json` 表示同一份已校验档案。重启程序后仍可以通过 object_id 读取档案。

### 限制和失败语义

服务依据实际解码结果判断图片类型，不信任扩展名或浏览器声明的 MIME。默认限制如下，可通过环境变量配置；数值必须为正整数。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DIGITIZER_DATA_DIR` | 项目根目录下 `data` | 图片、JSON 和临时文件的根目录 |
| `DIGITIZER_MAX_IMAGE_BYTES` | `15728640` | 单张图片最大 15 MiB |
| `DIGITIZER_MAX_FILES` | `20` | 一次上传最大图片数 |
| `DIGITIZER_MAX_TOTAL_BYTES` | `104857600` | 一次上传图片总大小最大 100 MiB |
| `DIGITIZER_MAX_IMAGE_PIXELS` | `40000000` | 单张图片解码后的最大像素数 |

例如：

```powershell
$env:DIGITIZER_DATA_DIR = "D:\digitizer-data"
$env:DIGITIZER_MAX_FILES = "10"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

一批图片全部通过验证后才创建档案。任何图片验证失败，撤销本次上传，清理临时文件；不会保存只包含部分上传图片的档案。单张或整批大小超限返回 `413`，图片数量、空名称、缺少图片和图片损坏等返回 `422`，实际图片格式不受支持返回 `415`。不存在的档案返回 `404`，存储失败返回 `500`。错误响应统一包含 `detail.code` 和中文 `detail.message`。

图片合法后，某个处理模块失败会生成带错误标识的失败结果，并继续执行其他模块。档案仍会保存，整体状态为 `completed_with_warnings`；普通成功状态为 `completed`。前端不会收到异常堆栈或服务器绝对路径，排查时根据错误标识查服务日志。保存失败不会返回创建成功。

## 数据契约

- `schema_version`：当前为 `0.1.0`，用于后续数据兼容和迁移。
- `object_id`、`asset_id`：随机 UUID，文件系统保存时不使用上传文件名作为路径。
- `created_at`：带时区的创建时间。
- `images`：包含原文件名、相对存储路径、MIME、宽高、字节数和可选校验值。
- `category`：默认 `null`。
- `materials`、`components`：结构化候选列表。空列表表示尚无结果，不表示已确认不存在。
- `measurements`：默认包含 length、width、height 的空占位，不表示物体被假定为长方体；参数名称可扩展为厚度、孔径、孔距等。未知时 `value`、`unit` 和 `uncertainty` 为 `null`，`status`、`source` 为 `unknown`，`verified_by_user` 为 `false`。状态区分 `unknown`、`estimated`、`measured`、`confirmed`。不确定度表示与测量值相同单位的绝对误差。`confirmed` 必须显式记录用户确认或 `verification_method` 独立验证依据；算法输出数值不会自动升级为已确认。
- `geometry`：默认 `scale_status: "uncalibrated"`、`model_status: "not_started"`，所有模型路径为 `null`。`mesh_role` 区分 `unknown`、`appearance` 外观网格、`engineering` 工程几何；`engineering_status` 初始 `not_validated`，模型生成成功不等于通过工程验证。已校准必须记录 `scale_reference`，通过工程验证还必须记录验证方法。`quality_notes`、`unknown_regions` 留作质量描述与未知区域，空列表表示尚未评估。
- `component_relations`：通过部件 UUID 引用的结构或空间关系，默认空列表；引用必须属于当前档案。材料、部件、关系标记为 `confirmed` 时也需要显式验证依据。V0.1 不实际识别或生成这些结果。
- `missing_information`：列出仍待补充的档案字段，包括类别、材料、尺寸、几何和部件等。
- `processing_results`：按执行顺序保留各模块的数据、状态、警告和错误；模块状态支持 `success`、`partial`、`skipped`、`failed`、`needs_input`。
- `warnings`：汇总需要关注的问题。缺失信息和模块处理状态具有不同含义，完整执行元数据模块不代表已经取得物体的真实尺寸或模型。

## 目录与职责

```text
app/
  main.py                         应用装配与配置入口
  config.py                       本地目录和上传限制配置
  api/routes.py                   HTTP 接口
  models/                         Pydantic 统一数据契约
  modules/base.py                 模块接口与处理上下文
  modules/image_metadata.py       真实图片元数据提取
  modules/image_quality.py        图像质量启发式筛查
  modules/object_segmentation.py  可选的框选前景分割
  modules/sam2_backend.py         延迟加载的官方 SAM 2.1 Tiny 接口
  services/pipeline.py            顺序执行、异常隔离和结果交接
  services/profile_assembler.py   类型校验、适配器合并与缺失信息整理
  services/image_collection.py    主项目可直接调用的收集服务与输入接口
  storage/local_storage.py        上传事务与安全 JSON 写入
  web/                            中文 HTML、CSS 和 JavaScript
tests/                            不依赖网络、AI 或 GPU 的自动化测试
data/                             默认数据目录，实际生成文件不纳入版本控制
```

每个物体保存为：

```text
data/objects/{object_id}/
  images/{asset_id}.png            扩展名按实际解码格式确定
  masks/{asset_id}_{artifact_id}.png  可选的转正后二值分割掩码
  profile.json                    包含模块结果的完整档案
```

临时上传文件同样位于配置的数据目录中：`.multipart` 用于 HTTP 多部分解析的临时文件，`.staging` 用于尚未发布的上传事务。JSON 通过临时写入和替换保存，完整对象再原子发布到 `objects/{object_id}`，避免输出半份 JSON；失败上传的暂存数据会清理。

默认上传流程中，每张图片只执行一次完整合法性验证（格式和像素限制检查、`verify()`、重新打开后 `load()`）。本次调用内的 `InspectedImage` 记录真实检查所得元数据及文件身份、大小和修改时间；元数据与质量模块确认文件未变后复用验证依据，继续输出标准 `ProcessingResult`。质量模块仍需读取像素以计算指标，这次读取用于实际视觉计算。凭据不写入 JSON，不跨请求缓存，也不接受外部元数据字典作为验证依据。模块没有有效凭据、文件发生变化或像素限制变严时，仍调用已有完整验证逻辑。

HTTP 解析器目前在图片超过 1 MiB 时将上传流写入磁盘，然后存储事务再写一份暂存图片；较小图片的解析流留在内存，只有事务写入磁盘。改扩展名和提交目录都是同一数据目录内的移动，不再复制图片内容。暂时保留两层临时目录，以维持解析失败清理和整批事务回滚；具体审查与改动说明见 [重复处理审查报告](docs/refactor-review.md)。

`ProfileAssembler` 只负责数据创建、合并和校验，由应用服务调用本地存储完成落盘，避免数据整理层同时承担文件系统职责。

## 图像质量筛查

默认流水线按 `ImageMetadataProcessor → ImageQualityProcessor` 执行。质量结果保存在 `processing_results` 中 `module_name: "image_quality"` 的 `data`，创建、查询和 JSON 下载均可取得，不增加档案顶层字段，也不改变 `schema_version: "0.1.0"`。

每张成功检查的图片记录 `asset_id`、`filename`、原始 `width/height`、实际分析图 `analysis_width/analysis_height`、`mean_luminance`、`contrast_stddev`、`edge_strength`、`status`、`issues` 和中文 `suggestions`。使用资源 UUID 关联图片，同名文件仍分别记录。批量数据包含 `input_count`、`checked_count`、`review_count`；检查失败的图片保存在标准结果的 `errors` 中，不能当作通过筛查。

`good` 表示当前启发式指标没有发现明显问题，`review` 表示建议复查。`review` 仍是一次成功完成的分析，通过中文警告让档案成为 `completed_with_warnings`，不会拒绝保存合法照片。逐图异常分别记录，其余图片继续分析；全部失败也保留已验证的原图和带错误的基础档案。

阈值集中在 `app/modules/image_quality.py` 的 `QualityThresholds` / `DEFAULT_THRESHOLDS`，可通过 `ImageQualityProcessor(thresholds=...)` 自定义。以下是第一版初始值，只有合成测试图的行为验证，尚未经真实手机照片数据集校准：

| 筛查项 | 初始规则 | 含义 |
| --- | --- | --- |
| 分辨率 | 短边小于 480 或总像素小于 307200 | 提醒使用更高分辨率原图、靠近物体并补拍细节 |
| 过暗 / 过亮 | 灰度均值小于 45 / 大于 210 | 建议改善照明、避开强反光或调整曝光 |
| 低对比度 | 灰度标准差小于 20 | 建议改善照明并增加物体和背景的区分度 |
| 低细节响应 | 相邻灰度像素绝对差的均值小于 4 | 建议复查对焦、物体纹理和拍摄距离；平滑表面也可能响应低 |

亮度、对比度和边缘响应都使用 0–255 的 8 位灰度尺度，边缘指标不是拉普拉斯方差。分析图保持比例、不放大，最长边默认 1024；缩小后处理 EXIF 方向，透明区域合成白色背景后转灰度，原图不改写。JPEG 可利用解码缩减；PNG 等格式仍可能解码整张原图，因此峰值内存也受原图像素上限约束，不能只按缩略图大小估算。

原始宽高保留编码方向，分析图宽高按 EXIF 转正后的方向记录。普通 16 位灰度 PNG 会映射到 0–255；16 位灰度同时带透明色的 PNG 暂不支持质量评估，会明确记录该图评估未完成，避免 Pillow 转换造成错误曝光结论。这不会拒绝保存已经通过合法性验证的原图，也不会阻止其他图片评估。

这是启发式筛查，不是可靠的模糊诊断。整图平均指标可能遗漏局部过曝、局部模糊或物体过小，也会受背景、噪声及平滑表面影响。`good` 不保证适合三维重建，更不证明物体尺寸可信。Pillow 的缩略和方向处理行为见 [Image 文档](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail)与 [ImageOps 文档](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose)。

## 物体定位与前景分割

本版仅接入 **SAM 2.1 Hiera Tiny + 人工矩形框提示**。用户上传前展开每张照片，框住目标，再建立档案；矩形只给出大致位置，像素掩码必须由模型生成。框坐标按 EXIF 转正后的显示图片归一化，后端转为对应原图像素坐标。结果页并排显示原图、黑白掩码、状态、边界框、面积比例及中文提示。

默认未启用，未配置模型时不会假装成功；上传、元数据和质量评估仍正常保存。未启用或不可用会产生明确的未执行状态与说明，档案通常为 `completed_with_warnings`。没有模型依赖时可以正常启动；配置启用、有框且权重可用时才延迟加载，同一处理器缓存一个模型实例。

分割结果保存在 `processing_results.object_segmentation`（按 `module_name` 查找），不增加档案顶层字段，也不改变版本。掩码是与 EXIF 转正原图同尺寸的独立 `L` PNG，`255` 为前景、`0` 为背景；文件随原图和 JSON 一起暂存、提交或回滚，不把像素矩阵放进 JSON。全图、极小面积及低质量分数需人工确认，空掩码不保存。SAM 的候选分数不是目标身份正确率，任何结果都需检查是否覆盖选中的物体。

选型依据、许可证、可选 CPU/Windows 安装、权重校验、配置变量、API 提示格式、坐标与状态契约、测试范围见[分割模块说明](docs/segmentation.md)。本轮未安装大型依赖或下载权重，因此尚未完成真实模型推理和真实照片效果验收。

## 添加处理模块

新模块继承 `BaseProcessor`，声明唯一 `name`，实现同步的 `process(context)`，返回 `ProcessingResult`。上下文包含 `images`、`profile`、`previous_results`、`data_dir` 和 `resolve_asset(asset)`。处理模块读取上下文并返回自己的结果；不要直接修改整个档案或绕过存储层覆盖 JSON。

例如新增一个输出已知类别的演示模块（这里的固定类别仅用于说明接口，不是识别算法）：

```python
from app.models.processing_result import ProcessingResult
from app.modules.base import BaseProcessor


class ExampleCategoryProcessor(BaseProcessor):
    name = "example_category"

    def process(self, context):
        return ProcessingResult(
            module_name=self.name,
            status="success",
            data={"category": "演示类别"},
        )
```

为该模块定义 Pydantic 输出模型，并注册只更新相应字段的合并适配器：

```python
from pydantic import BaseModel

from app.modules.image_metadata import ImageMetadataProcessor
from app.services.pipeline import Pipeline
from app.services.profile_assembler import ProfileAssembler


class CategoryData(BaseModel):
    category: str


def apply_category(profile, result):
    payload = CategoryData.model_validate(result.data)
    profile.category = payload.category
    return profile


assembler = ProfileAssembler()
assembler.register_adapter("example_category", apply_category)
pipeline = Pipeline(
    [ImageMetadataProcessor(), ExampleCategoryProcessor()],
    assembler=assembler,
)
```

Python 集成时使用 `ImageCollectionService(pipeline=pipeline)`；HTTP 集成时使用 `create_app(pipeline=pipeline)` 创建应用，接口和网页无需修改。需要新档案字段时，在对应统一模型中显式定义类型；仅产生模块专属结果时，可以保存在该模块经过验证的 `data` 中。适配器收到档案副本，最终输出还会统一校验，禁止替换物体身份、原始图片和创建时间等受保护字段。

无适配器的模块仍会保留标准化的模块结果，不会把任意字典自动写入物体档案顶层。内置元数据、质量和分割模块均有专用输出模型与适配器，通过 `asset_id` 对应图片；质量与分割适配器只验证结果，不改写物体事实。分割文件另由流水线和存储层检查真实存在、归属及几何一致性。显式传入 `processors` 或 `pipeline` 时仍完全采用调用方配置，不自动追加默认模块。

`Pipeline.run(existing_profile, data_dir)` 和 `ProfileAssembler.merge_result(existing_profile, result)` 可以在已有档案副本上继续补充数据，保留已有模块记录。当前 HTTP 创建接口和 `collect()` 每次创建新档案；面向同一档案的持久化更新、追加图片及并发修订接口尚未实现，后续可在现有数据契约上增量加入。

V0.1 在当前 HTTP 请求内顺序处理。后续接入耗时三维重建时，应在现有 Pipeline 外增加任务调度、进度查询和恢复机制；请求路由、模块算法与数据契约各自演进。

后续路线依次为真实照片阈值校准与更细致的采集引导、COLMAP 多视角重建、Open3D 点云处理与几何分析、真实尺度校准与误差验证、视觉理解与分割、规则几何 CAD 导出，以及复杂物体的局部结构与质量验证。COLMAP、Open3D、CadQuery 均未作为当前依赖安装。复杂物体可以保留经过校准的网格、局部结构和未知区域，无需强行生成参数化 CAD。

## 自动化测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试在隔离的临时数据目录中生成小图片，覆盖模型序列化与未知值、多图保存、元数据提取、模块结果合并与失败隔离、无效输入、上传和像素限制、路径安全、失败清理、存储异常，以及创建响应、查询、下载、磁盘 JSON 一致性与重启读取。测试不访问外部 AI 服务、不下载模型、不要求 GPU。

还覆盖截断 multipart 请求整批拒绝、无 Content-Length 的流式请求超限、HTTP 临时文件落在指定数据目录并清理、Python 服务接口直接调用，以及测量确认依据和外观模型/工程验证状态的区分。

重复处理优化的回归测试还检查每张采集图片只完整验证一次、伪造元数据不能跳过检查、文件变化后重新验证、模块部分或全部失败仍保留档案，以及写盘、改名、JSON 保存和目录提交失败后的回滚。直接调用服务时仍检查名称、数量、各项限制及二进制流类型，并保留调用方对流的所有权。

质量评估测试覆盖细节图、低分辨率、过暗、过亮、低对比度、低细节响应、多图、逐图故障、EXIF/透明度、像素及路径限制、数据模型一致性、默认流水线和 HTTP 查询/下载/落盘一致性。页面渲染测试检查中文建议、重复文件名的资源对应关系和不可信内容的安全文本显示。

2026-10-29 本轮完整验证为 **178 passed，1 warning**，其中包含原有 104 项测试。唯一警告是既有 Starlette TestClient/httpx 弃用提示。页面 DOM 回归通过本机 Node 执行实际 `app.js` 的选图、提交和渲染事件链；没有 Node 时仅这组可选测试会跳过，服务本身不依赖 Node。浏览器页面已能打开，本轮文件选择接口故障导致未完成真实浏览器上传，不能将自动化结果视作手机实机验收。

分割测试另覆盖空/全图/异常掩码、模型与权重不可用、提示与 EXIF 坐标、混合图片故障、PNG 文件保存与安全预览、产物残留清理及提交失败回滚。模型边界测试使用只存在于测试代码的后端注入；真实权重烟测默认跳过，必须手动配置后显式开启，不能用测试掩码冒充已运行真实模型。

2026-10-30 分割模块整合后完整结果为 **295 passed、1 skipped、1 warning**。跳过的是未配置依赖和权重的真实模型烟测；已通过 Windows 长路径下的掩码保存、提交、读取和 HTTP 预览回归。详细测试命令与验证边界见[分割模块说明](docs/segmentation.md)。

手工验收时还应完成一次电脑浏览器与手机浏览器的真实上传，并核对页面查看、下载和本地文件。此步骤取决于实际浏览器、设备和局域网环境。
