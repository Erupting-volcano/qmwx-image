# 物体定位与前景分割：SAM 2.1 Tiny

本版使用**用户框选目标、预训练模型生成像素掩码**的交互方式。物体名称仍用于档案命名，不会自动定位物体，也不会推断真实尺寸。未安装可选依赖或未配置权重时，基础上传、元数据和质量评估仍然可用。

## 选型与核实来源

| 方案 | 官方条件与许可证 | 本次决定 |
| --- | --- | --- |
| SAM 3 文字提示 | 官方运行要求包括 Python 3.12+、PyTorch 2.7+ 和 CUDA 12.6 兼容 GPU；权重需申请访问并认证，使用自定义 SAM License | 本次不集成。当前环境没有 CUDA GPU 和已授权权重，不适合快速完成本地验证 |
| SAM 2.1 Hiera Tiny 框提示 | 代码及权重为 Apache 2.0；Python 3.10+、PyTorch 2.5.1+、torchvision 0.20.1+；存在 CPU 路径，Windows 官方强烈建议 WSL Ubuntu | 只接入此后端，默认 CPU；用户提供一个矩形框，模型细化前景边界 |

官方来源：[SAM 3 仓库](https://github.com/facebookresearch/sam3)、[SAM 3 许可证](https://github.com/facebookresearch/sam3/blob/main/LICENSE)、[SAM 2 仓库](https://github.com/facebookresearch/sam2)、[SAM 2 许可证](https://github.com/facebookresearch/sam2/blob/main/LICENSE)。

可选依赖固定在审阅过的 SAM 2 官方提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。适配器使用其真实接口：`build_sam2()`、`SAM2ImagePredictor.set_image()`、`predict(box=..., multimask_output=False)` 和 `reset_predictor()`。参考[官方预测器源码](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py)。

CPU 配置关闭模型和预测器中的可选 CUDA 后处理，不代表承诺 CPU 实时性能。没有接入第二套大型模型，没有训练模型，也没有引入云端 AI API。

## 默认运行与启用方法

默认 `DIGITIZER_SEGMENTATION_ENABLED=false`，无需安装任何新增依赖。流水线仍记录每张图片的 `disabled` 状态和中文说明；这不是分割成功。顶层档案可以是 `completed_with_warnings`，表示原图和已有分析保存成功，但分割未执行。

以下安装是**用户主动选择的可选步骤**，本次开发没有执行下载或安装。PyTorch 和模型权重会占用较多磁盘空间。Windows 官方建议优先考虑 WSL Ubuntu；下面的 PowerShell 命令展示原生 Windows CPU 配置方式，原生运行仍需在你的环境完成真实模型验收。

建议使用独立环境，不覆盖现有 `.venv`。准备兼容的 Python（例如已安装的 Python 3.11）后：

```powershell
py -3.11 -m venv .venv-seg
.\.venv-seg\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-seg\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$env:SAM2_BUILD_CUDA = "0"
.\.venv-seg\Scripts\python.exe -m pip install -r requirements-segmentation.txt --no-build-isolation
```

请依据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择相互匹配的 torch/torchvision 版本。若采用 WSL，使用 Linux 对应的环境和命令；不要把 Windows 虚拟环境直接搬入 WSL。需要 CUDA 时应先核实 NVIDIA GPU、驱动及对应轮子，而不是只更改设备字符串。

Tiny 权重约 **156 MB**，需要手动从[官方权重链接](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)或 [Meta 官方模型页](https://huggingface.co/facebook/sam2.1-hiera-tiny/blob/main/sam2.1_hiera_tiny.pt)取得。不要执行会下载全部型号的脚本。

官方模型页提供的 Tiny 文件 SHA256：

```text
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
```

可以使用 `Get-FileHash -Algorithm SHA256` 核对下载文件。把文件放在 `weights/` 等明确的本地目录，随后设置：

```powershell
$env:DIGITIZER_SEGMENTATION_ENABLED = "true"
$env:DIGITIZER_SEGMENTATION_CHECKPOINT = (Resolve-Path .\weights\sam2.1_hiera_tiny.pt).Path
$env:DIGITIZER_SEGMENTATION_DEVICE = "cpu"
.\.venv-seg\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

配置项：

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DIGITIZER_SEGMENTATION_ENABLED` | `false` | 是否尝试执行；支持 true/false 或 1/0 |
| `DIGITIZER_SEGMENTATION_CHECKPOINT` | 空 | 本地 SAM 2.1 Tiny 权重，不接受自动下载作为替代 |
| `DIGITIZER_SEGMENTATION_DEVICE` | `cpu` | 仅允许 cpu/cuda；CUDA 不可用时明确返回不可用 |

权重和可选依赖只在启用且有有效目标框时加载。同一处理器缓存一个预测器，用锁保护 `set_image/predict`，避免并发请求串用图像特征；每个进程有自己的模型实例。修改配置或修复依赖后重启服务。权重目录和常见权重扩展名已加入 Git 忽略规则，权重不会随源码提交。

## 页面操作

1. 选择照片，逐图展开“框选分割目标”。
2. 在预览图上用鼠标或触屏拖出矩形，包住一个需要分割的物体。可以反向拖动、重画或清除。
3. 点击原有“建立数字档案”。模型已配置时会生成真实掩码；没有框的图片记录 `needs_input`。
4. 在“物体定位与前景分割”结果区查看原图、独立掩码、状态、面积、边界框及说明。并排预览中的白色是前景，黑色是背景。

矩形只是模型输入提示，程序没有把矩形直接填充成生产掩码。普通有效掩码仍要求用户核对是否选对目标；模型不能证明框中物体的身份。目标不清楚时，应重新框选或补拍，而不是根据候选分数认定结果一定正确。

本版创建新档案时提交目标框，不提供在已有档案上修改掩码的编辑平台。重新框选后再次创建会生成新的采集记录，不会悄悄覆盖旧档案。

## 提示与坐标约定

HTTP 创建接口新增可选 multipart 文本字段 `segmentation_boxes`，内容为 JSON 数组，长度必须与 `images` 一致；每项是 `null` 或 `[x0,y0,x1,y1]`。顺序按实际图片上传顺序关联，不按文件名关联。

输入坐标是 **EXIF 转正后显示图的比例坐标**：每个数在 `[0,1]`，左上小于右下。前端使用图像实际显示矩形换算比例，因此 CSS 缩放不改变提示意义。后端先按 EXIF 转正，再乘送入模型图像的宽高，得到官方预测器要求的像素框。

输出 `width/height`、掩码和 `bbox` 同属 `coordinate_system: "exif_oriented_pixels"`。`bbox` 为整数 `[x0,y0,x1,y1]`：左上包含，右下不包含，原点在左上角。`ObjectProfile.images` 既有宽高仍保留编码方向，不修改其含义。旋转不改变总像素数或面积比例。

通过 Python 调用时，在原有 `ImageInput` 中可选提供：

```python
from app.models.segmentation import NormalizedBox
from app.services.image_collection import ImageInput

# stream 的所有权和关闭方式与原有接口相同。
image = ImageInput("photo.jpg", stream, segmentation_box=NormalizedBox(
    x0=0.1, y0=0.1, x1=0.9, y1=0.9,
))
```

## 数据与文件约定

结果进入 `processing_results` 中的 `object_segmentation`，不增加物体档案顶层字段，不改变版本 `0.1.0`。

每图记录 `asset_id`、文件名、状态、配置的 `model_name`、实际 `prompt_type/prompt_box`、转正宽高、掩码路径、bbox、面积比例、候选分数、警告和必要的错误编号。`model_name` 标识选定后端；只有有效生成掩码的状态表示确实运行得到结果，不能从后端名称推断模型已经执行。`confidence` 是 SAM 的掩码质量预测，不是物体识别正确率或经校准的置信度。

| 逐图状态 | 含义 |
| --- | --- |
| `success` | 模型返回有效掩码，仍需人工检查目标和边界 |
| `needs_confirmation` | 有掩码，但面积极端或候选分数偏低/不可用 |
| `not_found` | 模型返回空前景，没有保存伪造掩码 |
| `failed` | 本图分析或掩码保存失败，其他图片继续处理 |
| `disabled` | 未启用，没有执行 |
| `unavailable` | 缺权重、依赖、设备或安全事务存储，没有执行成功 |
| `needs_input` | 缺少用户框选，等待新一次带框的采集 |

批量 `success_count/review_count/failed_count/skipped_count` 为互斥数量，总和等于 `input_count` 和逐图记录数。`failed_count` 包含 `failed/not_found`，`skipped_count` 包含 `disabled/unavailable/needs_input`。未执行或失败的记录没有掩码路径、边界框或伪造面积/分数。

当前面积复查阈值为 `<=0.001` 或 `>=0.98`，掩码质量分数 `<0.7` 或无有效分数时也需确认。这些集中定义在处理模块中，只是初始合理性筛查规则，未经真实拍摄样本校准。全图掩码不能标为普通成功。

掩码文件为单帧 8 位 `L` PNG，像素值只允许 `0/255`，尺寸与转正原图一致；不把像素矩阵写进 JSON。文件名为：

```text
objects/{object_id}/masks/{asset_id}_{artifact_id}.png
```

处理模块仅调用本次事务的 `save_mask()`。文件先写到 `.staging/<transaction>/objects/.../masks`；JSON 验证通过后随整物体目录提交。模块或适配器失败产生的未引用掩码会清理；临时写入碎片清理失败会阻止发布并回滚，不会混入正式档案。

流水线和存储层检查掩码归属、真实 PNG 内容、像素值、尺寸、实际 bbox 和面积。预览接口只接受档案与图片 UUID：

```text
GET /api/v1/objects/{object_id}/images/{asset_id}/original
GET /api/v1/objects/{object_id}/images/{asset_id}/mask
```

接口从档案中找资源并验证目录边界，不接受任意文件路径作为参数。没有有效掩码时返回安全的 404；最新分割记录失败时，不会静默返回历史掩码充当当前结果。

## 验证范围与限制

自动化测试用明确的测试后端验证模型边界、状态、坐标、掩码保存、API 预览和事务故障；生产缺省路径只调用官方 SAM2 适配器，没有启发式填框或假分割兜底。模型适配器单测验证固定配置、延迟加载、初始化次数、依赖和权重错误缓存。

真实权重烟测只有手动配置好依赖和权重、显式设置 `DIGITIZER_RUN_SAM2_SMOKE=1` 时执行：

```powershell
$env:DIGITIZER_RUN_SAM2_SMOKE = "1"
.\.venv-seg\Scripts\python.exe -m pytest tests/test_object_segmentation.py -k real_sam2_checkpoint_smoke -q
```

本次开发环境没有可选模型依赖和权重，因此真实模型烟测未执行。自动化测试通过不等于已经验证真实照片分割效果或 CPU 耗时。浏览器触屏、手机 EXIF 预览与真实模型输出仍需实机验收。

2026-10-30 最终完整测试为 **295 passed、1 skipped、1 warning**；原有基线为 178 项通过。跳过项仅为上述真实权重烟测，唯一 warning 是既有 Starlette TestClient/httpx 弃用提示。执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=data/.qa/pytest-segmentation-final-long-path-20261030-b --tb=short --show-capture=no
```

本轮另修复了 Windows 深目录下双 UUID 掩码路径超过 260 字符导致的保存和读取失败。扩展路径表示只用于底层文件操作，JSON 仍使用安全相对路径；长路径回归覆盖真实保存、目录提交、HTTP 原图及掩码预览，并确认没有破坏原有图片验证凭据复用。被适配器拒绝的分割数据会清除失效产物引用，失败文件不会混入正式档案。

分割按原图像素上限限制输入，逐张处理并释放中间图片和预测器图像特征。原图转正、RGB 和原尺寸掩码仍需内存，不能只按模型内部输入分辨率估算。透明图合成白底；带透明色的 16 位灰度图暂不支持分割，会明确记录失败并保留原图。分割不是材料、真实尺度、三维重建或工程几何结果。
