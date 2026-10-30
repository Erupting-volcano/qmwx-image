# 物体定位与前景分割：百度智能云 API

本版使用**用户框选目标、百度智能云“智能抠图”API 生成像素掩码**的交互方式，替换原来的本地 SAM2 后端。物体名称仍用于档案命名，不会自动定位物体，也不会推断真实尺寸。

分割默认关闭。启用后，后端会通过 HTTPS 将整张经过 EXIF 转正与尺寸适配的照片发送至百度云处理，**不是只上传框内的裁剪图**，可能产生调用费用；原图、掩码和 JSON 档案仍保存在本地。未启用、未配密钥或云服务不可用时，基础上传、元数据和质量评估仍然可用，不会伪造分割成功。

## 选型与官方来源

百度智能云的智能抠图支持矩形框引导，可以沿用现有框选页面与不规则物体的像素掩码流程。调用国内服务端接口，无需在本机安装 PyTorch、配置 GPU 或下载权重。本版只接入这一种云端分割后端，没有同时安装多套大型模型。

官方来源：

- [智能抠图接口文档](https://cloud.baidu.com/doc/IMAGEPROCESS/s/rm8zl3koj)：矩形框 `control` 提示、请求格式、尺寸限制及掩码返回值。
- [调用方式与鉴权示例](https://cloud.baidu.com/doc/IMAGEPROCESS/s/2k3bclou0)：获取 access token 的表单 POST 示例。
- [Access Token 鉴权说明](https://ai.baidu.com/ai-doc/REFERENCE/Ck3dwjhhu)。
- [智能抠图产品与开通入口](https://ai.baidu.com/tech/imageprocess/segment)：以账号控制台显示的权限、计费和额度为准。

结果中的 `model_name` 使用服务标识 `baidu_intelligent_segmentation`。官方接口没有公开承诺具体神经网络架构或版本，不能把该标识理解为 SAM2、SAM3 或某个已验证的模型版本。旧档案中的 `sam2.1_hiera_tiny` 标识继续保留、读取和预览。

## 开通与配置

1. 在百度智能云开通图像增强与特效中的“智能抠图”能力，并确认应用具有调用权限。
2. 在相应应用中取得 **API Key 与 Secret Key**。本接口不是千帆 OpenAI 兼容接口，不使用千帆 Bearer 密钥。
3. 只在运行后端的机器上设置环境变量，然后重启服务。

例如在 PowerShell 中设置当前终端进程的变量：

```powershell
$env:DIGITIZER_SEGMENTATION_ENABLED = "true"
$env:BAIDU_SEGMENTATION_API_KEY = "你的应用 API Key"
$env:BAIDU_SEGMENTATION_SECRET_KEY = "你的应用 Secret Key"
$env:DIGITIZER_SEGMENTATION_TIMEOUT_SECONDS = "30"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

以上是占位示例，不要将真实密钥复制到聊天、网页、源码、测试或 Git。密钥只在后端使用，不会放入浏览器代码或公开配置接口。仓库中的 `.env.example` 只有空白变量参考；应用读取进程环境变量，**不会自动加载 `.env` 文件**。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DIGITIZER_SEGMENTATION_ENABLED` | `false` | 是否启用云端分割 |
| `BAIDU_SEGMENTATION_API_KEY` | 空 | 百度应用 API Key，仅服务端使用 |
| `BAIDU_SEGMENTATION_SECRET_KEY` | 空 | 百度应用 Secret Key，仅服务端使用 |
| `DIGITIZER_SEGMENTATION_TIMEOUT_SECONDS` | `30` | 单次 HTTP 请求超时秒数，允许 1–120 |

旧的本地权重路径、设备配置及本地模型可选依赖不再用于当前分割后端。启用云端 API 不需要额外安装本地视觉模型依赖。关闭分割可将 `DIGITIZER_SEGMENTATION_ENABLED` 改回 `false` 并重启。

照片有云端处理需求时才应开启分割。每张带有效目标框的图片会触发一次分割请求，价格及额度请查[官方产品页](https://ai.baidu.com/tech/imageprocess/segment)和账号控制台；本项目不承诺固定价格或免费额度。

## 页面操作

1. 选择照片，逐图展开“框选分割目标”；页面提示分割是否启用，以及照片外发和可能计费。
2. 在预览图上用鼠标或触屏拖出矩形，包住一个需要分割的物体。可以反向拖动、重画或清除。
3. 点击“建立数字档案”。启用且配置有效时，后端调用百度云；没有框的图片记录 `needs_input`。
4. 在“物体定位与前景分割”结果区查看原图、独立掩码、状态、面积、边界框及说明。并排预览中的白色是前景，黑色是背景。

矩形只是接口的目标提示，程序不会直接填充矩形充当生产掩码。百度接口不返回可靠的掩码质量分数，`confidence` 为 `null`；有有效掩码时仍记录需要人工确认，用户应检查目标和轮廓。目标不清楚时重新框选或补拍，不能根据服务返回了掩码就认定结果正确。

本版在创建新档案时提交目标框，不提供已有档案的掩码编辑平台。重新框选后再次创建会生成新的采集记录，不会覆盖旧档案。

## 实际调用与图片适配

访问令牌通过固定地址的表单 POST 获取：

```text
POST https://aip.baidubce.com/oauth/2.0/token
grant_type=client_credentials&client_id=<API Key>&client_secret=<Secret Key>
```

后端缓存有效令牌并按过期时间刷新；不会每张图片重复获取令牌。分割使用官方接口：

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

## 数据与文件约定

结果进入 `processing_results` 中的 `object_segmentation`，不增加物体档案顶层字段，不改变版本 `0.1.0`。每图记录 `asset_id`、文件名、服务标识、状态、实际 `prompt_type/prompt_box`、转正宽高、掩码路径、bbox、面积比例、候选分数、警告及必要的错误编号。只有有效生成掩码的状态表示已经得到结果，不能从服务标识推断请求已经执行。

| 逐图状态 | 含义 |
| --- | --- |
| `success` | 存量或自定义后端的有效掩码，仍应人工检查目标与边界 |
| `needs_confirmation` | 有掩码，需要人工确认；当前百度接口不提供可靠质量分数，使用此状态 |
| `not_found` | 接口返回空前景，没有保存伪造掩码 |
| `failed` | 本图分析、云请求或掩码保存失败，其他图片继续处理 |
| `disabled` | 未启用，没有发送分割请求 |
| `unavailable` | 缺少有效服务配置、服务权限或安全事务存储，没有执行成功 |
| `needs_input` | 缺少用户框选，等待新一次带框的采集 |

批量 `success_count/review_count/failed_count/skipped_count` 为互斥数量，总和等于 `input_count` 和逐图记录数。`failed_count` 包含 `failed/not_found`，`skipped_count` 包含 `disabled/unavailable/needs_input`。未执行或失败的记录没有掩码路径、边界框或伪造面积、分数。当前面积复查阈值为 `<=0.001` 或 `>=0.98`；只是初始合理性筛查规则，未经真实拍摄样本校准。全图掩码不能标为普通成功。

掩码文件为单帧 8 位 `L` PNG，像素值只允许 `0/255`，尺寸与转正原图一致；不把像素矩阵写进 JSON。文件名为：

```text
objects/{object_id}/masks/{asset_id}_{artifact_id}.png
```

处理模块仅调用本次事务的 `save_mask()`。文件先写到 `.staging/<transaction>/objects/.../masks`；JSON 验证通过后随整物体目录提交。模块或适配器失败产生的未引用掩码会清理；临时写入碎片清理失败会阻止发布并回滚，不会混入正式档案。流水线和存储层检查掩码归属、真实 PNG 内容、像素值、尺寸、实际 bbox 和面积。

预览接口仍只接受档案与图片 UUID：

```text
GET /api/v1/objects/{object_id}/images/{asset_id}/original
GET /api/v1/objects/{object_id}/images/{asset_id}/mask
```

接口从档案中查找资源并验证目录边界，不接受任意文件路径作为参数。没有有效掩码时返回安全的 404；最新分割记录失败时，不会静默返回历史掩码充当当前结果。已有 Windows 长路径支持、图片验证复用与事务安全检查继续保留。

## 验证范围与限制

自动化测试使用明确的测试后端及模拟 HTTP 响应，验证鉴权、请求格式、尺寸与坐标映射、状态、坏响应、单图失败隔离、掩码保存、查询预览和事务回滚；测试不需要真实密钥，不会外发照片或产生 API 调用费用。生产默认后端只调用百度智能抠图接口，没有假分割兜底。

本次没有配置实际账号密钥，没有执行真实云端推理；自动化测试通过不等于已经验证账号权限、额度、真实照片分割效果或网络耗时。使用者开通服务后仍需用非敏感样例验收。DOM 自动化不等于已经完成全部手机触屏和 EXIF 实机验收。

分割只能给出图像中的前景区域，不提供物体类别、真实尺寸、材料属性或三维结构。带透明色的 16 位灰度图片仍不支持，需转换为 8 位 PNG 或 JPEG；普通透明图片在发送副本上与白底合成，原图保持不变。
