"""Optional, local-only SAM 2.1 Tiny adapter; no imports or downloads at startup."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from PIL import Image


SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class SegmentationUnavailable(RuntimeError):
    """A safe explanation for optional model configuration or loading failures."""


@dataclass(frozen=True)
class MaskPrediction:
    mask: Image.Image
    score: float | None = None


class Sam2Backend:
    """One cached predictor per configured processor, serialized across requests."""

    def __init__(self, checkpoint: Path | None, device: str = "cpu") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("SAM2 device must be cpu or cuda")
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.device = device
        self._predictor = None
        self._torch = None
        self._numpy = None
        self._load_error: str | None = None
        self._lock = Lock()

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if self._load_error is not None:
            raise SegmentationUnavailable(self._load_error)
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise SegmentationUnavailable("未找到本地SAM2权重，请配置权重路径后重新启动服务；不会自动下载。")
        try:
            import numpy as np
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            self._load_error = "未安装或无法加载SAM2可选依赖，请按分割安装说明配置后重新启动服务。"
            raise SegmentationUnavailable(self._load_error) from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise SegmentationUnavailable("已配置CUDA，但当前运行环境没有可用CUDA设备。")
        try:
            model = build_sam2(
                SAM2_CONFIG, str(self.checkpoint), device=self.device, apply_postprocessing=False,
            )
            self._predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
        except Exception as exc:
            self._load_error = "SAM2模型加载失败，请检查本地Tiny权重、可选依赖和设备配置后重新启动服务。"
            raise SegmentationUnavailable(self._load_error) from exc
        self._torch, self._numpy = torch, np
        return self._predictor

    def predict(self, image: Image.Image, box: tuple[float, float, float, float]) -> MaskPrediction:
        # set_image/predict share mutable embeddings; never interleave two requests.
        with self._lock:
            predictor = self._load()
            np, torch = self._numpy, self._torch
            with torch.inference_mode():
                try:
                    predictor.set_image(image)
                    masks, scores, _ = predictor.predict(
                        box=np.asarray(box, dtype=np.float32), multimask_output=False,
                        return_logits=False, normalize_coords=True,
                    )
                    array = np.asarray(masks)
                    if array.shape != (1, image.height, image.width):
                        raise ValueError("Unexpected SAM2 mask dimensions or candidate count")
                    if not np.isfinite(array).all() or not ((array == 0) | (array == 1)).all():
                        raise ValueError("SAM2 returned a non-binary mask")
                    mask = Image.fromarray((array[0] * 255).astype(np.uint8))
                    score = None
                    try:
                        score_array = np.asarray(scores).reshape(-1)
                        if len(score_array) == 1 and np.isfinite(score_array[0]) and 0 <= score_array[0] <= 1:
                            score = float(score_array[0])
                    except (TypeError, ValueError):
                        # Missing/invalid quality scores must not fabricate certainty.
                        pass
                    return MaskPrediction(mask=mask, score=score)
                finally:
                    predictor.reset_predictor()
