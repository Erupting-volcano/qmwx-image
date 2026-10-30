"""Future SAM 2.1 adapter. No inference, heavy imports, or downloads occur here.

This class deliberately always refuses prediction, even if dependencies and a
checkpoint happen to exist. Filling configuration is not deploying the model.
"""

from importlib.util import find_spec
from pathlib import Path

from app.modules.segmentation_backend import (
    MaskPrediction, NotImplementedBackendError, SegmentationRequest, check_request_capabilities,
)
from app.modules.segmentation_factory import SAM2_CAPABILITIES, SAM2_NOT_IMPLEMENTED


class Sam2Backend:
    model_id = "sam2.1_hiera_tiny"
    model_name = model_id
    model_version = "2.1"
    capabilities = SAM2_CAPABILITIES
    status = "not_implemented"

    def __init__(self, checkpoint_path: Path | str | None = None, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        self.device = device

    def check_dependencies(self) -> dict:
        """Explicit local diagnostic only; no module import or GPU probe.

        Presence alone does not prove package version, CUDA, Python compatibility,
        checkpoint validity, or a working model. This adapter remains unimplemented.
        """
        missing = []
        for name in ("torch", "torchvision", "sam2"):
            try:
                present = find_spec(name) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                missing.append(name)
        return {
            "status": self.status, "missing_dependencies": tuple(missing),
            "checkpoint_configured": self.checkpoint_path is not None,
            "checkpoint_exists": self.checkpoint_path.is_file() if self.checkpoint_path is not None else False,
            "device_valid": self.device in {"auto", "cpu", "cuda"},
            "reason": SAM2_NOT_IMPLEMENTED,
        }

    def predict(self, request: SegmentationRequest) -> MaskPrediction:
        check_request_capabilities(request, self.capabilities)
        raise NotImplementedBackendError(SAM2_NOT_IMPLEMENTED)
