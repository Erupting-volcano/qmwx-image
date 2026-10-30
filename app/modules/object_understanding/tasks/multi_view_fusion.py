from app.models.object_understanding import FusionData, FusionInput
from .base import TaskDefinition

TASK = TaskDefinition("multi_view_fusion", "带来源、别名、共识与冲突的语义融合", FusionData, FusionInput)
