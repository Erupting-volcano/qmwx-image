from app.models.object_understanding import AppearanceData
from .base import TaskDefinition

TASK = TaskDefinition("appearance", "直接可见的颜色、形状、表面与特征", AppearanceData)
