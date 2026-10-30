from app.models.object_understanding import ClassificationData
from .base import TaskDefinition

TASK = TaskDefinition("classification", "有证据的物体类别、名称及用途候选", ClassificationData)
