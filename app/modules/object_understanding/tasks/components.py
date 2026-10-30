from app.models.object_understanding import ComponentsData
from .base import TaskDefinition

TASK = TaskDefinition("components", "可见部件及其来源证据", ComponentsData)
