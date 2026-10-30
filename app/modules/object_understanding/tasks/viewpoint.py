from app.models.object_understanding import ViewpointData
from .base import TaskDefinition

TASK = TaskDefinition("viewpoint", "语义视角候选而非相机姿态", ViewpointData)
