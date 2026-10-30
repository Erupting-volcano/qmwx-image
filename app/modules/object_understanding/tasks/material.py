from app.models.object_understanding import MaterialData
from .base import TaskDefinition

TASK = TaskDefinition("material", "材料候选及其外观依据和不确定性", MaterialData)
