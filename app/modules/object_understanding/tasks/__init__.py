"""Explicit, small task registry; no provider initialization or model selection."""

from . import appearance, classification, components, material, multi_view_fusion, viewpoint


TASKS = {module.TASK.task_id: module.TASK for module in (
    classification, appearance, material, components, viewpoint, multi_view_fusion,
)}


def get_task(task_id: str):
    try:
        return TASKS[task_id]
    except (KeyError, TypeError):
        raise ValueError("未注册的物体理解任务") from None


def compose_output_schema(task_ids) -> dict:
    """Hoist namespaced definitions so nested task schemas keep valid references."""
    task_ids = tuple(task_ids)
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("组合输出结构需要非空且不重复的任务")
    properties, definitions = {}, {}
    for task_id in task_ids:
        schema = get_task(task_id).schema()

        def rewrite(value):
            if isinstance(value, dict):
                return {key: (f"#/$defs/{task_id}_{child.removeprefix('#/$defs/')}"
                              if key == "$ref" and isinstance(child, str) and child.startswith("#/$defs/")
                              else rewrite(child)) for key, child in value.items()}
            if isinstance(value, list):
                return [rewrite(child) for child in value]
            return value

        for name, definition in schema.pop("$defs", {}).items():
            definitions[f"{task_id}_{name}"] = rewrite(definition)
        properties[task_id] = rewrite(schema)
    return {"type": "object", "properties": properties, "required": list(task_ids),
            "additionalProperties": False, "$defs": definitions}
