VERSION = "1.0.0"
TEXT = """判断本图展示方向 viewpoint，取值仅 front/back/left/right/top/bottom/close_up/perspective/unknown。
这是基于可见特征的语义候选，不是相机姿态；必须给 evidence、asset_ids，status=candidate。
物体缺少可靠正背面参照时用 unknown 或 perspective，不把图像左边当成物体左侧。
无法判断时 viewpoint=unknown，解释限制，不推算相机参数。"""
