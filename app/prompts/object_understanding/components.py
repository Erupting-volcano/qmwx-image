VERSION = "1.0.0"
TEXT = """识别 visible_components 中可见的组成部分，记录 component_name、description、evidence、asset_ids。
直接看见的部件标记 observed，解释性部件候选标记 candidate，绝不标记 confirmed。
只列可见部件；不假定照片外的底座、内部螺钉、隐藏连接或空间关系，也不推断精确尺寸。
不能区分整体与部件时保留未知，不为凑齐常见结构而补造。"""
