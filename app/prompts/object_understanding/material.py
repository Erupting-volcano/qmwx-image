VERSION = "1.0.0"
TEXT = """根据外观提出 material_candidates，而不是材料定论。
每项包含 material_name、evidence、asset_ids、uncertainty，status 只能是 candidate。
塑料、陶瓷、金属、玻璃、木材、纸、橡胶、复合材料仅是示例，不限制真实候选。
外观可能相似，应解释不确定性，不判断不可见的内部材料或合金牌号。confidence 可为 null；给出分数也不是经校准概率。
无法提出有依据的候选时返回空列表和 unknown_information。"""
