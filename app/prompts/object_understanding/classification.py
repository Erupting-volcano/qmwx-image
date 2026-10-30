VERSION = "1.0.0"
TEXT = """根据可见证据给出物体大类 broad_category 和尽可能具体但不过度猜测的类别 category。
类别、物体名称和可能用途均为 candidate；保留有依据的多个类别候选。不能确定具体类别时使用较宽类别。
为 category_candidates、object_name、possible_uses 分别记录证据和 asset_ids，不把用户名称当作已验证事实。
未知内容写入 unknown_information，不为填满字段而猜测。"""
