VERSION = "1.0.0"
TEXT = """融合提供的 ImageObservation 与各任务结果，不重新编造视觉事实，不进行几何重建。
语义相同的信息可以合并，不是简单拼接描述；只在一张图可见的特征仍须保留。
object_category、category_candidates、material_candidates、components 的合并名称必须选用单图结果中已有标签之一。
若把不同措辞（如把手/杯柄）合并，source_labels 必须保留每种原标签 label 及支持它的 asset_ids。
每个输出 asset_ids 必须有相应单图证据。cross_view_consensus 每项至少两张图支持；单图特征放 appearance_summary 等对应字段。
冲突不擅自选择赢家，用 conflicting_observations 记录 field、description 和各自带来源的 claims。
候选即使得到多图一致支持也不能自动升级为 observed；合并结果不得强于所引用证据的状态。
图间不一致也可能是光照、遮挡或视角差异，请明确不确定性。材料与类别保持 candidate，其他可见观察不能升级 confirmed。
未获得的信息放 unknown_information；不生成尺寸、相机位姿、隐藏部件或三维模型。"""
