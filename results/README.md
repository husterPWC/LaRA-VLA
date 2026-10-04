# 复现结果

本目录只保存体积小、可以直接审查的 JSON/CSV 结果摘要。原始日志、视频、
checkpoint 和生成产物保存在被 Git 忽略的目录中。

- `r1_checkpoint_preflight.json`：checkpoint 严格加载、tokenizer、processor、
  模型 shape、真实图像预处理和归一化统计检查。
- `r1_official_checkpoint_smoke.json`：使用发布 checkpoint 在单张 RTX 3090
  上执行的一次真实 LIBERO rollout。
