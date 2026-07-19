# 跨模型评估集:22 个 bundle-sensitive query

在 Sonnet 干净数据(289 对,94正/195负)中,这 22 个 query 的 bundle 扰动会
**翻转成功/失败**(19 个)或**改变 reward 大小**(3 个)—— 即 bundle 真正起作用的地方。
每个 query × 4 bundle = **88 对,正 46 / 负 42**。这是测试门控"bundle 敏感的成功预测"
能否跨模型泛化的评估集。

22 个 query:
adaptive-cruise-control, civ6-adjacency-optimizer, court-form-filling, data-to-d3,
debug-trl-grpo, drone-planning-control, energy-ac-optimal-power-flow, exceltable-in-ppt,
exoplanet-detection-period, fix-erlang-ssh-cve, gravitational-wave-detection,
invoice-fraud-detection, jpg-ocr-stat, lab-unit-harmonization, lean4-proof,
mars-clouds-clustering, organize-messy-files, pddl-tpp-planning, powerlifting-coef-calc,
r2r-mpc-control, seismic-phase-picking, tictoc-unnecessary-abort-detection

已覆盖(截至打包):DeepSeek 62/88;qwen3-max / glm-4.6 / kimi-k2 各 0/88。
