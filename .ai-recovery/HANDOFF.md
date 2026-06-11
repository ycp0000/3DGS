# Claude Code Recovery Handoff

Previous Claude Code session exceeded context limit. /compact failed.
Do NOT resume the old long conversation. Continue from repository state.

## Update 2026-06-03 EndoMoeGaussian engineering pass

### 已完成
- 新增/接入 `cams_gs_moe` 主线：三套独立 CAMS-GS experts + Gaussian-level MoE router。
- 修复新 MoE 文件的 Python 3.7 兼容风险，避免 `|` union / builtin generic 标注破坏服务器旧环境。
- 修复 EndoMoe scheduler 漏训 shared `tracking_time_encoder` 的问题，并补充 phase trainability 单测。
- 更新 README，将项目主线从 CAMS-GS baseline 改为 EndoMoeGaussian，并补充 cutting/pulling 训练、监控、对比命令。
- `.pytest_tmp_local/` 因 Windows 权限/句柄无法删除，已加入 `.gitignore`，避免污染 git 状态。

### 当前设计决策
- fine 动态阶段以 identity canonical static Gaussian 为起点，不再先经过原始随机 deformation backbone。
- EndoMoe 使用 `E_global / E_local / E_full` 三个独立专家，router 在 expert 训练后再学习 Gaussian 级组合。
- expert 训练阶段和 joint finetune 训练 shared `tracking_time_encoder`；router-only 阶段冻结 time encoder，只训练 router。
- EndoNeRF 数据必须使用绝对路径、`poses_bounds.npy` 和 `extra_mark='endonerf'`。

### 仍需做什么
- 继续复核 README 命令中的具体数据目录名是否与服务器实际目录一致。
- 如需 push，先处理 `.git/index.lock` 或残留 git 进程问题，且不要提交 `.ai-recovery/`、`.claude/logs/`、`.pytest_tmp_local/`。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_local`：67 passed。
- `python -m py_compile train.py scene/deformation.py models/tracking/cams_gs_moe_tracking.py models/tracking/cams_gs_visibility.py`：passed。
- README/type-hint 更新后已重新运行上述两项验证，仍为 67 passed / py_compile passed。
- time encoder scheduler 修复后重新运行：`python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_local`：67 passed。

### 下一步最小任务
- 在服务器用 `arguments/endonerf/cutting_endomoeg.py` 跑一次短动态启动验证，确认 fine 阶段 PSNR 不再跌到 9 左右。

## Update 2026-06-03 CAMS-GS fine-stage collapse follow-up

### 已完成
- 根据服务器日志定位到新异常：`phase=global_only` 时进度条仍显示 `uG0/uG1/uG2=0.33/0.33/0.33`。
- 修复 `gaussian_renderer/__init__.py` 中 pixel routing 的背景污染问题：路由权重渲染改用黑背景 rasterizer，并用 `gaussian_pi_geo_prior` 屏蔽 inactive expert。
- 增加白背景回归测试，覆盖 inactive expert 因背景被误判 covered 的场景。

### 当前设计决策
- CAMS-GS/EndoMoe 的 Gaussian-level prior 是专家激活的硬约束；pixel routing 只能在 active expert 内做可见性/覆盖权重细化。
- 背景颜色不能参与 expert coverage 判定，否则 white background 会把 inactive expert 权重推成近似均匀分布。

### 仍需做什么
- 运行 renderer pixel routing 相关测试和整组 CAMS-GS/EndoNeRF preset 测试。
- 测试通过后提交并推送修复。

### 运行过哪些测试
- `python -m pytest tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_ignores_background_for_inactive_experts tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_masks_uncovered_experts_and_aggregates_radii tests/test_disentangled_moe_tracking.py::test_cams_global_only_phase_disables_local_and_cut_graph_contributions -v --tb=line --basetemp .pytest_tmp_local`：3 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_local`：68 passed。
- `python -m py_compile gaussian_renderer/__init__.py models/tracking/cams_gs_tracking.py models/tracking/motion_decomposition.py`：passed。

### 下一步最小任务
- 提交并推送 renderer pixel routing 修复，然后让服务器重新拉取并短跑 fine 启动阶段。

## Update 2026-06-03 render/cfg_args/tensorboard fixes

### 已完成
- 定位 TensorBoard 缺失原因：`training()` 绕过了 `prepare_output_and_logger()`，直接把 `tb_writer` 设为 `None`。
- 修复 `arguments/__init__.py` 读取旧 `cfg_args` 时遇到 `nan/inf` 报 `NameError` 的问题。
- 修复 `render.py` 点云重建对 depth/mask 固定 `squeeze(0)` 的假设，兼容 `[H,W]`、`[1,H,W]`、`[H,W,1]`、`[1,1,H,W]`。
- 增加 `cfg_args nan` 和 render depth shape 回归测试。

### 当前设计决策
- 继续兼容历史 `cfg_args` 的 `Namespace(...)` 格式，但用受限 eval 环境，只暴露 `Namespace/nan/inf`。
- TensorBoard writer 恢复到 `./output/<expname>/events.out.tfevents...`，训练结束显式 `flush/close`。
- render 重建阶段不再假设 depth 一定有 channel 维度。

### 仍需做什么
- 运行新增测试、相关 preset/render 测试和 py_compile。
- 测试通过后提交并推送到 GitHub。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py::test_get_combined_args_reads_cfg_args_with_nan_values tests/test_disentangled_moe_tracking.py::test_render_reconstruction_accepts_hw_and_channel_first_depth_shapes -v --tb=line --basetemp .pytest_tmp_local`：2 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_local`：70 passed。
- `python -m py_compile train.py render.py arguments/__init__.py gaussian_renderer/__init__.py`：passed。

### 下一步最小任务
- 提交并推送本轮 `train/render/arguments` 修复。

## Update 2026-06-03 latest TensorBoard black-frame analysis

### 已完成
- 解析最新事件文件 `events.out.tfevents.1780478441...61438.0`，确认 fine 初始没有崩：step 1 L1=0.0375。
- 定位两个黑帧/跳变来源：
  - fine step 500->501 和 3000->3001 的巨大跳变来自 fine 阶段 opacity reset。
  - 中后期 `usage_geo_global/local/cut_graph` 同时为 0 的 285 个 step 来自 pixel routing 覆盖为空时输出全零权重。
- 修复 `train.py`：opacity reset 仅允许 coarse 阶段执行，fine 动态拟合不再重置静态重建得到的 opacity。
- 修复 `gaussian_renderer/__init__.py`：pixel routing 在无专家覆盖像素上使用 Gaussian prior fallback，而不是全零权重。
- 增加 renderer fallback 回归测试。

### 当前设计决策
- EndoGaussian/CAMS-GS 动态阶段必须保留静态阶段重建好的 opacity；fine 阶段 reset opacity 会直接制造黑帧和 L1/PSNR 跳变。
- pixel routing 必须满足“任意像素都有合法 expert 权重”；动态专家 coverage 缺失时回退到 Gaussian-level prior。
- 当前训练曲线显示模型有潜力：best fine L1=0.01197，但黑帧和 opacity reset 破坏了最终收敛统计。

### 仍需做什么
- 运行 renderer fallback 测试、全相关测试和 py_compile。
- 若测试通过，提交并推送修复。
- 服务器重新拉取后重新跑训练，重点验证 step 500/3000 不再出现 L1≈0.5 跳变，且 `usage_geo_*` 不再同时为 0。

### 运行过哪些测试
- `python -m pytest tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_falls_back_to_gaussian_prior_when_coverage_is_empty tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_ignores_background_for_inactive_experts tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_masks_uncovered_experts_and_aggregates_radii -v --tb=line --basetemp .pytest_tmp_local`：3 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_local`：71 passed。
- `python -m py_compile train.py render.py arguments/__init__.py gaussian_renderer/__init__.py models/tracking/cams_gs_tracking.py models/tracking/motion_decomposition.py`：passed。

### 下一步最小任务
- 提交并推送本轮 opacity reset 与 pixel routing fallback 修复。

## Repository

Working directory:
C:\Users\93895\Desktop\0427Moe研究\EndoMoeGaussian

Recovery files:
- .ai-recovery\branch.txt
- .ai-recovery\recent-commits.txt
- .ai-recovery\status.txt
- .ai-recovery\diffstat.txt
- .ai-recovery\changed-files.txt
- .ai-recovery\wip.patch
- .ai-recovery\untracked-files.txt
- .ai-recovery\project-file-snapshot.txt

## Main project goal

Implement and debug CAMS-GS / CAMS-GS-MoE pipeline for EndoNeRF-style dynamic surgical scene reconstruction.

## Known previous MoE plan

Plan name:
Split CAMS-GS into 3 Independent MoE Experts, Option B.

Motivation:
Original CAMS-GS fine-stage training collapsed:
- PSNR dropped from about 24.97 to 9.47.
- Motion field saturated early.
- Router collapsed.
- Visibility was not yet enabled, so failure happened in geometry stage.
- Root cause: curriculum-based 6-phase CAMS-GS design is incompatible with the MoE two-stage training style.

## MoE expert design

Use 3 independent self-contained CAMSGSTracking experts:

1. E_global
   - active_geo = 1
   - active_vis = 1
   - enable_visibility = False
   - trains: time_encoder, base_deformation, base_grid, motion_global

2. E_local
   - active_geo = 3
   - active_vis = 1
   - enable_visibility = False
   - trains: E_global modules + motion_local + cut_graph

3. E_full
   - active_geo = 3
   - active_vis = 2
   - enable_visibility = True
   - trains: E_local modules + visibility + appearance + lifecycle

## Intended implementation files from previous plan

Create:
- models/tracking/cams_gs_moe_tracking.py
- models/tracking/cams_gs_moe_scheduler.py
- arguments/endonerf/cutting_cams_gs_moe_expert_global.py
- arguments/endonerf/cutting_cams_gs_moe_expert_local.py
- arguments/endonerf/cutting_cams_gs_moe_expert_full.py
- arguments/endonerf/cutting_cams_gs_moe_router_only.py
- arguments/endonerf/cutting_cams_gs_moe_joint_finetune.py
- scripts/train_cams_gs_moe.sh

Modify:
- models/tracking/__init__.py
- scene/deformation.py
- scene/tracking_losses.py
- scene/gaussian_model.py

## Key implementation decisions

1. CAMSGSMoETracking should contain 3 independent CAMSGSTracking instances:
   - expert_global
   - expert_local
   - expert_full

2. Do not rewrite existing CAMSGSTracking internals unless strictly necessary.

3. Each expert uses a fixed TrackingPhase instead of the old curriculum scheduler.

4. Stage 1:
   - Train one expert at a time.
   - Use force_geo_expert to select the active expert.

5. Stage 2:
   - Load 3 expert checkpoints.
   - Freeze all expert parameters.
   - Train only the router.

6. Stage 3:
   - Optional joint finetune.

7. Renderer should reuse existing K-pass multi-expert rendering and weight splatting if already implemented.

8. Output format from CAMSGSMoETracking must be compatible with renderer, including tensors such as geo_expert_means3d with shape [N, 3, 3] or project-consistent equivalent.

## Current known task state from interrupted session

**Latest update (2026-06-02):**

### ✅ Completed Tasks

1. **Motion field initialization fix with active_geo masking**
   - Commit: fb12b44
   - Files: 4 (renderer, cams_gs_tracking, motion_decomposition, tests)
   - Changes: +298/-21
   - Tests: 50/50 passing
   - Design: Phase-aware active_geo masking in router and motion blending
   - Enables: MoE three-stage training (E_global → E_local → E_full)

2. **Motion magnitude regularization**
   - Commit: 0a5f77e
   - Files: 4 (motion_decomposition, cams_gs_tracking, tracking_losses, tests)
   - Changes: +60
   - Tests: 50/50 passing
   - Added: L_motion_mag loss with configurable weights
   - Purpose: Prevents motion field from growing too large during training

3. **Fix norm calculation timing (Critical)**
   - Commit: fbb200d
   - Files: 1 (motion_decomposition)
   - Changes: +4/-4 (moved 3 lines)
   - Tests: 50/50 passing
   - Fix: Compute norms AFTER masking instead of before
   - Impact: Loss now penalizes actual applied motion, not raw deltas
   - Result: When active_geo=1, local/cut_graph norms correctly equal 0

4. **Fix CAMS-GS fine-stage PSNR collapse root cause**
   - Files: 2 (scene/deformation.py, tests/test_disentangled_moe_tracking.py)
   - Tests: 51/51 passing
   - Root cause: CAMS-GS forward_dynamic first called _forward_original(), so fine stage activated random original EndoGaussian backbone before CAMS head.
   - Evidence: Tensorboard fine iter 1 had tiny CAMS deltas (d_mu_norm=0.000128, scale/rot/opacity near zero) but L1 jumped 0.0330→0.4356, proving CAMS deltas were not the destructive source.
   - Fix: For tracking_mode='cams_gs', use identity base (input xyz/scales/rotations/opacity) directly; do not pass through _forward_original before CAMS head.
   - Design decision: CAMS-GS is the deformation model, not a residual on top of random original EndoGaussian deformation.
   - Regression test: test_cams_gs_uses_identity_base_instead_of_original_backbone_deformation.

### 📊 Current Status
- Branch: main
- Working tree: has new identity-base fix pending commit
- All tests: passing (51/51)

### 🔍 Code Review Findings
✅ **Theoretical correctness**: CAMS-GS now starts dynamic training from static Gaussian identity state
✅ **Numerical stability**: Proper clamping (1e-8), fallback mechanism
✅ **Gradient flow**: Masking preserves gradients, torch.where differentiable
✅ **Critical fixes applied**: Norm timing corrected; original backbone bypassed for CAMS-GS fine stage

### 🎯 Next Tasks
- open: SSH to autodl_356 and update code
- open: Launch training on autodl_356

### 📝 Key Design Decisions
1. **Active_geo masking**: Dual-layer (router + motion) for end-to-end consistency
2. **Fallback**: Global-only mode when weights sum to 0
3. **Norm calculation**: AFTER masking to reflect actual applied motion (fixed)
4. **Loss weights**: lambda_motion_mag_global=1e-4, local/cut_graph=2e-5

### 🔄 Next Minimum Task
Ready for deployment: All fixes validated, tests passing, code reviewed.

Recent visible task state:
- 9 tasks total
- 7 done ✅
- 0 in progress
- 2 open

Older plan state:
- 49 tasks total
- 44 done
- 1 in progress: Implement CAMS expert proposal plumbing
- 4 open:
  - Reframe MoE story
  - Design reviewer-facing ablations
  - Implement renderer pixel routing
  - Update losses, presets, and tests

## Files already known to have local changes from git warnings

Likely changed:
- gaussian_renderer/__init__.py
- models/tracking/cams_gs_tracking.py
- models/tracking/motion_decomposition.py
- tests/test_disentangled_moe_tracking.py

Claude must verify with:
- .ai-recovery\changed-files.txt
- .ai-recovery\diffstat.txt
- .ai-recovery\wip.patch

## Recovery protocol for new Claude Code session

1. First read:
   - .ai-recovery\HANDOFF.md
   - .ai-recovery\status.txt
   - .ai-recovery\diffstat.txt
   - .ai-recovery\changed-files.txt
   - .ai-recovery\untracked-files.txt
   - .ai-recovery\wip.patch only if needed

2. Inspect only files related to the diff and known task list.
   Do not scan the whole repository.

3. Reconstruct the current implementation state from git diff.

4. Produce a short continuation plan.

5. Continue with the smallest safe next step:
   - finish motion field initialization fix
   - add motion magnitude regularization
   - update code on autodl_356
   - launch training

6. Keep outputs short.
   Do not paste long logs.
   For tests, summarize only failures and final status.

7. After every meaningful step, update .ai-recovery\HANDOFF.md.

## Safety constraints

- Preserve existing tracking_type='cams_gs' behavior.
- Avoid breaking original training path.
- Keep changes minimal and testable.
- Do not delete existing code unless clearly obsolete.
- Avoid broad project scans.
- Do not read old Claude transcript unless explicitly requested.

## Update 2026-06-03 route-gated local residual fix

### 已完成
- 复核 GitHub：`origin/main` 与本地 `HEAD` 都在 `2625c128ccd68b7471551c512ddecb6e814696a3`，上一轮 push 成功。
- 复核旧 TensorBoard：旧日志中 step 500/3000 跳变由 fine opacity reset 触发，step 3460 之后黑帧由 pixel routing 空覆盖触发；这两处上一轮修改位置正确。
- 新增本轮最小修复：`MotionDecomposition` 中 rotation/scale/opacity residual 改为受 `local_mix + cut_graph_mix` 门控；global-only 或 route 到 global 时不再修改局部外观几何参数。
- `geo_expert_scales/rotations/opacity_logits` 改为按专家语义输出：global expert 使用 canonical 参数，local/cut expert 使用 local residual 参数。
- 加强 `test_motion_decomposition_global_only_masks_non_global_mixes`，强制 local residual heads 非零并验证 global-only 不改变 scale/rotation/opacity。

### 当前设计决策
- `global` expert 只负责平滑平移场，不应承载局部 scale/rotation/opacity residual。
- `local/cut_graph` expert 才允许使用局部 scale/rotation/opacity residual；最终输出按 geometry route 的 local/cut 概率混合。
- pixel routing 的专家渲染应看到各自专家的参数，而不是所有专家共享同一份已混合 scale/rotation/opacity。

### 仍需做什么
- 服务器拉取新提交后，用全新 expname 重跑，验证 fine 阶段 `mean_norm_d_rot/mean_norm_d_scale` 不再在 global-only 阶段饱和到 ~0.17。
- 若仍出现 PSNR 6.x 跳变，下一步重点检查 fine 阶段 densify/prune 事件与跳变步是否一致。
- 建议拿新 run 的 TensorBoard 事件文件再次解析，不能再用 17:41 的旧事件判断本轮补丁效果。

### 运行过哪些测试
- `python -m pytest tests/test_disentangled_moe_tracking.py::test_motion_decomposition_global_only_masks_non_global_mixes tests/test_disentangled_moe_tracking.py::test_cams_gs_forward_emits_patch_c_aux_and_supports_tracking_losses -v --tb=line --basetemp .pytest_tmp_local`：2 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_validation_20260603181752`：71 passed。
- `python -m py_compile train.py render.py arguments/__init__.py gaussian_renderer/__init__.py models/tracking/cams_gs_tracking.py models/tracking/motion_decomposition.py`：passed。

### 下一步最小任务
- 提交并 push 本轮 route-gated residual 修复，然后让服务器用新 expname 重训，重点观察 step 500/3000/3460 附近是否还跳变。

## Update 2026-06-03 route-gated local residual push

### 已完成
- 已提交并推送 `306bc61 fix(model): gate local residuals by geometry route` 到 `origin/main`。
- `git ls-remote origin refs/heads/main` 确认为 `306bc616a8ecc758719a309bffb1e7477748b0ee`。

### 当前设计决策
- 服务器必须拉到 `306bc61` 或之后的提交，才能验证本轮 residual route-gating 是否消除 PSNR 6.x 间歇跳变。

### 仍需做什么
- 服务器使用全新 expname 重训，避免沿用旧输出目录里的旧 checkpoint/event/cfg。

### 运行过哪些测试
- 本轮 push 前已完成 71 个相关测试和 py_compile。

### 下一步最小任务
- 在服务器执行 `git log -1 --oneline` 确认 `306bc61`，再用新 expname 开始短跑验证。

## Update 2026-06-03 latest dynamic-loss TensorBoard analysis

### 已完成
- 解析最新事件文件 `events.out.tfevents.1780480829...75522.0`，确认 fine 只跑到 3452 step，`phase_visibility_enabled` 全程为 0，渲染时尚未进入 visibility/lifecycle 动态阶段。
- 日志显示 `usage_geo_global≈1`、`usage_geo_local=0`、`usage_geo_cut_graph=0`，local/cut 动态专家没有真正参与，动态自然退化为 global-only。
- 发现训练损失中的 geo balance 优先使用 `pixel_routing_weights`；当 pixel coverage/active expert 变为 0 时，balance loss 无法有效把 Gaussian router 拉回 local/cut。
- 修复 `scene/tracking_losses.py`：`L_balance_geo` 和 `usage_geo_*` 改用可微的 Gaussian-level `pi_geo.mean`；pixel routing usage 改为 `pixel_usage_geo_*` 诊断项。
- 修复 `_build_geo_target()`：`target_usage_geo_*` 只用于 CAMS-GS 的 `global/local/cut_graph` route，不污染旧 static/hexplane/smooth 目标。
- 新增测试：验证 pixel usage collapse 时，`L_balance_geo` 仍从 `pi_geo` 产生梯度。

### 当前设计决策
- Gaussian-level router prior 是训练 route balance 的主监督；pixel routing 是渲染空间细化和诊断，不应作为恢复 dead expert 的唯一训练信号。
- CAMS-GS 的 local/cut 动态必须先在 Gaussian route 中被拉起来，再让 pixel routing 做可见性/覆盖细化。
- 最新日志不是完整动态训练：只到 3452/9000，visibility 阶段 6300 后才开启，因此不能用该渲染判断最终动态效果。

### 仍需做什么
- 服务器拉取新提交后，用全新 expname 完整跑到至少 7000 step，再观察 `usage_geo_local/cut_graph` 是否非零、`phase_visibility_enabled` 是否变 1。
- 如果 route 仍塌到 global，下一步调整 `lambda_balance_geo/lambda_route_conf_geo` 或加入 local/cut forced warmup。

### 运行过哪些测试
- `python -m pytest tests/test_disentangled_moe_tracking.py::test_tracking_losses_use_covered_pixel_routing_weights_only tests/test_disentangled_moe_tracking.py::test_tracking_losses_balance_uses_gaussian_route_prior_when_pixel_usage_collapses -v --tb=line --basetemp .pytest_tmp_route_loss`：2 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_route_full_20260603193459`：72 passed。
- `python -m py_compile scene/tracking_losses.py train.py gaussian_renderer/__init__.py models/tracking/cams_gs_tracking.py models/tracking/motion_decomposition.py`：passed。

### 下一步最小任务
- 提交并 push route balance 使用 Gaussian prior 的修复，让服务器重跑新日志验证 local/cut route 是否开始参与。

## Update 2026-06-03 route balance push

### 已完成
- 已提交并推送 `0beffb8 fix(loss): balance geometry route from gaussian prior` 到 `origin/main`。
- 远端 `origin/main` 确认为 `0beffb88a7797a4780aa424fb5842bda568df74c`。

### 当前设计决策
- 后续服务器日志中 `usage_geo_*` 将表示 Gaussian-level route prior；`pixel_usage_geo_*` 才表示渲染 pixel routing usage。

### 仍需做什么
- 服务器必须拉取到 `0beffb8` 后用新 expname 完整训练；旧事件 `1780480829` 不能验证本轮修复。

### 运行过哪些测试
- push 前已完成 `72 passed` 和 py_compile。

### 下一步最小任务
- 新日志中优先看 `usage_geo_local/cut_graph` 是否从 0 抬起，再看 `pixel_usage_geo_*` 是否仍有空覆盖。

## Update 2026-06-03 pixel routing dynamic starvation analysis

### 已完成
- 解析最新事件 `events.out.tfevents.1780490978...105918.0`，确认训练完整跑到 fine 9000，且不再有 PSNR/L1 崩溃跳变。
- 日志显示 Gaussian-level route 已基本达到目标：末尾 `usage_geo_global/local/cut_graph≈0.455/0.083/0.462`，visibility 也在 6300 后开启并收敛到 `stable/transient≈0.853/0.147`。
- 关键异常：`pixel_usage_geo_global=1`、`pixel_usage_geo_local=0`、`pixel_usage_geo_cut_graph=0` 全程不变，说明最终渲染图像实际只使用 global expert。
- 理论定位：pixel routing 多专家渲染绕开了 `MotionDecomposition` 已混合的 Gaussian-level MoE 动态状态，导致 photometric loss 看不到 local/cut 专家；local/cut 只能被 balance 拉权重，不能被图像监督学到有效动态。
- 修复 `gaussian_renderer/__init__.py`：新增 `_use_pixel_routing()`，真实训练默认读取 `args.use_pixel_routing=False`，因此默认走 Gaussian-level MoE 混合后的 `means/scales/rotations/opacity` 单次渲染。
- 新增 `arguments/__init__.py` 参数 `use_pixel_routing=False`，保留显式打开 pixel routing 的实验能力。
- 新增 renderer 回归测试：关闭 pixel routing 时，即使 aux 提供 expert proposals，也只调用一次 rasterizer 并使用混合后的动态状态。

### 当前设计决策
- EndoMoe/CAMS-GS 的主训练路径应先让 Gaussian-level MoE 动态被 photometric loss 直接监督。
- Pixel routing 可作为后续显式消融/高级 per-pixel 组合模块，但不能默认介入早期或主线训练，否则会造成 expert starvation。
- 当前事件里场景静态感的直接原因不是时间没传，而是最终渲染权重在 pixel 层退化为 global-only。

### 仍需做什么
- 服务器拉取新提交后用全新 expname 重训；新日志中默认应不再出现 `pixel_usage_geo_*`，因为 pixel routing 关闭。
- 若 Gaussian-level 混合渲染后仍动态不足，下一步再调 global/local/cut 专家的容量与阶段，例如降低 global 阶段时长或增加 local/cut forced warmup。

### 运行过哪些测试
- `python -m pytest tests/test_disentangled_moe_tracking.py::test_renderer_defaults_to_gaussian_moe_blend_when_pixel_routing_disabled tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_preserves_expert_appearance_and_opacity_controls tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_masks_uncovered_experts_and_aggregates_radii tests/test_disentangled_moe_tracking.py::test_renderer_pixel_routing_falls_back_to_gaussian_prior_when_coverage_is_empty tests/test_endonerf_presets.py::test_endonerf_presets_only_use_known_parser_keys -v --tb=line --basetemp .pytest_tmp_pixel_switch`：5 passed。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -v --tb=line --basetemp .pytest_tmp_pixel_full_20260603213504`：73 passed。
- `python -m py_compile arguments/__init__.py gaussian_renderer/__init__.py scene/tracking_losses.py train.py render.py models/tracking/cams_gs_tracking.py models/tracking/motion_decomposition.py`：passed。

### 下一步最小任务
- 提交并 push 默认关闭 pixel routing 的修复，让服务器重跑验证 local/cut 动态是否进入实际 rendered image。

## Update 2026-06-03 pixel routing default-off push

### 已完成
- 已提交并推送 `4912a1b fix(renderer): train with gaussian moe blend by default` 到 `origin/main`。
- 远端 `origin/main` 确认为 `4912a1b9e3351b2c8c667a139b86eb171db2c8fc`。

### 当前设计决策
- 默认训练/渲染使用 Gaussian-level MoE blended state；只有显式设置 `use_pixel_routing=True` 时才启用 pixel-level expert compositing。

### 仍需做什么
- 服务器拉取 `4912a1b` 后全新训练，验证 `pixel_usage_geo_*` 不再出现，且 local/cut 的实际 rendered dynamic 开始影响图像。

### 运行过哪些测试
- push 前已完成 `73 passed` 和 py_compile。

### 下一步最小任务
- 新 run 若 3000 左右 PSNR 仍低于 EndoGaussian，继续检查 global expert 过强与 local/cut photometric 梯度大小。

## Update 2026-06-10 latest 15000-step TensorBoard architecture audit

### 已完成
- 完整解析 `output/last` 的 52 个 scalar；fine 阶段共 15000 step，无 NaN/Inf。
- 确认当前 run 使用 `tracking_type='cams_gs'` 的六阶段 curriculum，并不是 `cams_gs_moe` 独立专家路径。
- 分阶段 L1 均值为：global `0.03321`、graph `0.02964`、local `0.02369`、motion `0.01901`、visibility `0.01774`、joint `0.01725`。
- 对比 9000-step run：最后 1000 step L1 从 `0.01814` 改善到 `0.01716`，但 total loss 从 `0.05046` 上升到 `0.06809`；local+cut motion magnitude 从约 `0.223` 增长到约 `0.495`。
- 定位到 curriculum、梯度连接和容量设计中的多处根因。

### 当前设计决策
- 单纯继续增加 iterations 不足以追平 EndoGaussian；应先修复阶段和梯度设计。
- `graph_bootstrap` 在 `active_geo=1` 时训练 `tracking_cut_graph`，但输出被 global mask 完全覆盖，属于无效阶段。
- 15000 iterations 会按比例把 visibility 推迟到 10500、lifecycle 推迟到 12750；同时 LR schedule 仍在 9000 step 达到最低值，晚启用模块从极低 LR 开始训练。
- `cams_gs` 为避免动态启动崩溃而绕过 HexPlane backbone，损失了 EndoGaussian 的主要时空表达能力。
- 默认 `no_do=True`、`use_pixel_routing=False` 时 visibility/lifecycle 基本不参与最终图像形成；appearance residual 未被 visibility route 正确门控。
- `L_geo_temp` 因单帧训练未产生，`L_geo_spatial` 在本 run 中也未出现；动态时空正则实际上没有工作。
- `cams_gs_moe` 仍存在专家同构/嵌套、阶段切片训练而非独立预训练、`full` expert target 配置名仍写成 `cut_graph` 等设计问题，不能直接作为最终 MoE-GS 实现。

### 仍需做什么
- 先补齐固定验证视角 PSNR、depth loss、TV loss、各参数组 LR/grad norm、点数和 per-expert 指标日志。
- 修正阶段为绝对 step，并为晚启用模块使用 stage-local LR schedule。
- 删除或重写无效 `graph_bootstrap`，修复 temporal/spatial regularization。
- 将 HexPlane baseline 恢复为 identity-safe shared/base expert，再在其上增加异构 residual experts。
- 重新设计 visibility/lifecycle，使其真正通过颜色/opacity 参与 photometric loss，并采用 identity-safe 初始化。
- 修正 `cams_gs_moe` 的 `target_usage_geo_full`、独立专家训练和 pixel-space router。

### 运行过哪些测试
- 未修改代码，因此未运行单元测试。
- 使用 TensorBoard LegacyEventFileLoader 完整读取 `output/last`，并按六个阶段统计趋势、极值、跳变和相关性。
- 与 `events.out.tfevents.1780493919...120162.0` 做同口径 9000/15000-step 对比。
- 独立 Codex 审核调用两次均在 120 秒超时，未返回可用结论。

### 下一步最小任务
- 先只修改训练日志：恢复固定验证 PSNR，并记录 depth/TV、LR、grad norm 和 per-expert motion；运行最小日志回归测试后再改模型结构。

## Update 2026-06-10 engineering redesign plan

### 已完成
- 新增 `plan/task_plan.md`、`plan/notes.md`、`plan/endomoegaussian_engineering_plan.md`。
- 明确最终主线为 identity-safe HexPlane shared base、三个异构 residual experts、Gaussian prior 与 pixel router 双层路由。
- 明确采用独立专家训练、router-only 训练和低学习率 joint finetune，而不是单次 curriculum 中切片训练同构专家。

### 当前设计决策
- `cams_gs` 仅保留为诊断基线，最终方法在 `cams_gs_moe` 下重构。
- 固定 usage target 只作为早期先验，最终路由必须由 image-space photometric objective 决定。
- 所有晚启用模块使用绝对阶段边界和 group-local LR age。
- visibility/lifecycle 必须真实改变 rendered RGB/alpha，并使用 identity-safe 初始化。

### 仍需做什么
- 实现绝对阶段与 group-local LR。
- 恢复 identity-safe HexPlane shared base。
- 重构异构专家、独立 checkpoint 流程和 pixel router。
- 补齐 temporal/spatial 正则与完整诊断。

### 运行过哪些测试
- 本步仅新增设计文档，未运行代码测试。

### 下一步最小任务
- 修改 `TrackingPhase`、`EndoMoEGaussianScheduler` 和 `GaussianModel.update_learning_rate()`，建立 group-local LR 契约。

## Update 2026-06-10 absolute stages and group-local LR foundation

### 已完成
- `TrackingPhase` 新增按参数组前缀查询的 `group_schedule_progress` 契约。
- `EndoMoEGaussianScheduler` 的默认阶段改为绝对 step，不再随总 iterations 按比例漂移。
- local/full 专家阶段冻结共享 time encoder，避免后训练专家破坏已训练专家的输入表征。
- `GaussianModel.update_learning_rate()` 支持按参数组的 stage-local progress 计算 deformation/grid LR。

### 当前设计决策
- global expert 阶段训练共享 time encoder；后续专家和 router 使用冻结的共享编码，直到独立专家编码器重构完成。
- joint finetune 不再解冻共享 time encoder，专家 residual 使用 0.1 LR scale。
- 每个新启用专家从本阶段 LR 起点开始，而不是继承全局 iteration 已衰减到最低值的 LR。

### 仍需做什么
- 更新 15000-step EndoMoe preset 的绝对阶段边界和 LR horizon。
- 增加 scheduler 与 group-local LR 回归测试。
- 后续替换共享 time encoder 为专家内部独立编码器。

### 运行过哪些测试
- 本步修改后尚未运行测试。

### 下一步最小任务
- 更新 EndoMoe presets 和测试，验证阶段边界、冻结契约及局部 LR 数值。

## Update 2026-06-10 EndoMoe 15000-step preset correction

### 已完成
- cutting/pulling EndoMoe presets 更新为 15000 fine iterations，LR horizon 同步为 15000。
- 专家阶段固定为 global `0-2000`、local `2000-5000`、full `5000-8000`、router `8000-12000`、joint `12000-15000`。
- 新增 `target_usage_geo_full` 参数，并将 EndoMoe 第三专家 target 从错误的 `cut_graph` 名称改为 `full`。

### 当前设计决策
- `cams_gs` 继续使用 `target_usage_geo_cut_graph`；`cams_gs_moe` 使用 `target_usage_geo_full`，两套专家语义不再混用。
- 15000-step 配置必须同步扩展 position/deformation/grid LR horizon。

### 仍需做什么
- 更新 preset/scheduler 单测。
- 验证 `full` target 实际构造为 `0.35/0.35/0.30`。
- 验证每个专家阶段的 LR 从初始值重新按本阶段进度衰减。

### 运行过哪些测试
- 本步尚未运行测试。

### 下一步最小任务
- 修改相关测试并运行 scheduler、preset、py_compile 最小验证。

## Update 2026-06-10 scheduler and local-LR verification

### 已完成
- 更新 scheduler 单测，确认共享 time encoder 只在 global expert 阶段训练。
- 增加默认阶段不会随 15000/30000 iterations 拉伸的回归测试。
- 增加 `GaussianModel.update_learning_rate()` 使用 group-local progress 的数值测试。
- 更新 EndoMoe preset 测试，确认 15000-step 边界与 `global/local/full=0.35/0.35/0.30` target。

### 当前设计决策
- Phase 3 基础设施完成，进入 identity-safe shared HexPlane base 实现。

### 仍需做什么
- 让 `cams_gs_moe` 恢复 EndoGaussian HexPlane base，同时确保 fine step 1 严格 identity。
- 为 shared base 使用独立参数组，只在 base 阶段训练。

### 运行过哪些测试
- scheduler/preset/local-LR 定向测试：`5 passed`。
- 相关 Python 文件 `py_compile`：passed。

### 下一步最小任务
- 修改 `scene/deformation.py` 和 EndoMoe scheduler，使共享 HexPlane base identity-safe 且可冻结。

## Update 2026-06-10 identity-safe shared HexPlane base

### 已完成
- `cams_gs_moe` 重新启用 EndoGaussian HexPlane deformation backbone。
- shared base 的 position/scale/rotation/opacity 输出层在模型初始化后严格置零，fine step 1 保持 canonical identity。
- `cams_gs_moe` forward 先计算 shared HexPlane base，再将其作为三个 residual experts 的输入。
- shared base 使用独立 optimizer group：`tracking_shared_base_deformation/grid`。
- EndoMoe global 阶段训练 shared base；后续 expert/router 阶段可冻结 shared base。

### 当前设计决策
- `cams_gs` 旧诊断路径继续使用 identity base，不改变其历史行为。
- 只有最终 `cams_gs_moe` 主线恢复 HexPlane 容量，并通过零输出初始化解决动态启动崩溃。
- shared base 不是第四个 router expert，而是所有专家共享的低频动态基座。

### 仍需做什么
- 将现有三个嵌套 CAMSGSTracking copies 替换为 global smooth、tissue local、tool/contact 三个真正异构 residual experts。
- 为 shared base 增加独立 counterfactual 验证和梯度日志。

### 运行过哪些测试
- identity shared-base、旧 cams identity、scheduler 与模型构造定向测试：`4 passed`。
- 修改文件 `py_compile`：passed。

### 下一步最小任务
- 定义统一 expert proposal 接口，并先实现 global smooth 与 tissue local 两个异构专家。

## Update 2026-06-10 foundation full verification

### 已完成
- 对本轮 absolute stages、group-local LR、EndoMoe preset 和 identity-safe shared HexPlane base 运行完整相关回归。

### 当前设计决策
- 当前基础层可继续进入异构专家重构，不需要回退已有修改。

### 仍需做什么
- Phase 5：替换 nested homogeneous experts。
- Phase 6：实现 Gaussian prior + pixel-space router。
- Phase 7：修复 visibility/lifecycle 与 temporal/spatial supervision。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -q --tb=line --basetemp .pytest_tmp_endomoeg_foundation_full`：`76 passed`。
- `git diff --check`：passed，仅有 Windows LF/CRLF 提示。

### 下一步最小任务
- 新增异构 expert proposal 模块，并接入 `CAMSGSMoETracking`，保持现有 aux/render 契约。

## Update 2026-06-10 heterogeneous expert proposal implementation

### 已完成
- 新增 `models/tracking/endomoeg_experts.py`，定义统一完整 Gaussian proposal contract。
- 实现 `GlobalSmoothExpert`：独立时间编码、低频全局位移、无局部外观自由度。
- 实现 `TissueLocalExpert`：独立时间编码、空间条件局部位移及 bounded rotation/scale。
- 实现 `ToolContactExpert`：空间、opacity、view direction、camera depth、screen projection 条件，支持 motion/rotation/scale/opacity/appearance/visibility/lifecycle。
- `CAMSGSMoETracking` 不再嵌套三套带内部 router 的 CAMSGSTracking，而是直接组合三个异构完整 proposals。

### 当前设计决策
- 每个专家内部有独立时间编码器，避免共享 encoder 漂移破坏已训练专家。
- 所有几何和 appearance residual 输出头严格零初始化。
- global/local 专家保持 visibility/lifecycle identity；只有 tool/contact 专家承担遮挡与外观变化。
- 保留旧 `cut_graph_motion` aux 键用于兼容，同时新增语义正确的 `full_motion`。

### 仍需做什么
- 校正 tool/contact opacity gate 的 identity 精度并增加单测。
- 更新 exports、expert 类型测试、参数组与 forced-route 梯度测试。
- 运行完整相关测试后再实现独立专家 checkpoint workflow。

### 运行过哪些测试
- 本步尚未运行测试。

### 下一步最小任务
- 修正 tool/contact alpha-to-opacity 映射并补异构专家 identity/差异化/梯度回归测试。

## Update 2026-06-10 heterogeneous expert safety contracts

### 已完成
- tool/contact transient 与 lifecycle 初始化改为近严格 identity：transient bias `-12`、persistent/transient lifecycle bias `+12/-12`。
- opacity 改为在 alpha 概率域乘 visibility/lifecycle gate，再映射回 logit，避免错误地把 `logit(gate)` 直接加到 opacity。
- 导出三个异构 expert 类型。
- 增加 expert 类型、参数独立性、identity proposal 和 forced-local 梯度隔离测试。

### 当前设计决策
- forced expert 预训练时，只有被选择专家接收 photometric gradient；router 和其余专家必须零梯度。
- tool/contact 的 appearance residual 始终由 transient probability 门控。
- visibility/lifecycle 从近 identity 开始，但通过完整 opacity proposal 接入 photometric loss。

### 仍需做什么
- 运行异构专家定向测试和完整相关回归。
- 若通过，继续实现独立专家 checkpoint/stage workflow，并移除共享 outer time encoder 对 MoE 专家的依赖。

### 运行过哪些测试
- 本步尚未运行测试。

### 下一步最小任务
- 运行 expert identity、forced gradient、proposal shape、scheduler 和 py_compile 验证。

## Update 2026-06-10 heterogeneous experts full verification

### 已完成
- 修复 tool/contact identity gate 的初始 opacity 微扰，使用相对初始 gate 校准。
- 异构 expert proposal、forced gradient、shared base、旧 CAMS/renderer/loss 全部回归通过。

### 当前设计决策
- 三个专家已经不再共享内部时间编码；outer `tracking_time_encoder` 现在仅应服务 Gaussian router。
- 因此下一步需调整 scheduler：expert 阶段冻结 outer time encoder，router-only 阶段训练 outer time encoder + router。

### 仍需做什么
- 修正 outer time encoder 的阶段归属。
- 实现 expert/shared-base 独立组件 checkpoint 保存与恢复。
- 完成后进入 Gaussian prior/pixel router 重构。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py ...`：`78 passed`。
- `git diff --check`：passed，仅有 LF/CRLF 提示。

### 下一步最小任务
- 调整 EndoMoe scheduler 的 router time encoder 归属并增加回归测试。

## Update 2026-06-10 independent component checkpoint foundation

### 已完成
- outer `tracking_time_encoder` 改为只在 router-only 阶段训练，joint 阶段以 0.1 LR 微调；expert 阶段不再训练。
- `CAMSGSMoETracking` 新增 global/local/full/router 独立 state 导出与加载。
- `deform_network` 新增 shared_base、三个 experts、router+time_encoder 的带架构元数据组件 checkpoint API。
- fine 训练阶段切换时自动保存已完成组件到 `output/<exp>/endomoeg_components/*.pth`。
- 显式单阶段训练结束时同样保存对应组件。

### 当前设计决策
- shared base 与 global expert 在 global 阶段结束时分别保存。
- local/full 专家各自独立保存；router checkpoint 必须同时包含 outer time encoder。
- 组件 checkpoint 必须验证 `tracking_type` 和 `tracking_arch_version`，禁止静默加载旧架构。

### 仍需做什么
- 增加组件 round-trip、错误架构拒绝和阶段切换保存测试。
- 增加从指定组件目录装配 router/joint 作业的加载流程。
- 完成后进入 Phase 6 router 重构。

### 运行过哪些测试
- router time encoder 阶段归属定向测试：`2 passed`。
- checkpoint API 新增后尚未运行测试。

### 下一步最小任务
- 增加 component checkpoint round-trip 与 phase-save mapping 测试并运行完整回归。

## Update 2026-06-10 independent component assembly loading

### 已完成
- `arguments/__init__.py` 新增 `endomoeg_component_dir` 与 `endomoeg_strict_component_loading`，允许显式指定独立组件目录和缺失组件策略。
- `deform_network` 新增目录级组件装载接口，可按名称加载 `shared_base/global/local/full/router` checkpoint。
- `train.py` 在 coarse checkpoint 恢复后、fine optimizer 建立前装配请求的 EndoMoe 组件。
- 显式单阶段作业建立依赖约束：local/full 依赖 shared base；router 依赖 shared base 与三个专家；joint 额外依赖 router。

### 当前设计决策
- 独立专家训练不串行继承其他专家，只继承同一 shared base；这样避免专家间表征漂移和训练顺序耦合。
- router-only 必须从冻结的完整专家集合开始，joint 必须从完整专家集合与已训练 router 开始。
- 连续训练模式不要求组件目录；只有显式非 global 单阶段作业缺少目录时才立即报错。

### 仍需做什么
- 增加严格/非严格目录加载、参数解析与显式阶段依赖映射回归测试。
- 验证组件装载发生在 fine optimizer 创建前，且不被 coarse restore 覆盖。
- 测试通过后结束 Phase 5，进入 Gaussian prior 与 pixel-space router 重构。

### 运行过哪些测试
- 本步三个文件修改后尚未运行测试。

### 下一步最小任务
- 补充组件目录加载测试，并运行定向 pytest、py_compile 与相关完整回归。

## Update 2026-06-10 validated EndoMoe stage dependencies

### 已完成
- 将 EndoMoe 阶段别名、规范名称与所需组件依赖集中到 `cams_gs_moe_tracking.py`，scheduler 与训练装配共享同一份契约。
- 未知显式阶段不再静默进入 joint，而是立即抛出 `ValueError`。
- 增加严格/非严格组件目录加载测试，以及 global/local/full/router/joint 依赖映射测试。

### 当前设计决策
- 阶段规范化属于模型训练协议，不应在 `train.py` 重复维护。
- 非严格加载仅用于诊断或消融；正式独立阶段训练默认严格检查全部依赖。
- router 与 joint 的依赖必须显式、可测试，避免缺失专家后仍开始训练造成不可解释退化。

### 仍需做什么
- 增加 parser 默认值与配置覆盖测试。
- 运行新增定向测试；若通过，再运行两个相关测试文件的完整回归。
- 更新 Phase 5 状态并开始 Gaussian prior router 重构。

### 运行过哪些测试
- `python -c "import train"` 在本机依赖初始化阶段 30 秒超时，因此不采用直接导入整个训练入口的测试方式。
- 新增测试尚未执行。

### 下一步最小任务
- 补 parser 测试后运行组件加载与阶段规范化定向 pytest。

## Update 2026-06-10 Phase 5 completion verification

### 已完成
- 增加 `endomoeg_component_dir` 与严格加载默认值的 parser 回归测试。
- 完成独立 shared base、global、local、full、router 保存、恢复、目录装配与架构拒绝测试。
- 完成阶段别名和组件依赖测试，未知阶段能够快速失败。
- `plan/task_plan.md` 将异构 residual experts 与独立 checkpoint workflow 标记为完成，正式进入 Phase 6。

### 当前设计决策
- Phase 5 的工程完成标准不仅是专家结构存在，还包括独立训练、独立保存、确定性装配与错误依赖拒绝。
- router 重构必须建立在冻结且可复现装配的三专家集合上，不能继续边训练专家边猜路由。

### 仍需做什么
- 重构 Gaussian prior router 的输入、soft top-2、容量/置信度调度与退化保护。
- 在 Gaussian prior 稳定后实现 pixel-space router，确保 photometric gradient 可达每个被覆盖专家。
- 后续修复 visibility/lifecycle 与 temporal/spatial supervision。

### 运行过哪些测试
- 组件加载与阶段契约定向测试：`5 passed`。
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -q ...`：`83 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 审计当前 `VolumeAwareGaussianRouter` 的输入和混合路径，先实现可测试的 soft top-2 Gaussian prior。

## Update 2026-06-10 Gaussian-prior semantic repair

### 已完成
- Gaussian router 输入从“变形后坐标 + 原始 opacity logit + residual norm”改为 canonical 坐标、shared-base motion、opacity probability、各专家 residual magnitude 与时间特征。
- shared-base motion 与 expert residual magnitude 均按 scene scale 归一化，避免不同场景尺度破坏路由标定。
- router-only 阶段增加高温 dense warm-up，达到配置比例后切换 soft top-2；joint 阶段固定使用 soft top-2。
- `CAMSGSMoETracking` 明确区分 total motion、shared-base motion 与 residual motion；`d_mu` 现在表示 canonical 到最终状态的完整位移。
- EndoMoe checkpoint 架构版本提升为 `endomoeg_v2`，拒绝误加载旧 router 输入维度。

### 当前设计决策
- Gaussian prior 必须观察 baseline 已解释的运动，否则无法学习“哪个 residual expert 在 shared base 之上仍有优势”。
- opacity 采用概率而非未界定 logit 作为 router 特征，motion 采用 scene-scale normalized 表示。
- soft top-2 只在 dense warm-up 后启用，避免随机初始化时永久饿死第三专家。
- `means3d_canonical` 与 `d_mu` 的语义必须覆盖完整动态链，temporal/spatial regularization 才不会漏掉 HexPlane 主运动。

### 仍需做什么
- 增加 router dense-to-top2 调度、稀疏归一化、scene-scale invariance 与 full-motion 语义测试。
- 验证旧 CAMS 路径和异构专家 checkpoint 回归不受影响。
- 测试通过后继续加入 router capacity/advantage diagnostics，再进入 pixel-space router。

### 运行过哪些测试
- 本步三个代码文件修改后尚未运行测试。

### 下一步最小任务
- 补充 Gaussian-prior 数值与 scheduler 回归测试，并运行定向 pytest。

## Update 2026-06-10 Gaussian-prior foundation verification

### 已完成
- 增加 dense warm-up 到 soft top-2 的阶段边界测试。
- 增加 top-2 权重稀疏性与归一化测试。
- 增加 motion feature 对 scene scale 的不变性测试。
- 增加 shared-base deformation 被计入完整 `d_mu` 的语义测试。
- 更新架构版本回归为 `endomoeg_v2`。

### 当前设计决策
- 当前 soft top-2 是对 dense softmax 权重做 top-k 选择后重新归一化，保留两个专家的连续混合权重。
- dense prior 继续保存在 `gaussian_pi_geo_dense`，用于监控稀疏化前是否已经发生专家塌缩。
- raw router logits 单独输出，避免用 `log(sparse_weight)` 伪造 logits。

### 仍需做什么
- 将固定 usage target 从永久约束改为 router warm-up prior，并逐步衰减。
- 将 route confidence 从早期强制决策改为后期增强，避免未学会前过早尖锐化。
- 增加 dense usage、sparse usage、零路由率与专家有效样本量日志。

### 运行过哪些测试
- Gaussian-prior 定向测试：`5 passed`。
- 首次完整回归仅因版本预期仍为 v1 失败；修正测试后完整相关回归：`87 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 实现 router balance/confidence 的阶段化权重，并增加损失数值测试。

## Update 2026-06-10 staged router regularization contract

### 已完成
- `TrackingPhase` 新增独立的 `route_balance_scale` 与 `route_confidence_scale`。
- router-only 阶段的 usage balance 从 1.0 线性衰减到可配置弱先验，避免固定比例永久压制 image-space objective。
- route confidence 在 dense warm-up 与 sparse 切换前保持为 0，之后才逐步增强。
- joint 阶段仅保留可配置的极弱 balance prior，同时完整启用 confidence。

### 当前设计决策
- balance 的职责是防止 early dead expert，不是规定最终每个场景都必须满足固定 35/35/30。
- confidence 的职责是后期减少模糊混合；在专家优势尚未形成时施加会制造随机塌缩。
- 默认 router 末期 balance scale 为 0.10，joint 为 0.05；实际 loss 还需在 `tracking_losses.py` 接入该缩放。

### 仍需做什么
- 在 `compute_tracking_losses()` 中应用两个 phase scale，并记录实际 scale。
- 增加损失数值测试，确认 early confidence 为零、late confidence 增强、joint balance 只剩弱先验。
- 完整回归后补 dense/sparse route degeneration diagnostics。

### 运行过哪些测试
- 本步三个文件修改后尚未运行测试。

### 下一步最小任务
- 修改 `scene/tracking_losses.py` 并补 router regularization 数值测试。

## Update 2026-06-10 staged router losses connected

### 已完成
- `compute_tracking_losses()` 现在将 `route_balance_scale` 乘到 usage balance loss。
- `route_confidence_scale` 现在乘到 confidence sharpening loss。
- 两个实际 scale 都进入训练日志，便于核对阶段调度是否真正生效。
- 增加 scheduler scale 趋势与 loss 精确数值测试。

### 当前设计决策
- 未提供 phase scale 的旧模型路径默认仍使用 1.0，避免改变 `cams_gs` 与 heterogeneous baseline 的历史行为。
- scale 被限制在 `[0, 1]`，防止错误配置放大 regularization。
- router warm-up 早期 confidence loss 应严格为零，usage balance 随阶段逐步降权。

### 仍需做什么
- 运行 staged-loss 定向测试和完整相关回归。
- 增加 dense/sparse usage、zero fraction 与 effective expert count 诊断。
- 诊断通过后评估 Gaussian prior Phase 是否完成。

### 运行过哪些测试
- 本步两个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 router scheduler/loss 定向测试与 py_compile。

## Update 2026-06-10 staged router loss verification

### 已完成
- 验证 router-only 早期 confidence scale 为 0，后期逐步增加。
- 验证 balance scale 随 router 训练衰减，joint 阶段固定为 0.05 弱先验。
- 验证 `L_balance_geo` 与 `L_route_conf_geo` 精确乘入阶段 scale。

### 当前设计决策
- EndoMoe preset 的 35/35/30 仅是 early anti-collapse prior；最终专家占比允许由数据决定。
- joint finetune 中 balance 不完全删除，而是保留 5% 权重作为 dead-expert 防线。

### 仍需做什么
- 增加 dense/sparse route degeneration diagnostics。
- 检查 TensorBoard 汇总是否自动记录新增指标。
- 完成 Gaussian prior 后进入 pixel-space router。

### 运行过哪些测试
- staged router scheduler/loss 定向测试：`3 passed`。
- 完整相关回归：`88 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 增加 dense usage、sparse zero fraction、effective expert count 与 route disagreement 指标。

## Update 2026-06-10 router degeneration diagnostics

### 已完成
- tracking loss metrics 新增每专家 dense usage 与 sparse route coverage。
- 新增 sparse zero fraction、sparse/dense effective expert count 和 sparse-dense L1 差异。
- 增加构造型测试，可区分“dense router 已塌缩”和“仅 top-2 稀疏化导致零权重”。

### 当前设计决策
- `usage_geo_*` 表示实际参与 Gaussian mixture 的 sparse 权重。
- `dense_usage_geo_*` 表示 top-k 之前的 router 意图。
- `route_coverage_geo_*` 表示每个专家获得非零 photometric path 的 Gaussian 比例，是判断 expert starvation 的核心指标。
- effective expert count 使用 inverse Simpson index，比单独 entropy 更直观。

### 仍需做什么
- 运行诊断定向测试和完整回归。
- 更新调试打印列表，使服务器控制台也能快速看到 coverage/effective count。
- 若验证通过，Gaussian prior 基础 Phase 可收口，下一步进入 pixel router。

### 运行过哪些测试
- 本步两个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 router diagnostics 定向测试与完整相关回归。

## Update 2026-06-10 Gaussian-prior phase completion

### 已完成
- dense/sparse starvation diagnostics 定向测试通过。
- 控制台 500-step debug 摘要新增 zero fraction、effective expert count、balance/confidence scale，以及每专家 dense usage/coverage。
- Gaussian prior 的 canonical/shared-base/residual 语义、scene-scale normalization、dense warm-up、soft top-2、阶段化 regularization 与退化诊断已形成完整闭环。

### 当前设计决策
- Gaussian prior 阶段可以结束；后续不再通过固定 usage target 代替 image-space router 学习。
- pixel router 必须使用专家各自渲染得到的 RGB/depth/alpha/coverage，并以 Gaussian prior 作为 log-prior，而不是重新绕开已训练专家。

### 仍需做什么
- 审计当前 renderer 的 pixel routing 输入和 fallback 行为。
- 实现可训练 PixelSpaceRouter，并确保未覆盖像素、黑帧与专家 starvation 有确定性保护。
- 之后再将 visibility/lifecycle 接入最终 alpha/RGB。

### 运行过哪些测试
- router diagnostics 定向测试：`3 passed`。
- 完整相关回归：`89 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 只审计 `gaussian_renderer/__init__.py` 的 pixel expert compositing，定义 PixelSpaceRouter 输入输出契约。

## Update 2026-06-10 identity-safe PixelSpaceRouter module

### 已完成
- 新增共享卷积 `PixelSpaceRouter`，输入每专家 RGB、相对 RGB、相对 depth、Gaussian prior、projected motion 与 coverage。
- 最后一层严格零初始化，因此初始 pixel weights 精确退化为归一化 Gaussian prior，不会在启用瞬间破坏已有渲染。
- 未覆盖像素使用全局 Gaussian fallback prior；无 prior 或无 coverage 的专家被确定性 mask。
- router checkpoint 现在同时包含 Gaussian router 与 pixel router；架构版本提升为 `endomoeg_v3`。
- optimizer 参数组拆为 `tracking_moe_router_gaussian` 与 `tracking_moe_router_pixel`，仍由 `tracking_moe_router` 前缀统一调度。

### 当前设计决策
- Pixel router 学习的是 Gaussian prior 上的 residual logits，而不是从零重学路由。
- 使用所有专家共享的 score network，避免专家特定 head 通过固定通道身份形成无意义偏置。
- relative RGB/depth 表达专家间局部差异，projected motion 表达动态证据，coverage 负责可见性约束。

### 仍需做什么
- renderer 需要在一次辅助 rasterization 中生成 prior、projected motion 与 coverage maps。
- 接入 `route_endomoeg_pixels()`，并保留旧 fake deformation 的兼容 fallback。
- 增加 identity、fallback、梯度与黑帧防护测试。

### 运行过哪些测试
- 本步两个文件修改后尚未运行测试。

### 下一步最小任务
- 修改 renderer 的 pixel compositing，调用可训练 PixelSpaceRouter。

## Update 2026-06-10 trainable pixel routing integration

### 已完成
- renderer 的辅助 rasterization 改为一次输出 Gaussian prior signal、projected residual motion 与 dynamic coverage。
- 真实 EndoMoe deformation 通过 `route_endomoeg_pixels()` 调用可训练 PixelSpaceRouter。
- 旧 fake deformation 与其他路径保留原手工 compositing fallback，避免破坏诊断测试。
- deformation aux 新增 pixel prior、projected motion、coverage 与 residual logits。
- 导出 `PixelSpaceRouter`，增加 prior identity、无覆盖 fallback 和 photometric gradient 测试。

### 当前设计决策
- dynamic coverage 只由 deformation points 贡献，静态公共高斯不会伪造所有专家均覆盖。
- projected motion 与 Gaussian prior 在同一次辅助 rasterization 中生成，避免再增加一次 rasterizer 调用。
- Pixel router 最后一层零初始化，启用 pixel routing 时第一步仍等价于 Gaussian prior compositing。

### 仍需做什么
- 运行 PixelSpaceRouter 定向测试和 renderer 相关回归。
- 将 EndoMoe presets 显式打开 `use_pixel_routing=True`，并补 parser/preset 测试。
- 增加 renderer 调用 trainable route 的集成测试及黑帧保护断言。

### 运行过哪些测试
- 本步三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 PixelSpaceRouter identity/fallback/gradient 与现有 renderer pixel-routing 回归。

## Update 2026-06-10 EndoMoe pixel-routing preset activation

### 已完成
- PixelSpaceRouter identity、fallback、gradient、checkpoint 与旧 renderer 回归共 8 项通过。
- parser 新增 `moe_pixel_router_hidden_dim`。
- cutting/pulling EndoMoe presets 显式设置 `use_pixel_routing=True` 与 pixel router hidden dim 32。

### 当前设计决策
- 仅 EndoMoe 主线默认启用 trainable pixel routing；`cams_gs` 诊断 baseline 继续保持默认关闭。
- expert forced-route 阶段即使 pixel routing 开启，也因 Gaussian prior 为 one-hot 而保持单专家训练语义。

### 仍需做什么
- 更新 preset 回归断言。
- 增加 renderer 确实调用 trainable pixel route 的集成测试。
- 运行完整相关回归并检查架构 v3 checkpoint round-trip。

### 运行过哪些测试
- PixelSpaceRouter 与 renderer 定向测试：`8 passed`。
- 相关文件 `py_compile`：passed。

### 下一步最小任务
- 补 EndoMoe preset 与 renderer trainable-route 集成测试。

## Update 2026-06-10 pixel-routing integration contracts

### 已完成
- preset 测试新增 `use_pixel_routing=True` 与 hidden dim 32 断言。
- renderer 测试新增 trainable route 调用捕获，验证 RGB/depth/prior/motion/coverage 的形状契约。
- 验证 pixel residual logits 返回 deformation aux，后续可直接进入 TensorBoard 诊断。

### 当前设计决策
- renderer 与 PixelSpaceRouter 之间使用纯 tensor 契约，不让 renderer 依赖具体网络实现。
- pixel router 返回 residual logits 而非最终 masked logits，便于监控网络自身是否开始偏离 Gaussian prior。

### 仍需做什么
- 运行 preset、pixel router、renderer 与 checkpoint 定向测试。
- 运行完整相关回归与 diff check。
- 若通过，Phase 6 可标记完成；下一步进入 visibility/lifecycle photometric integration。

### 运行过哪些测试
- 本步两个测试文件修改后尚未运行。

### 下一步最小任务
- 运行 Phase 6 定向与完整回归。

## Update 2026-06-10 Phase 6 completion verification

### 已完成
- EndoMoe preset、PixelSpaceRouter、renderer integration 与 component checkpoint 定向测试：`6 passed`。
- 修正架构版本断言为 `endomoeg_v3` 后，完整相关回归：`92 passed`。
- Phase 6 的 Gaussian prior 与 pixel-space router 已标记完成。

### 当前设计决策
- 最终图像由 trainable pixel router 在专家 render 之间组合，Gaussian router 提供稳定 log-prior。
- pixel router 初始严格等价于 Gaussian prior，随后由 photometric loss 学习局部 residual routing。
- 无 dynamic coverage 的像素始终回退到 Gaussian fallback prior，避免全 `-inf`、NaN 或纯黑帧。

### 仍需做什么
- Phase 7：检查 visibility/lifecycle 的 expert-axis 对齐和最终 alpha/RGB 梯度。
- 修复 temporal adjacent sampling 与真实 KNN spatial regularization。
- 为 pixel route 增加 TensorBoard coverage、entropy 和 residual-logit 统计。

### 运行过哪些测试
- Phase 6 定向测试：`6 passed`。
- 完整相关回归：`92 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 审计 `ToolContactExpert` 的 visibility/lifecycle 输出如何映射到三个 geometry experts 与最终 pixel alpha。

## Update 2026-06-10 visibility/lifecycle double-gating fix

### 已完成
- 确认 `ToolContactExpert` 已在 alpha probability 域将 visibility 与 lifecycle gate 写入 `opacity_logits`。
- 确认 pixel renderer 此前又将 `vis_expert_visibility_alpha` 与 `lifecycle_expert_alpha` 乘到激活后的 opacity，形成二次门控。
- EndoMoe aux 新增 `expert_opacity_includes_visibility=True`，renderer 据此跳过重复 gate。
- 旧 CAMS/fake deformation 未声明该标记时仍维持原 renderer 门控行为。
- 增加 EndoMoe opacity proposal 契约测试。

### 当前设计决策
- visibility/lifecycle 的唯一几何作用点是 expert proposal 的 opacity probability；renderer 不得重复应用。
- appearance residual 仍由 renderer 按 expert 施加，因为它没有预编码进 Gaussian SH/color proposal。
- 显式 aux capability flag 优于按 tracking type 猜测，便于旧模型与测试兼容。

### 仍需做什么
- 增加 renderer 回归，精确验证声明 embedded gate 时 opacity 不再二次衰减。
- 检查 `d_opacity_logit` 是否应报告完整 gated delta，而不是仅 raw opacity head delta。
- 检查 lifecycle/visibility 正则是否与实际 alpha 语义一致。

### 运行过哪些测试
- 本步三个文件修改后尚未运行测试。

### 下一步最小任务
- 增加 renderer 单次门控数值测试并运行定向回归。

## Update 2026-06-10 transient appearance and visibility decoupling

### 已完成
- `ToolContactExpert` 新增独立 `visibility_head`，identity 初始化为 open。
- `transient_head` 仅控制 transient appearance probability，不再反向定义 visibility。
- opacity gate 改为独立 visibility alpha × lifecycle persistent alpha。
- `d_opacity_logit` 改为报告最终有效 opacity delta；raw opacity head delta 单独保存为 `raw_d_opacity_logit`。
- 架构版本提升为 `endomoeg_v4`。
- 增加“高 transient 仍保持可见”和“visibility 独立压制 opacity”测试。

### 当前设计决策
- transient 表示需要 appearance residual 的动态组织/器械区域，不等价于不可见。
- visibility 表示 view-dependent occlusion/visibility，直接控制 alpha。
- lifecycle 表示跨时间持久/消失状态，和 visibility 相乘进入 alpha。
- 三种语义必须分头建模，否则 appearance gradient 会被同一概率同步抹除。

### 仍需做什么
- 运行 identity、解耦、单次门控与 checkpoint 架构定向测试。
- 更新 aux，显式暴露 transient probability 与 visibility alpha 的统计。
- 检查 loss 中 `pi_vis`/visibility sparse/lifecycle balance 是否应分别作用于 transient、visibility、lifecycle。

### 运行过哪些测试
- 单次门控修复定向测试：`3 passed`。
- 本轮解耦修改尚未运行测试。

### 下一步最小任务
- 运行 ToolContact 解耦与 EndoMoe checkpoint 定向回归。

## Update 2026-06-10 normalized transient routing semantics

### 已完成
- 修正解耦后 `pi_vis=[visibility, transient]` 不归一化的问题。
- `pi_vis` 现在严格表示 `[stable_probability, transient_probability]`，两项和为 1。
- visibility alpha 继续独立控制 opacity，不参与 stable/transient normalization。
- global/local identity experts 显式输出零 transient probability。
- EndoMoe aux 新增 full expert transient probability 与 per-geometry-expert transient probability。

### 当前设计决策
- `pi_vis` 只承担 stable/transient appearance routing 和相关 balance/confidence loss。
- `visibility_alpha` 单独承担 occlusion alpha。
- `lifecycle_probs` 单独承担 persistent/transient lifecycle；三者不再共享概率语义。

### 仍需做什么
- 运行概率归一化、visibility 解耦、checkpoint 与完整相关回归。
- 为 visibility occlusion 增加独立 sparse/identity regularization 与日志。
- 检查 lifecycle balance 是否应仅在 full expert 获得足够 pixel coverage 后启用。

### 运行过哪些测试
- 解耦前定向测试：`6 passed`。
- 本轮概率语义修正尚未运行。

### 下一步最小任务
- 运行 transient/visibility 概率语义定向测试与完整回归。

## Update 2026-06-10 EndoMoe visibility and lifecycle loss repair

### 已完成
- 修复 `_add_cams_patch_c_losses()` 只识别旧 CAMS 阶段名、导致 EndoMoe appearance/lifecycle 正则从未生效的问题。
- `moe_expert_full` 与 `moe_joint_finetune` 现在启用 appearance、visibility、transient 与 lifecycle 约束。
- 新增独立 occlusion sparse prior 与 transient appearance sparse prior。
- lifecycle regularization 从 `logits²` 改为 transient probability，避免将 `+12/-12` identity bias 强行拉向零。
- 增加 EndoMoe 阶段下各 loss 精确数值测试。

### 当前设计决策
- visibility occlusion、transient appearance、lifecycle disappearance 都从 identity 稀疏先验开始，仅在 photometric objective 有证据时打开。
- 不再惩罚 lifecycle logits 的绝对幅度，因为高幅度正是 identity-safe persistent 初始化所需。
- full expert 独立训练阶段必须学习这些控制；router-only 阶段冻结专家，不额外施加专家正则。

### 仍需做什么
- 为 cutting/pulling EndoMoe preset 显式设置新权重与更保守的 lifecycle persistent target。
- 运行 loss 定向测试与完整回归。
- 检查 TensorBoard 是否记录新增 visibility/transient/lifecycle 指标。

### 运行过哪些测试
- transient/visibility 语义定向测试：`5 passed`。
- 本轮 loss 修复尚未运行。

### 下一步最小任务
- 更新 EndoMoe presets 的 visibility/lifecycle 权重与 target。

## Update 2026-06-10 conservative EndoMoe visibility presets

### 已完成
- cutting/pulling EndoMoe preset 将 stable/transient target 从 85/15 调整为 98/2。
- lifecycle persistent target 调整为 0.98。
- 显式配置 appearance、visibility occlusion、transient sparse、lifecycle balance 与 lifecycle sparse 权重。
- visibility balance/confidence 权重降低，避免辅助先验压过 photometric objective。

### 当前设计决策
- 内窥镜动态重建中绝大多数 Gaussian 在大多数时刻应保持可见且持久；遮挡、transient appearance 和 lifecycle disappearance 应由局部图像证据打开。
- 2% transient 仅作为防止分支完全死亡的弱先验，不是最终占比约束。

### 仍需做什么
- 更新 preset 测试断言。
- 运行 visibility/lifecycle loss、preset、checkpoint 与 renderer 完整回归。
- 通过后继续 temporal adjacent sampling 与 spatial KNN 修复。

### 运行过哪些测试
- 本步两个 preset 文件修改后尚未运行测试。

### 下一步最小任务
- 补 preset target/weight 回归并运行 Phase 7 当前修改的完整测试。

## Update 2026-06-10 visibility/lifecycle verification complete

### 已完成
- preset、EndoMoe visibility/lifecycle loss、transient/visibility 解耦、单次 opacity gate 与 v4 checkpoint 定向测试：`6 passed`。
- 完整相关回归：`97 passed`。
- `pi_vis`、visibility alpha、transient appearance 与 lifecycle persistent 已形成互不混淆的概率语义。

### 当前设计决策
- visibility/lifecycle 子阶段完成，可以进入 temporal/spatial supervision。
- 服务器新日志应重点观察 `mean_visibility_alpha`、`mean_occlusion_probability`、`mean_transient_probability`、`mean_lifecycle_persistent`，它们不应在训练早期突变。

### 仍需做什么
- 修复 temporal loss 的相邻时间采样，使每次训练真正产生 `d_mu_sequence/time_sequence`。
- 替换当前错误依赖 `distCUDA2` 标量输出排序的 spatial KNN 实现。
- 增加 pixel router coverage/entropy 日志。

### 运行过哪些测试
- Phase 7 visibility/lifecycle 定向测试：`6 passed`。
- 完整相关回归：`97 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 审计 `train.py` 中 temporal pair 的生成与 aux merge 路径。

## Update 2026-06-10 real adjacent-time deformation queries

### 已完成
- 新增 `utils/temporal_utils.py`，提供排序去重时间与最近相邻时间查询。
- 修复 `deform_network.forward/forward_dynamic` 未接收 camera 的问题；ToolContact 的 view direction/depth/screen features 现在真正可达。
- fine 训练在 `lambda_geo_temp>0` 时，使用同一参考相机姿态与最近真实相邻 timestamp 额外查询一次 deformation。
- temporal loss 使用当前帧与相邻时间的同一组 Gaussian `d_mu`，不再依赖随机 dataloader batch 恰好包含多帧。
- 删除随机 batch 时间序列作为 temporal pair 的旧逻辑。

### 当前设计决策
- temporal regularization 比较同一 camera conditioning 下的相邻时间，隔离时间变化，避免相机运动污染 3D velocity。
- 相邻时间来自数据集实际 timestamp，而非固定假设 `1/N`。
- 额外查询只执行 deformation network，不执行第二次 rasterization，控制训练开销。

### 仍需做什么
- 增加 nearest-adjacent-time 边界测试与 camera forwarding 测试。
- 验证 temporal query 的图可反向传播，且 current/adjacent `d_mu` shape 一致。
- 完整回归后修复 spatial KNN。

### 运行过哪些测试
- visibility/lifecycle 完整相关回归：`97 passed`。
- 本轮 temporal/camera 修改尚未运行。

### 下一步最小任务
- 增加 temporal utility 与 camera forwarding 回归测试，并运行 py_compile。

## Update 2026-06-10 scalable spatial KNN regularization

### 已完成
- 移除将 `distCUDA2` 标量最近距离误当作邻居矩阵的失效实现。
- 新 spatial loss 在最多 2048 个随机 Gaussian 子集上执行 chunked exact KNN，避免完整 `N²` 内存。
- motion 先按 scene scale 归一化，再计算邻域 robust difference。
- 邻居贡献使用自适应空间距离核权重，降低跨组织边界过度平滑。
- 新增 sample size、K 与 chunk size 配置，以及实际 sample/neighbor count 日志。

### 当前设计决策
- 使用 stochastic sampled graph regularization，在每轮覆盖不同 Gaussian，同时保持计算量可控。
- KNN 索引与空间权重不参与梯度；梯度仅流向 motion。
- 使用 Charbonnier 型 motion difference，而非平方差，降低器械/切割边界异常值主导。

### 仍需做什么
- 增加 coherent motion 零损失、局部 outlier 正损失与 gradient 测试。
- 运行 temporal/camera 与 spatial KNN 完整回归。
- 根据服务器速度决定 sample size 是否从 2048 下调到 1024。

### 运行过哪些测试
- temporal utility、camera forwarding 与 temporal loss 定向测试：`3 passed`。
- 本轮 spatial 修改尚未运行。

### 下一步最小任务
- 增加 spatial KNN 数值和梯度回归测试。

## Update 2026-06-10 pixel-router observability completion

### 已完成
- pixel usage 统计优先使用真实 `pixel_expert_coverage`，不再因 fallback weights 将背景误判为动态覆盖。
- 新增 pixel route entropy、max probability、covered fraction 与每专家 coverage。
- 新增 pixel router residual logits 的 mean/max absolute magnitude。
- 扩展 covered-pixel routing 测试，验证 coverage、entropy 与 residual diagnostics。

### 当前设计决策
- pixel usage 只在至少一个动态专家真实 raster coverage 的像素上统计。
- fallback 负责保证图像有限且不黑屏，但不能被计入专家训练覆盖率。
- residual logits 初始应接近零；快速增大表示 pixel router 正在压过 Gaussian prior，需要监控。

### 仍需做什么
- 运行 spatial KNN、pixel diagnostics、temporal query 的定向与完整回归。
- 将新增 pixel metrics 加入 500-step 控制台摘要。
- 完成后评估 Phase 7 是否可收口。

### 运行过哪些测试
- spatial KNN 与 temporal/camera 定向测试：`4 passed`。
- 本轮 pixel diagnostics 尚未运行。

### 下一步最小任务
- 运行 Phase 7 全部定向测试与完整相关回归。

## Update 2026-06-10 Phase 7 completion verification

### 已完成
- temporal neighbor、camera forwarding、spatial KNN、pixel diagnostics、visibility/lifecycle 定向测试：`6 passed`。
- 完整相关回归达到 `100 passed`。
- 500-step 控制台摘要加入 pixel route、temporal pair 与 spatial KNN 核心指标。
- Phase 7 标记完成，进入 diagnostics/reproducibility 阶段。

### 当前设计决策
- temporal loss 使用同姿态相邻真实 timestamp deformation query。
- spatial loss 使用 scene-normalized stochastic sampled KNN。
- visibility、transient appearance、lifecycle 与 geometry/pixel routing 均具有独立语义和可观测指标。

### 仍需做什么
- 恢复固定视角 train/test validation PSNR/SSIM/LPIPS，避免只看随机训练 batch。
- 增加 optimizer group LR/grad norm 与 per-expert counterfactual diagnostics。
- 更新实验命令、README 与可复现配置说明。

### 运行过哪些测试
- Phase 7 定向测试：`6 passed`。
- 完整相关回归：`100 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 审计 `training_report()` 中被整段注释的固定视角 validation，并安全恢复轻量评估。

## Update 2026-06-10 fixed-view validation restoration

### 已完成
- 新增确定性 evenly-spaced fixed-view 选择工具，每个 train/test split 最多评估 4 帧。
- `training_report()` 恢复固定视角 L1、PSNR、SSIM、LPIPS 与样例图 TensorBoard 日志。
- validation 仅在 `testing_iterations` 执行，不影响每步训练开销。
- renderer 新增 `update_deformation_stats=False`，validation render 不再污染 deformation accumulation 与后续动态点判定。
- validation 使用当前设备，不再硬编码 CUDA。

### 当前设计决策
- 固定视角指标用于比较阶段、配置和 baseline；随机训练 batch PSNR 仅作为优化即时反馈。
- 每个 split 选择覆盖首尾与中间时间的固定视角，保证日志跨 run 可比。
- validation 图像只记录前两个 fixed views，控制 TensorBoard 体积。

### 仍需做什么
- 增加 fixed-view selection 与 renderer stats-disable 测试。
- 验证 training_report 参数调用和 py_compile。
- 后续补 optimizer-group LR/grad norm diagnostics。

### 运行过哪些测试
- Phase 7 完整相关回归：`100 passed`。
- 本轮 validation 修改尚未运行。

### 下一步最小任务
- 增加 fixed-view selection 与 validation stats isolation 回归测试。

## Update 2026-06-10 optimizer-group gradient diagnostics

### 已完成
- 新增 optimizer group metrics 工具。
- fine 训练每 10 步记录每组实际 LR、全局 L2 grad norm 与 gradient parameter coverage。
- 可直接区分冻结组、LR 为零、无 photometric gradient 和仅部分 head 收到梯度。
- 增加构造型数值测试。

### 当前设计决策
- 诊断按 optimizer group 而非模块名聚合，与阶段 scheduler 的 trainable prefix 完全对齐。
- gradient coverage 按参数元素数量计算，避免一个小 bias 有梯度就掩盖大部分权重 dead。
- 每 10 步记录以平衡可观测性和 TensorBoard 体积。

### 仍需做什么
- 运行 fixed validation 与 optimizer diagnostics 定向测试。
- 运行完整相关回归。
- 更新 README 的训练、独立阶段、组件装配和实验命令。

### 运行过哪些测试
- fixed-view selection 与 stats isolation 定向测试：`3 passed`。
- optimizer diagnostics 尚未运行。

### 下一步最小任务
- 运行 Phase 8 当前修改的定向与完整回归。

## Update 2026-06-10 Phase 8 diagnostics verification

### 已完成
- fixed-view validation、stats isolation、optimizer group metrics 与 spatial diagnostics 定向测试：`4 passed`。
- 完整相关回归达到 `103 passed`。
- TensorBoard 现可同时观察固定视角质量、双层路由、visibility/lifecycle、temporal/spatial 与每组梯度健康度。

### 当前设计决策
- 工程诊断已足以判别 PSNR collapse、黑帧、专家 starvation、router collapse、无 temporal pair 与无 spatial gradient。
- Phase 8 剩余工作集中于实验脚本与文档，不再扩展模型结构。

### 仍需做什么
- 更新 README：方法结构、完整训练、独立 expert/router/joint 作业、组件目录与 TensorBoard 指标。
- 给出服务器下一步 baseline/shared-base/full model/ablation 命令。
- 最后运行文档相关 diff review。

### 运行过哪些测试
- Phase 8 定向测试：`4 passed`。
- 完整相关回归：`103 passed`。
- 相关文件 `py_compile`：passed。
- `git diff --check`：passed，仅 Windows LF/CRLF 提示。

### 下一步最小任务
- 读取 README 当前训练章节并添加 EndoMoeGaussian 实验协议。

## Update 2026-06-10 independent-stage command support

### 已完成
- 新增 `endomoeg_component_output_dir`，global/local/full/router 可将组件写入同一共享目录。
- 新增 `endomoeg_stage_iterations`，配置加载后覆盖 fine iterations 与 LR horizon，避免独立阶段仍被主 preset 固定为 15000 步。
- 连续 full run 未设置这些参数时维持原有输出布局与 15000-step schedule。
- 增加 parser 安全默认值测试。

### 当前设计决策
- 所有独立阶段共享一个 assembly directory，router/joint 可直接严格加载完整组件集合。
- stage iterations 与主 preset iterations 分离，避免复制五套近重复 config。
- coarse static reconstruction 暂仍在每个独立作业中确定性重建；后续可再增加 static checkpoint reuse。

### 仍需做什么
- 运行 parser、py_compile 与完整相关回归。
- README 给出 global→local→full→router→joint 的共享组件目录命令。
- 更新旧 9000-step 与同构专家描述。

### 运行过哪些测试
- Phase 8 diagnostics 完整相关回归：`103 passed`。
- 本轮独立阶段入口尚未运行。

### 下一步最小任务
- 运行新参数定向测试后修改 README。

## Update 2026-06-10 README and experiment protocol

### 已完成
- README 新增 EndoMoeGaussian v4 架构说明。
- 更新主 preset 为 1000 coarse + 15000 fine 与绝对阶段边界。
- 增加连续训练命令和 global→local→full→router→joint 独立组件装配命令。
- 增加 fixed-view、双层 route、visibility/lifecycle、temporal/spatial 与 optimizer gradient 监控清单。
- 增加 baseline、continuous、independent assembly 与必要 ablation 实验协议。
- 标明 v1/v2/v3 组件 checkpoint 不兼容 v4。

### 当前设计决策
- 独立组件训练是主要 MoE-GS-style 方法验证；连续 15000-step run 用作工程对照。
- 结论必须基于固定视角 validation 和 baseline/shared-base/best-single-expert 比较，不能只看随机训练 PSNR。

### 仍需做什么
- 运行最终完整相关回归、py_compile 与 diff review。
- 检查 git status，保留 `.ai-recovery/` 与 `.claude/logs/` 不提交。
- 尚未 commit/push，等待最终代码审查后执行。

### 运行过哪些测试
- 独立阶段参数定向测试：`3 passed`。
- README 修改不涉及运行时代码。

### 下一步最小任务
- 执行最终 103+ 回归与工作区审查，确认可进入 commit/push。

## Update 2026-06-10 final implementation audit

### 已完成
- Phase 8 诊断、测试、独立阶段命令与 README 实验协议全部完成。
- 最终相关回归为 `103 passed`，所有本轮修改 Python 文件通过 `py_compile`。
- `git diff --check` 通过，仅有 Windows LF/CRLF 提示。
- `plan/task_plan.md` 已将 Phase 8 标记完成，工程进入 Phase 9 服务器实验阶段。

### 当前设计决策
- 当前代码冻结为 EndoMoeGaussian v4 工程实现，不再在缺少新服务器日志前继续扩展模型结构。
- 后续结论以 fixed-view validation、baseline、单专家、独立组件装配和完整模型对照为准。
- `.ai-recovery/`、`.claude/logs/` 与 pytest 临时目录不进入代码提交。

### 仍需做什么
- 在服务器运行 README 中的 baseline、continuous full、independent assembly 和关键 ablation。
- 根据新 TensorBoard 的 fixed-view、router、visibility/lifecycle、temporal/spatial 与 optimizer-group 指标决定下一轮修改。
- 当前尚未 commit/push。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py tests/test_disentangled_moe_tracking.py -q --tb=line --basetemp .pytest_tmp_endomoeg_final_audit`：`103 passed in 3.47s`。
- 相关 Python 文件 `python -m py_compile ...`：passed。
- `git diff --check`：passed，仅 LF/CRLF warning。
- Codex 外部审核尝试两次：指定 `gpt-5.2-codex` 不受当前账号支持；默认模型调用超时，未获得可用审核结论。

### 下一步最小任务
- 审核提交文件列表后，执行 Conventional Commit 并推送 GitHub；随后在服务器启动 Phase 9 第一组 baseline/full 对照实验。

## Update 2026-06-10 recovery snapshot refresh

### 已完成
- 清理本轮已知 pytest 临时目录。
- 刷新 `status.txt`、`diffstat.txt`、`changed-files.txt` 与 `untracked-files.txt`，不再引用旧 CAMS-GS diff。

### 当前设计决策
- recovery 快照只反映当前工作区；源码提交仍排除 `.ai-recovery/` 与 `.claude/logs/`。

### 仍需做什么
- 提交前确认 `plan/` 是否随工程文档一并纳入版本控制。

### 运行过哪些测试
- 本节仅刷新恢复元数据；最终代码验证结果保持 `103 passed`、`py_compile` passed、`git diff --check` passed。

### 下一步最小任务
- 等待用户确认后执行 commit/push，或先在本地审阅拟提交文件清单。

## Update 2026-06-10 GitHub push complete

### 已完成
- 将 21 个源码、测试、README 与工程计划文件提交为 `8b3eb20`。
- Conventional Commit：`feat(model): rebuild EndoMoeGaussian v4 pipeline`。
- 已推送到 `origin/main`，远端 `refs/heads/main` 与本地 HEAD 均为 `8b3eb2018c8d2a35d4b9e4f2764afa59d08e2a54`。

### 当前设计决策
- `.ai-recovery/` 与 `.claude/logs/` 保持本地未跟踪，不进入 GitHub。
- 当前代码版本冻结，等待服务器训练数据，不在训练验证前继续修改结构。

### 仍需做什么
- 用户在服务器执行 `git pull origin main`，按 README 的 Phase 9 命令运行 baseline 与 EndoMoeGaussian v4。
- 收集新的 TensorBoard、fixed-view 指标和渲染结果后再决定下一轮优化。

### 运行过哪些测试
- 推送前相关回归：`103 passed`。
- `py_compile`：passed。
- `git diff --check`：passed，仅 LF/CRLF warning。
- 推送后本地与远端 commit hash 完全一致。

### 下一步最小任务
- 在服务器拉取 `8b3eb20` 并启动第一组训练验证。

## Update 2026-06-10 clean post-push snapshot

### 已完成
- recovery 的 status、diffstat、changed-files 与 untracked-files 已更新为推送后的真实状态。

### 当前设计决策
- Git 工作树没有未提交的跟踪文件；仅保留 recovery 与 Claude 会话日志。

### 仍需做什么
- 等待服务器训练验证。

### 运行过哪些测试
- 本节仅更新恢复快照，无代码变更。

### 下一步最小任务
- 拉取并训练。

## Update 2026-06-10 Python 3.7/3.8 annotation compatibility root cause

### 已完成
- 定位服务器导入失败根因：README 环境为 Python 3.7，服务器为 Python 3.8，但新代码混入 PEP 585 `tuple[...]` 与 PEP 604 `A | B` 注解。
- `utils/eval_utils.py` 与 `utils/temporal_utils.py` 已改用 `typing.Tuple`。
- `models/tracking/heterogeneous_moe_tracking.py` 已改用 `typing.Union`。

### 当前设计决策
- 不以单独添加 postponed annotations 掩盖问题；运行时代码公开注解统一使用 Python 3.7 可解析、可求值的 `typing` 类型。
- 兼容目标按 README 的 Python 3.7 执行，因此同时禁止 PEP 585 与 PEP 604 注解。

### 仍需做什么
- 修正 `scene/tracking_losses.py` 中剩余现代注解。
- 新增静态兼容测试，覆盖本轮 EndoMoeGaussian 运行时文件，防止再次引入高版本注解。

### 运行过哪些测试
- 当前完成根因审计，尚未运行修正后测试。

### 下一步最小任务
- 完成损失模块注解修正并加入 Python 3.7 注解契约测试。

## Update 2026-06-10 Python annotation compatibility implementation

### 已完成
- `scene/tracking_losses.py` 的返回值和可选参数统一改为 `Tuple` / `Optional`。
- 新增 `tests/test_python_compatibility.py`，静态检查 EndoMoe 运行时文件中的 PEP 585 与 PEP 604 注解。
- 兼容测试自身使用 `getattr(..., "posonlyargs", ())`，可运行于 Python 3.7 的 AST API。

### 当前设计决策
- 兼容测试覆盖训练入口、renderer、tracking、scene、arguments 与新增 utils，而不是只检查首个报错文件。
- 保留 `from __future__ import annotations` 的现有模块，但不依赖它绕过高版本类型语法。

### 仍需做什么
- 运行兼容测试、相关完整回归、`py_compile` 和注解模式复查。
- 验证通过后提交并推送修复。

### 运行过哪些测试
- 本节完成代码与测试实现，尚未执行。

### 下一步最小任务
- 运行最小兼容回归并确认训练入口导入链不再包含高版本注解。

## Update 2026-06-10 Python compatibility verification complete

### 已完成
- 全部相关运行时文件已消除 PEP 585 与 PEP 604 注解。
- `typing.get_type_hints` 可解析 fixed-view、temporal 与 scene-scale 接口。
- Python 3.9 解释器可直接执行兼容契约并导入不依赖 CUDA 的修正模块。

### 当前设计决策
- 项目兼容下限继续以 README 的 Python 3.7 为准。
- 由于本机没有 Python 3.7/3.8，使用 AST 契约检测这两个版本不支持的注解形式，并用可用的 Python 3.9 做真实解释器验证。

### 仍需做什么
- 创建修复提交并推送 `origin/main`。
- 用户服务器拉取后重新执行训练入口，确认已越过 `utils.eval_utils` 导入阶段。

### 运行过哪些测试
- 相关完整回归：`104 passed in 3.94s`。
- 修正文件与兼容测试 `py_compile`：passed。
- `get_type_hints`：passed。
- Python 3.9 兼容契约与模块导入：passed。
- 现代注解残留检查：none。
- `git diff --check`：passed，仅 LF/CRLF warning。

### 下一步最小任务
- 提交并推送 Python 3.7/3.8 兼容修复。

## Update 2026-06-10 Python compatibility fix pushed

### 已完成
- 修复提交为 `c13da84`：`fix(runtime): support Python 3.7 annotations`。
- 已推送到 `origin/main`，本地与远端完整哈希均为 `c13da8478a86ead9f26e8f63cc838a9330ba1df6`。

### 当前设计决策
- EndoMoe 运行时注解必须符合 Python 3.7；兼容测试作为长期提交门禁保留。

### 仍需做什么
- 服务器执行 `git pull origin main` 后重新启动训练。
- 若出现下一处导入错误，优先检查服务器依赖版本与 README 环境契约，而非局部绕过。

### 运行过哪些测试
- `104 passed`。
- Python 3.9 真实解释器兼容验证：passed。
- 本地与远端 divergence：`0 0`。

### 下一步最小任务
- 服务器确认 HEAD 为 `c13da84` 并重新运行训练命令。

## Update 2026-06-10 renderer depth contract root cause

### 已完成
- 确认 CUDA rasterizer 源码的标准 depth 输出为 `[1,H,W]`。
- 定位 MoE 深度融合缺少显式通道处理：RGB 使用 `weights.unsqueeze(1)`，depth 却直接与 `[E,H,W]` 权重相乘。
- renderer 新增 rasterizer 边界规范化，统一输出 `[1,H,W]`。
- expert depth 在路由前统一为 `[E,H,W]`，融合后显式恢复 `[1,H,W]`。
- 增加单通道 CUDA 风格 expert depth 与 legacy replicated depth 回归测试。

### 当前设计决策
- 不在 `render.py` 重建阶段容忍错误输出；`gaussian_renderer` 作为产生端负责稳定 depth 契约。
- 支持已知 legacy `[3,H,W]` replicated depth，并在边界聚合为单通道；后续所有训练、路由、保存和重建只接收 `[1,H,W]`。

### 仍需做什么
- 运行 renderer 定向测试与完整相关回归。
- 检查训练 depth loss 的输入形状仍为 `[B,1,H,W]`。
- 验证通过后提交并推送。

### 运行过哪些测试
- 当前完成两文件修改，尚未执行修正后测试。

### 下一步最小任务
- 运行 depth contract 与 pixel routing 定向测试。

## Update 2026-06-10 renderer depth contract verification

### 已完成
- CUDA 风格 `[1,H,W]` expert depth 会在路由前转换为 `[E,H,W]`，最终融合为 `[1,H,W]`。
- direct rasterization 与 legacy replicated `[3,H,W]` depth 均在 renderer 边界规范化。
- 旧 fake rasterizer 已按真实 raster settings 动态生成图像和 depth 尺寸。

### 当前设计决策
- `render.py::_tensor_to_hw_numpy` 继续严格拒绝三通道 depth，用于捕获 renderer 契约回归。
- 训练、渲染、PixelSpaceRouter 与点云重建共享同一单通道 depth 语义。

### 仍需做什么
- 创建 renderer 修复提交并推送。
- 用户服务器拉取后重新执行 test render 与 metrics。

### 运行过哪些测试
- depth/pixel-routing 定向测试：`3 passed`。
- 完整相关回归：`105 passed in 3.77s`。
- `py_compile`：passed。
- `git diff --check`：passed，仅 LF/CRLF warning。

### 下一步最小任务
- 提交并推送 renderer depth contract 修复。

## Update 2026-06-11 frozen Router training stage

### 已完成
- `endomoeg_pipeline_stage='router'` 已接入独立训练循环。
- Stage 3 加载并冻结三个 complete expert，不创建 Gaussian optimizer，也不运行 densification/pruning。
- 每帧分别渲染三专家、splat volumetric routing features、执行 pixel routing 并融合 RGB/depth。
- Router 使用独立 LR 参数组：per-Gaussian logits、Gaussian feature MLP、pixel router。
- Loss 包含重建 L1、SSIM、oracle-error distillation 和 anti-starvation。
- 训练从 dense routing 过渡到 soft top-2。
- warmup 后强制检查三条 Router 梯度链；任何一组无梯度立即报错。
- 最终 fixed-view validation 后保存 `router.pth` identity-bound bundle。

### 当前设计决策
- Stage 3 完全禁止专家更新；联合微调不会复用该循环的 optimizer。
- Router 的 final quality 以 fixed-view test PSNR/SSIM 为准。
- pixel residual 指标与三组梯度 norm 每步写入 TensorBoard，避免再次出现“配置启用但实际未训练”。

### 仍需做什么
- 为 Stage 3 render/gradient/train helper 增加定向测试。
- 检查 current dataset 与 bundle source_path 一致性。
- 实现 Router bundle 加载渲染和可选 Stage 4。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 添加 Stage 3 的 synthetic frozen ensemble 测试并运行。

## Update 2026-06-11 Router bundle identity contract

### 已完成
- 新增版本化 frozen-expert Router bundle。
- Router bundle 绑定 source canonical fingerprint、三个专家训练后 fingerprint、point count、tracking architecture 和单专家验证 PSNR。
- 加载/验证时替换任何专家、改变 topology 或来源 canonical 都会 fail-fast。
- Router state、训练 config 与最终 validation metrics 一并保存。

### 当前设计决策
- Router checkpoint 不只是网络权重，而是“Router + 精确专家集合”的装配契约。
- 不支持把同角色的另一个 checkpoint 热替换进既有 Router，即使网络结构相同。

### 仍需做什么
- 运行 Router bundle 验证测试。
- 实现 Stage 3 训练循环和 Router validation。
- render/eval 按 Router manifest 加载专家。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- Router bundle 测试通过后接入 Stage 3。

## Update 2026-06-11 volume-aware router core

### 已完成
- 新增 `EndoMoeVolumeAwareRouter`。
- 每个专家拥有与自身 point count 一致的 learnable per-Gaussian base logits。
- 共享 Gaussian feature MLP 使用 canonical xyz、deformation、view direction、opacity、scale、time 和 role embedding。
- 复用 pixel residual router，在 volumetric prior 上学习局部图像空间修正。
- 新增 dense→soft top-2 权重转换。
- 新增 oracle-error targets、最终重建 loss 和非均匀 minimum-usage anti-starvation。

### 当前设计决策
- Anti-starvation 只设置低 usage floor，不强制三专家均匀；contact 允许天然稀疏。
- Oracle target 由每个冻结专家相对 GT 的逐像素误差生成并 detach，Router 学习“谁在该像素更可靠”。
- Gaussian MLP 最后一层和 pixel residual 最后一层均零初始化，Router 初始输出严格继承 volumetric prior。

### 仍需做什么
- 运行 Router 单元测试与 Python compatibility。
- 将 frozen ensemble、renderer routing splat 和 Router loss 接入 Stage 3 训练循环。
- 保存 Router bundle 并支持 render/eval。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 运行 Router shape、gradient、oracle 和 starvation 定向测试。

## Update 2026-06-11 volumetric routing render boundary

### 已完成
- Renderer 新增可选 `return_routing_state`，普通训练/渲染默认行为不变。
- Routing state 包含当前专家自己的 canonical/deformed xyz、activated opacity/scale/rotation、motion、covariance 和 scene scale。
- 新增 differentiable routing-feature splat：将 per-Gaussian learnable logits、projected motion 和 coverage 投影到 pixel space。
- 增加 router logits 经 rasterizer 回传梯度的定向测试。

### 当前设计决策
- Stage 3 对每个不同 topology 的 expert 分别执行一次完整渲染和一次轻量 routing-feature splat。
- 不尝试在 3D Gaussian index 上对齐专家；所有组合都发生在 pixel space。
- Volumetric prior 使用每个专家自身 Gaussian 几何和 alpha compositing，因此保留遮挡与深度结构。

### 仍需做什么
- 运行 renderer routing gradient 测试。
- 实现 per-Gaussian logits + shared feature MLP + pixel residual router。
- 实现 oracle-error distillation 与 anti-starvation loss。

### 运行过哪些测试
- 本节完成 2 个文件修改，尚未执行测试。

### 下一步最小任务
- 先验证 routing-feature splat 的真实梯度契约。

## Update 2026-06-11 frozen expert ensemble

### 已完成
- 新增 `FrozenExpertEnsemble`，按 `global/local/contact` 固定顺序加载三个 complete expert bundle。
- 加载时校验同源 canonical fingerprint、角色、完整专家 tracking type、原始 hidden config 和 minimum PSNR。
- 每个专家按自己的 config 重建 GaussianModel，并恢复独立 point count、topology、appearance、deformation 与空间上下文。
- 加载完成后强制冻结所有 Gaussian 参数和 deformation 参数，并提供 fail-fast 检查。

### 当前设计决策
- Stage 3 不创建任何 Gaussian optimizer，专家模型只作为 frozen renderer。
- 三个专家允许 point count 完全不同；Router 只在 pixel space 融合，不建立 Gaussian index 对齐。
- 专家顺序固定为 global/local/contact，避免 checkpoint 与 Router channel 语义错位。

### 仍需做什么
- 运行 frozen-state 定向测试。
- renderer 输出 routing state 与 volumetric feature splatting。
- 实现 learnable per-Gaussian logits 和 pixel residual router。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 验证 frozen ensemble 边界后修改 renderer。

## Update 2026-06-11 expert spatial context contract

### 已完成
- Complete expert state 现在强制保存/恢复 `scene_scale`、`xyz_max` 和 `xyz_min`。
- Local/Contact 独立加载后会恢复与训练时一致的空间归一化边界，不再依赖当前进程重新推断。
- Expert bundle config 现在保存完整 Model/Hidden/Optimization 参数，用于 Stage 3 精确重建每个异构专家实例。

### 当前设计决策
- AABB 属于专家表示的一部分；尤其 Local/Contact 的 spatial refinement 不能借用另一 expert 或当前数据加载过程的边界。
- Router stage 必须根据每个 bundle 自己的 hidden params 实例化 GaussianModel，不能用一个公共 config 强行加载三种专家。

### 仍需做什么
- 运行 spatial-context round-trip 测试。
- 实现 frozen expert loader。
- renderer 暴露路由所需的 deformed Gaussian state。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 运行 bundle/Stage 1/2 回归后新增 frozen ensemble loader。

## Update 2026-06-11 Stage 1 and Stage 2 verification

### 已完成
- Stage 1/2 parser、绝对路径、bundle、完整专家结构和 Python 3.7 annotation 回归全部通过。
- 普通 preset 默认参数仍可解析，新 pipeline 未启用时不改变原训练入口。

### 当前设计决策
- Stage 2 的最终 expert bundle 必须来自最终 iteration 的 fixed-view validation，而不是随机训练 batch PSNR。
- Stage 3 加载时将再次执行来源 canonical fingerprint 和 minimum PSNR 双重门槛。

### 仍需做什么
- 实现三个不同 topology 专家的冻结容器。
- 实现 Volume-aware Pixel Router、oracle-error distillation 和 anti-starvation。
- Router 训练时禁止 Gaussian optimizer、densification 和专家梯度。

### 运行过哪些测试
- `python -m pytest tests/test_endonerf_presets.py tests/test_complete_endomoeg_experts.py tests/test_endomoeg_bundles.py tests/test_python_compatibility.py -q --tb=short --basetemp .pytest_tmp_stage12`
- 结果：`27 passed`。

### 下一步最小任务
- 新增与 Gaussian index 无关的多专家 pixel-router 模块和损失。

## Update 2026-06-11 Stage 1 and Stage 2 training entry

### 已完成
- Parser 新增新三阶段协议参数：pipeline stage、expert role、bundle 目录、canonical bundle、质量门槛。
- `canonical` stage 只运行 coarse static reconstruction，并输出 `canonical.pth`。
- `expert` stage 跳过重复 coarse，从 canonical bundle 初始化独立 GaussianModel，运行完整 fine optimization，保存 `{role}.pth` complete expert bundle。
- 最终 fixed-view test metrics 现在由训练循环返回，并写入 expert bundle；缺少最终 PSNR 时拒绝保存。
- 测试/保存 iteration 在配置合并后重新计算，避免 preset 覆盖 iterations 后遗漏最终验证。

### 当前设计决策
- 新协议使用 `endomoeg_pipeline_stage`，与旧 `endomoeg_stage` residual-component 路径隔离。
- 所有 bundle 目录和 canonical 路径必须为绝对路径。
- 三个专家作业都从同一 `canonical.pth` 读取静态 Gaussian 状态，但 fine 阶段之后各自保存独立 topology。

### 仍需做什么
- 运行 Stage 1/2 参数与现有训练回归测试。
- 为 cutting/pulling 提供角色专用 preset。
- 实现 Router stage 的多专家加载和冻结渲染。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 运行 parser、bundle、complete expert 和 training-report 定向测试。

## Update 2026-06-11 complete expert verification

### 已完成
- 新增三角色完整输出契约测试。
- 验证 Global 只使用完整 EndoGaussian backbone，不再附加 time-only residual。
- 验证 Local/Contact refinement 均能收到 photometric surrogate gradient。
- 验证 deformation integration、独立参数组、角色 architecture version 和 contact visibility phase。

### 当前设计决策
- Global 的 specialization 主要由独立训练配置中的低频时空分辨率与更强平滑约束实现，结构上不增加弱表达的全局平移 head。
- Local/Contact refinement 零初始化，因此从同一 canonical 启动时不会破坏完整 backbone 的初始输出。
- Contact 的 visibility/lifecycle 只在 contact expert 作业中启用，避免另外两个专家承担无关优化目标。

### 仍需做什么
- Parser 和 preset 增加新三阶段参数。
- 训练入口实现 canonical-only 与 single-expert 模式。
- 将固定视角验证结果写入 expert bundle 并执行质量门槛。

### 运行过哪些测试
- `python -m pytest tests/test_complete_endomoeg_experts.py tests/test_endomoeg_bundles.py tests/test_python_compatibility.py -q --tb=short --basetemp .pytest_tmp_complete_experts`
- 结果：`12 passed`。

### 下一步最小任务
- 接入 Stage 1 canonical bundle 保存和 Stage 2 canonical bundle 初始化。

## Update 2026-06-11 complete expert architecture

### 已完成
- 新增 `CompleteEndoMoeExpert` 与固定单专家 scheduler。
- `tracking_type='endomoeg_expert'` 现在支持 `global/local/contact` 三种角色。
- 三种角色均拥有完整 EndoGaussian HexPlane backbone；Local 在完整 backbone 后增加 tissue-local refinement，Contact 增加 contact/visibility/lifecycle/appearance refinement。
- Global 不再使用 time-only translation MLP，而由完整 EndoGaussian deformation field 独立拟合全场景。

### 当前设计决策
- 专家差异来自结构归纳偏置，而不是共享模型中的强制路由：Global=平滑完整主干，Local=完整主干+局部弹性，Contact=完整主干+接触遮挡。
- 每个 Stage 2 作业只实例化一个角色，因此 canonical Gaussians、HexPlane、refinement 和 densification topology 全部独立。
- 单专家 scheduler 只用于训练日志和 contact visibility loss 开关，不执行任何专家切换。

### 仍需做什么
- 增加三类专家结构、参数组和前向梯度测试。
- Parser 增加角色与 bundle 路径参数。
- Stage 1/2 训练入口接入 canonical bundle 初始化和 complete expert bundle 保存。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 添加 complete expert 定向测试，并修复发现的接口问题。

## Update 2026-06-11 expert bundle contract verification

### 已完成
- 新增 bundle 定向测试，覆盖 canonical round-trip、完整专家状态 round-trip、来源 fingerprint 错配、架构错配和旧 residual checkpoint 拒绝。
- Python 3.7 annotation contract 已纳入新增 EndoMoe runtime 模块。

### 当前设计决策
- 完整 expert bundle 的质量门槛在加载时执行，缺失 PSNR 或低于门槛时拒绝进入 Router stage。
- Bundle 恢复会重建专家自己的 point count、deformation table 和 deformation accumulator，禁止借用另一专家的 topology。

### 仍需做什么
- 新增三种完整专家的结构定义与 preset。
- Stage 1 保存 canonical bundle；Stage 2 从 canonical bundle 初始化并保存完整 expert bundle。
- Router stage 读取三个不同 point count 的冻结专家。

### 运行过哪些测试
- `python -m pytest tests/test_endomoeg_bundles.py tests/test_python_compatibility.py -q --tb=short --basetemp .pytest_tmp_endomoeg_bundle`
- 结果：`5 passed`。

### 下一步最小任务
- 定义 `global/local/contact` 三种完整 deformation profile，并接入现有 `deform_network`。

## Update 2026-06-11 complete expert bundle contract

### 已完成
- `GaussianModel` 新增显式 canonical state 与 complete expert state 的捕获/恢复接口，不再依赖匿名 checkpoint tuple 表达新专家。
- 新增版本化 `endomoeg_canonical_bundle` 与 `endomoeg_complete_expert_bundle`。
- Expert bundle 强制绑定角色、来源 canonical fingerprint、训练后 topology fingerprint、完整 Gaussian/deformation 状态与验证指标。
- 新流程明确拒绝旧的 residual component checkpoint。

### 当前设计决策
- Stage 1 canonical bundle 只保存静态 Gaussian 状态，不携带某个专家的 deformation 参数，允许三类专家从完全相同的静态起点初始化。
- Stage 2 expert bundle 保存专家训练后的独立 canonical cloud、Gaussian topology、appearance、deformation field，而不是孤立网络权重。
- Fingerprint 同时校验来源 canonical 一致性和训练后状态完整性，防止错配专家被静默装配。

### 仍需做什么
- 为 bundle validation、fingerprint 与 Gaussian state round-trip 增加定向测试。
- 定义三类完整专家的 tracking preset 和 Stage 1/2 命令入口。
- 实现多 topology 冻结专家 renderer 与 Router。

### 运行过哪些测试
- 本节完成 3 个文件修改，尚未执行测试。

### 下一步最小任务
- 添加 `tests/test_endomoeg_bundles.py`，先验证 bundle 契约和错配拒绝行为。

## Update 2026-06-11 latest TensorBoard root-cause audit

### 已完成
- 完整解析 `output/last` 的 147 个 scalar 趋势；fine fixed-view PSNR 从 1500 步约 30.55 后长期平台，15000 步为 31.34，最好约 31.46。
- 对照同场景 original baseline 的最终评估 PSNR 37.10，确认约 5.6 dB 差距不是随机 batch 波动。
- 定位阶段切换不连续：global→local 的 L1 约增加 18%，local→full 约增加 55%，同时 motion magnitude 约翻倍。
- 定位 router collapse：joint 阶段 global expert usage 接近 0，local 约 0.78，full 约 0.22。
- 对照 MoE-GS 原论文，确认其核心是“每个完整专家独立训练至可重建全场景，冻结全部专家，再单独训练 router”。

### 当前设计决策
- 当前 continuous 方案不符合 MoE-GS 两阶段训练：canonical Gaussians 在所有阶段持续更新并从约 48k 增密到约 138k，专家面对的输入分布持续漂移。
- 当前 independent component 方案也不闭合：local/full 训练时 canonical Gaussians 会更新，但组件 checkpoint 只保存 expert weights；这些共同收敛的 Gaussian 状态被丢弃，router 在另一份 canonical cloud 上加载专家。
- 当前三个 expert 是共享 HexPlane base 上的 residual adapter，不是 MoE-GS 的完整 dynamic-GS expert；global expert 只产生全点共享的 time-only translation，因此自然被更强的 local expert 淘汰。
- pixel router 在该日志中没有进入真实训练图：其 grad norm/coverage 恒为 0，且缺少 `pixel_router_residual_abs_*` 标签；现有单元测试只覆盖孤立模块，不覆盖真实 renderer 反传。
- 不再通过延长 continuous 训练或微调 loss 权重解决；下一版应优先重构 expert/router 训练协议和 checkpoint 语义。

### 仍需做什么
- 设计并实现独立完整专家容器：每个 expert 拥有自己的 GaussianModel、canonical cloud、deformation field 和 checkpoint。
- 三个专家从同一个 coarse checkpoint 复制初始化，但分别完整动态训练，并设置单专家质量门槛不低于 original baseline。
- router stage 加载并冻结全部完整专家，只优化 per-Gaussian volumetric weights 与 pixel residual router。
- 增加 renderer 级 pixel-router gradient contract test 和运行时 `pixel_router_active`/residual/gradient fail-fast。
- 将 continuous 15000-step residual-MoE 降为历史 ablation，不再作为 README 主推荐流程。

### 运行过哪些测试
- 本轮为日志与代码路径审计，未修改运行时代码，未运行训练测试。
- Codex 独立审核尝试两次：指定模型不受支持；默认模型调用超时，未获得可用审核结论。

### 下一步最小任务
- 先写独立完整 expert checkpoint/bundle 的数据契约与测试，禁止再保存“脱离其 canonical Gaussian 状态”的孤立 residual expert。

## Update 2026-06-10 renderer depth fix pushed

### 已完成
- 修复提交：`747de0f fix(renderer): enforce single-channel depth contract`。
- 已推送到 `origin/main`，本地与远端哈希均为 `747de0fadce58be0995cc5fed78ec7eba3229641`。

### 当前设计决策
- renderer 的公开 depth 契约固定为 `[1,H,W]`；任何 MoE 专家数都不能改变输出通道数。

### 仍需做什么
- 服务器拉取 `747de0f` 后重新运行 test render。
- render 成功后执行 `metrics.py`，无需重新训练现有 checkpoint。

### 运行过哪些测试
- `105 passed`。
- 本地与远端 divergence：`0 0`。

### 下一步最小任务
- 在服务器重新渲染并评估现有模型。

## Update 2026-06-11 Stage 3 source identity and gradient contract

### 已完成
- Frozen expert ensemble 增加绝对数据集路径校验，拒绝将其他场景训练的 expert bundle 静默装配到当前 Router。
- Stage 3 训练入口将当前 `dataset.source_path` 传入 ensemble identity validation。
- 新增 synthetic full-chain 测试，覆盖 Gaussian logits、Gaussian feature MLP、pixel router 到最终重建 loss 的真实梯度链，并覆盖断链 fail-fast。

### 当前设计决策
- Stage 3 的专家身份由 canonical fingerprint、expert fingerprint、point count、architecture version 与绝对 source path 共同约束。
- Router warmup 后三条可学习分支必须同时获得有限且非零的梯度，否则立即停止训练。

### 仍需做什么
- 将 `router_training.py` 纳入 Python 3.7 compatibility 审计。
- 运行 Router Stage 3 定向回归并修复暴露的问题。
- 完成 Router bundle 推理加载与可选受控联合微调。

### 运行过哪些测试
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 补齐 compatibility 列表并运行 Router Stage 3 最小测试集。

## Update 2026-06-11 frozen Router inference assembly

### 已完成
- 修正 `FrozenExpertEnsemble.load()` 的绝对路径校验顺序，避免 `abspath()` 使相对路径检查永久失效。
- Router bundle 显式保存并验证最终推理使用的 `inference_top_k`。
- 新增 `FrozenRouterAssembly`，严格按 Router bundle 内保存的 architecture config 重建 Router、校验精确 expert manifest、加载权重并冻结所有参数。

### 当前设计决策
- 推理不能依赖当前命令行中的 Router hidden dimensions；bundle 内训练时配置是唯一权威来源。
- 推理装配必须同时冻结完整专家和 Router，任何残留 trainable parameter 都视为装配失败。
- bundle 目录、显式 Router bundle 路径和数据集路径都必须是绝对路径。

### 仍需做什么
- 为推理装配增加 identity、路径与 strict state-load 测试。
- 将 Router assembly 接入 `render.py`，使 Router 实验不再加载旧单 Gaussian checkpoint。
- 将新模块纳入 Python 3.7 compatibility 审计。

### 运行过哪些测试
- 上一最小回归：`10 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 增加 frozen Router assembly 定向测试。

## Update 2026-06-11 frozen Router inference verification

### 已完成
- 将 `FrozenRouterAssembly`、严格 Loader 与 bundle-path resolver 纳入 `models.endomoeg` 公共 API。
- 将 `inference.py` 纳入 Python 3.7 AST compatibility 审计。
- 新增测试验证：推理严格使用 bundle 保存的 hidden dimensions、所有 Router 参数冻结、相对 bundle/source 路径被拒绝。

### 当前设计决策
- 推理 Router 的结构由 checkpoint 自描述，不允许当前 preset 静默覆盖。
- 公开 API 与 compatibility test 同步维护，防止服务器 Python 3.7 导入阶段才暴露错误。

### 仍需做什么
- 运行 inference/Router/compatibility 定向回归。
- 修改 `render.py`，按 `endomoeg_pipeline_stage='router'` 选择 frozen ensemble 渲染路径。
- 为渲染入口增加定向测试。

### 运行过哪些测试
- 推理装配实现前的基础回归：`15 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 frozen Router assembly 最小测试集。

## Update 2026-06-11 Router render entry

### 已完成
- 新增 `endomoeg_router_bundle` 参数，并在训练参数验证中强制显式路径为绝对路径。
- `render.py` 根据 `endomoeg_pipeline_stage='router'` 加载 frozen Router assembly。
- Router 渲染逐帧执行三套完整专家、volumetric routing feature splat 与 pixel-space 融合；不再加载或渲染旧单 Gaussian checkpoint。
- 显式 `--iteration` 必须与 Router bundle iteration 一致，避免输出目录标签与实际权重不一致。

### 当前设计决策
- Router bundle 的 iteration、top-k、hidden architecture 与 expert manifest 是推理唯一事实来源。
- 普通 EndoGaussian 渲染路径保持默认行为；只有显式 Router pipeline 才进入多专家融合。

### 仍需做什么
- 增加 `render_sets()` Router 分流的集成测试。
- 优化 Router 场景初始化，避免仅为相机加载而创建无用 Gaussian cloud。
- 运行 render/argument/compatibility 回归。

### 运行过哪些测试
- frozen Router assembly 回归：`12 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 增加 Router render entry 定向测试。

## Update 2026-06-11 Router render test and import-cycle fix

### 已完成
- 新增 Router render entry 集成测试，覆盖 frozen assembly 选择、bundle iteration、top-k 与实际 ensemble render 回调。
- 首次运行暴露 `scene.gaussian_model -> scene.deformation -> models.endomoeg -> ensemble -> scene.gaussian_model` 循环导入。
- 将 `GaussianModel` 重依赖移动到 `FrozenExpertEnsemble.load()` 局部导入，解除 package 初始化环，而不是在测试中规避 import。

### 当前设计决策
- EndoMoe package 的模块级导入不得反向依赖正在初始化的 Scene runtime。
- 只有真正加载 expert bundle 时才导入 `GaussianModel`，保持 deformation 子模块可独立初始化。

### 仍需做什么
- 重新运行 render/Router/compatibility 回归。
- 若入口通过，增加 camera-only Scene 初始化，删除 Router 渲染中的无用 Gaussian cloud 创建。
- 更新 Stage 3 状态并继续受控 joint 设计。

### 运行过哪些测试
- 首次 render 回归在 collection 阶段失败，已定位并修复循环导入根因。

### 下一步最小任务
- 重新运行相同最小测试集确认 import cycle 消失。

## Update 2026-06-11 camera-only Router scene

### 已完成
- `Scene` 新增显式 `initialize_gaussians` 契约；camera-only 模式允许 `gaussians=None`，但仍完整加载数据集与 train/test/video cameras。
- Router 渲染入口不再创建无用 `GaussianModel`，也不会从输出目录误加载历史单模型 point cloud。
- 集成测试新增 fail-fast：Router render 若实例化单 GaussianModel 立即失败。

### 当前设计决策
- Router 推理的场景对象只负责数据集与相机；所有几何、外观、deformation 状态均来自三个 identity-bound expert bundles。
- 普通训练和普通渲染仍默认 `initialize_gaussians=True`，不改变旧路径行为。

### 仍需做什么
- 运行 camera-only Scene 与 Router render 回归。
- 确认 Python 3.7 compatibility 与现有 Scene 初始化测试不受影响。
- 完成 Stage 4 受控联合微调协议或明确将其保持为可选实验。

### 运行过哪些测试
- 上一 Router render 回归：`14 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 render、Scene 相关最小回归。

## Update 2026-06-11 strict bundle path boundary

### 已完成
- 修正 expert/canonical bundle I/O 的绝对路径检查顺序，直接 API 不再接受相对路径。
- Router bundle save/load 使用统一的严格绝对路径 helper。
- 新增 canonical、expert 与 Router bundle 相对路径拒绝测试。

### 当前设计决策
- “配置层已校验”不能替代持久化边界自身校验；所有 bundle I/O API 必须独立 fail-fast。
- Stage 4 输出不会依赖进程当前工作目录，防止 checkpoint 被写到不可追踪位置。

### 仍需做什么
- 运行 bundle、render、compatibility 定向回归。
- 实现 Stage 4 deformation-only + Router controlled joint optimization。
- 为 joint checkpoint 定义不可覆盖 Stage 2/3 原始 bundle 的输出协议。

### 运行过哪些测试
- camera-only Router 场景回归：`29 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 验证 strict bundle I/O 后开始 joint optimizer contract。

## Update 2026-06-11 full expert-state identity

### 已完成
- Expert bundle 升级到 version 2，新增覆盖 canonical、deformation state、deformation accumulator、tracking architecture 与 spatial context 的完整 fingerprint。
- Router bundle 升级到 version 2，其 expert manifest 改为绑定完整 expert-state fingerprint，而非仅绑定 canonical topology。
- 新增 deformation tensor 被篡改时 expert bundle validation 必须失败的测试。

### 当前设计决策
- `trained_canonical_fingerprint` 只用于描述 topology/appearance；`expert_state_fingerprint` 才是动态专家身份。
- Router 装配与未来 joint checkpoint 必须绑定完整动态状态，禁止同 topology、不同 deformation 权重的静默替换。
- 旧 version 1 bundle 明确不兼容，必须按新代码重新训练/导出，避免虚假的身份安全。

### 仍需做什么
- 更新 Router identity 测试与公共 API 导出。
- 运行 bundle/Router 回归并验证 deformation mutation 被 Router 拒绝。
- 在完整状态身份契约上实现 Stage 4 joint 输出。

### 运行过哪些测试
- strict bundle path 回归：`21 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 更新 Router full-state identity 测试并运行最小回归。

## Update 2026-06-11 full-state Router manifest verification

### 已完成
- Router identity 测试改为替换完整 expert-state fingerprint，并验证装配被拒绝。
- `expert_fingerprint` 纳入 `models.endomoeg` 公共 API。
- Router bundle validation 显式检查每个 role 的完整 manifest 字段，损坏 bundle 返回可诊断错误而非 `KeyError`。

### 当前设计决策
- Router manifest 同时保留 canonical fingerprint 供 topology 审计，但匹配判据使用完整 expert-state fingerprint。
- Bundle schema 缺字段属于格式错误，必须在加载权重前失败。

### 仍需做什么
- 运行 version 2 expert/Router bundle 回归。
- 实现 joint trainable-parameter contract 与 anchor loss。
- 定义 joint 输出目录和更新后 expert/router bundle 保存顺序。

### 运行过哪些测试
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 bundle 与 Router identity 最小测试集。

## Update 2026-06-11 controlled joint training core

### 已完成
- 新增 Stage 4 参数：独立 Router LR、Global deformation LR、Local/Contact refinement LR、anchor、gradient clip、稀疏切换、PSNR 退化阈值与独立输出目录。
- Joint 输出目录默认 `<bundle_dir>/joint`，并明确禁止覆盖 Stage 2/3 原始 bundle。
- 新增 `train_controlled_joint()`：Router 全量小步更新；Global 仅更新完整 deformation；Local/Contact 仅更新各自 refinement。
- canonical geometry、appearance、opacity、scale、rotation 与 topology 始终冻结；不运行 densification/pruning。
- 加入参数锚定、三专家梯度契约、Router 梯度契约、gradient clipping、ensemble 与单专家双重 fixed-view quality gate。
- Joint 仅在质量门通过后保存三份新 expert version-2 bundles 与绑定其完整状态的新 Router bundle。

### 当前设计决策
- 主方法仍是 Static → Independent Experts → Frozen Router 三阶段；Joint 是第四阶段可选保守精修，不是必需阶段。
- Local/Contact 的完整 backbone 保持 Stage 2 能力，只允许结构归纳偏置对应的 refinement 适配融合上下文。
- Global 无独立 refinement，因此允许完整 deformation 以更低 LR 微调。
- 任一单专家或整体 Router 超过允许 PSNR 退化阈值时拒绝保存，防止 joint 通过牺牲专家完整性换取局部训练 loss。

### 仍需做什么
- 将 joint stage 接入 `train.py` camera-only 入口。
- 增加 trainable contract、anchor、quality gate 与保存身份测试。
- 将 `joint_training.py` 纳入 Python 3.7 compatibility。

### 运行过哪些测试
- full-state identity 回归：`21 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 为 controlled joint 核心增加纯 CPU 定向测试。

## Update 2026-06-11 joint entry and trainable contract tests

### 已完成
- Stage 3 Router device 不再依赖 dummy `scene.gaussians`，直接选择实际 runtime device。
- `train.py` 对 Router/Joint 使用 camera-only Scene，不实例化或初始化无用 Gaussian cloud。
- `endomoeg_pipeline_stage='joint'` 已接入 controlled joint trainer。
- 新增纯 CPU 测试覆盖：canonical 永久冻结、Global deformation 可训练、Local/Contact 仅 refinement 可训练、anchor drift 与整体/单专家 quality gate。

### 当前设计决策
- Router/Joint 的 Scene 只提供 cameras；全部可渲染状态必须来自 expert bundles。
- Joint optimizer 恰好包含三组 Router 参数与三组受控 expert 参数，任何额外 trainable state 都违反契约。

### 仍需做什么
- 运行 joint/train entry 定向回归并修复接口问题。
- 将 `joint_training.py` 纳入 Python 3.7 compatibility。
- 增加 joint output bundle identity 保存测试。

### 运行过哪些测试
- joint 入口接入前：`py_compile` passed，既有核心回归 `18 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 joint core、preset、Router 与 compatibility 最小测试集。

## Update 2026-06-11 joint compatibility and output identity tests

### 已完成
- `joint_training.py` 纳入 Python 3.7 AST compatibility 审计。
- 参数测试验证 Joint 默认输出到 `<bundle_dir>/joint`、禁止覆盖父 bundle、拒绝相对 Router bundle 路径。
- Joint 保存测试验证三份更新 expert payload 先替换 ensemble manifest，再构建绑定新 full-state fingerprints 的 Router bundle。
- 保存完成后 Router 与专家全部重新冻结。

### 当前设计决策
- Joint checkpoint 是新的不可变 assembly lineage，父 expert/router bundle 只读保留。
- 新 Router 必须绑定 Joint 后的 expert-state fingerprints；不能继续引用 Stage 2 expert identity。

### 仍需做什么
- 运行 joint save/compatibility/preset 回归。
- 检查实际 `build_expert_bundle` version-2 保存与 Loader round-trip。
- 补齐角色专用 presets 与 README 完整命令。

### 运行过哪些测试
- joint entry 回归：`30 passed`。
- 本次三个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 Joint 全部定向测试。

## Update 2026-06-11 stage-neutral EndoNeRF presets

### 已完成
- cutting/pulling 主 preset 从旧 `cams_gs_moe` continuous residual 路径切换为 stage-neutral complete-expert pipeline。
- Preset 使用 1000-step static canonical 与 9000-step independent expert 默认协议；Router/Joint 通过 `endomoeg_stage_iterations` 单独指定轮数。
- Preset 明确包含 complete expert motion caps、volume-aware Router、oracle/anti-starvation、gradient warmup 与 controlled joint 超参数。
- Joint 保存测试桩补齐 version-2 expert identity 并通过。

### 当前设计决策
- 同一场景 preset 只定义稳定的模型/优化超参数；`canonical/expert/router/joint` 由命令行显式指定，避免配置文件与 CLI stage 互相覆盖。
- 旧 `tracking_type='cams_gs_moe'` 不再是 `cutting_endomoeg.py` / `pulling_endomoeg.py` 的主入口。

### 仍需做什么
- 重写 README 主流程与完整命令。
- 运行 preset key/merge/compatibility 回归。
- 补充 Stage 2 三角色结构和 Stage 3/4 TensorBoard 监控项。

### 运行过哪些测试
- Joint 完整定向回归：`39 passed`。
- 两个 preset 修改后尚未运行测试。

### 下一步最小任务
- 运行 EndoNeRF preset 最小回归。

## Update 2026-06-11 preset CLI precedence contract

### 已完成
- EndoMoe preset 测试从旧 `cams_gs_moe` 连续训练断言升级为 complete-expert pipeline 参数断言。
- cutting/pulling preset 移除显式空 `endomoeg_pipeline_stage`，避免 mmcv config merge 覆盖命令行 stage。
- CLI 的 `--endomoeg_pipeline_stage`、`--endomoeg_expert_role` 与绝对 bundle 路径现在保持为运行时权威输入。

### 当前设计决策
- Stage-neutral 的含义是“不在 config 中声明 stage”，而不是把 stage 写成空字符串。
- Preset 可以固定模型结构和学习率，但不能覆盖一次具体作业的 pipeline role/stage。

### 仍需做什么
- 重写 README，给出不会被 config 覆盖的完整命令。
- 增加 CLI + preset 合并后 stage 保留的回归测试。
- 运行主流程较宽回归。

### 运行过哪些测试
- EndoNeRF preset/compatibility：`17 passed`。

### 下一步最小任务
- 增加 CLI stage precedence 测试后更新 README。

## Update 2026-06-11 Joint render assembly selection

### 已完成
- `render.py` 将 `router` 与 `joint` 都识别为多专家 assembly 推理，不再让 Joint 落回单 Gaussian 渲染。
- Router 阶段加载 `<endomoeg_bundle_dir>/router.pth` 或显式 Router bundle。
- Joint 阶段强制加载 `<endomoeg_joint_output_dir>/router.pth` 与同目录三专家，避免误用父 Stage 3 assembly。
- 新增 Joint render loader 路径测试。

### 当前设计决策
- Joint 输出目录本身就是新的完整 bundle directory；推理时不能混用父 Router 与 Joint experts。
- Router/Joint 均使用 camera-only Scene 和相同的 volume-aware pixel fusion。

### 仍需做什么
- 运行 Router/Joint render 回归。
- 增加 CLI stage precedence 测试。
- 重写 README 与服务器实验命令。

### 运行过哪些测试
- 本次两个文件修改后尚未运行测试。

### 下一步最小任务
- 运行 render 定向测试。

## Update 2026-06-11 pipeline documentation and CLI precedence

### 已完成
- 新增 preset merge 测试，确认 config 不覆盖 CLI 的 pipeline stage 与 expert role。
- README 完整重写为 Static Canonical → Independent Global/Local/Contact Experts → Frozen Router → Optional Controlled Joint。
- README 提供 `/root/3DGS` 数据路径与 `/root/autodl-tmp` 输出路径的完整 cutting/pulling 服务器命令。
- 文档覆盖 version-2 full-state identity、TensorBoard 指标、Router/Joint 渲染、metrics、输出布局、质量门与 ablation。
- 删除旧 continuous/component checkpoint 工作流作为推荐主流程，仅保留 legacy ablation 说明。

### 当前设计决策
- README 中所有关键路径均使用绝对路径。
- `MIN_EXPERT_PSNR` 必须依据原始 EndoGaussian fixed-view test PSNR 设置，而不是使用训练 batch PSNR。
- Joint render 直接从保存的 `cfg_args` 定位新 Joint assembly。

### 仍需做什么
- 运行 README/preset/render/compatibility 回归。
- 检查命令行参数实际存在且 merge 后值正确。
- 执行更宽测试、diff check 与独立代码审核。

### 运行过哪些测试
- Router/Joint render：`4 passed`。
- README 重写后尚未运行测试。

### 下一步最小任务
- 运行文档相关 preset 与命令契约测试。

## Update 2026-06-11 full regression

### 已完成
- 修正 smoke fake rasterizer，使其 depth/render 尺寸与声明的 `2×2` camera 一致；运行时单通道 depth 严格契约保持不变。
- 完整 `tests/` 回归全部通过。
- README、preset、Router/Joint render、bundle identity、controlled joint 与 Python 3.7 compatibility 均纳入回归。

### 当前设计决策
- 不放宽 depth canonicalizer 接受错误空间尺寸；测试替身必须模拟真实 rasterizer 契约。

### 仍需做什么
- 清理 pytest 临时目录。
- 执行独立 Codex diff 审核。
- 根据审核结果修正后做最终 `diff --check`。

### 运行过哪些测试
- `python -m pytest tests -q --tb=short --basetemp .pytest_tmp_full_endomoeg`
- 结果：`148 passed, 2 warnings`。

### 下一步最小任务
- 独立审核当前 diff 的理论与工程一致性。

## Update 2026-06-11 blocking identity-initialization audit fix

### 已完成
- 本地审计发现 `reset_backbone_to_identity()` 仅覆盖旧 `cams_gs_moe`，新 `endomoeg_expert` 的 HexPlane 输出头仍为随机初始化。
- 修正 complete expert backbone，使 global/local/contact 的 position/scale/rotation/opacity 输出头末层全部零初始化。
- 新增三角色端到端启动连续性测试：动态 fine 第一步的 means/scales/rotations/opacity 必须与 canonical 输入完全一致。

### 当前设计决策
- Stage 2 的完整专家可以拥有独立随机特征编码与隐藏层，但所有直接改变 Gaussian 状态的输出头必须从严格零残差开始。
- 该约束是防止 coarse→dynamic PSNR 从二十多直接跌到约 9 的核心运行时保证。

### 仍需做什么
- 运行 complete expert 与全量回归。
- 继续本地审计 Router/Joint 数值契约。
- Codex 独立审核因账户用量限制未完成，最终说明中必须如实记录。

### 运行过哪些测试
- 修复前全量回归：`148 passed`，但该初始化语义此前没有对应测试。

### 下一步最小任务
- 验证三角色 identity start 并重新跑全量测试。

## Update 2026-06-11 strict quality and manifest audit

### 已完成
- Router manifest validation 现在逐项核对 full-state fingerprint、canonical fingerprint、point count、tracking architecture 与 expert validation PSNR。
- Router point-count mapping 必须保持 `global/local/contact` 固定顺序。
- Router/Joint 训练强制要求显式正数 `endomoeg_min_expert_psnr`，不能再以默认 0 绕过单专家质量门。
- 参数测试覆盖缺失质量阈值时 fail-fast。

### 当前设计决策
- MoE-GS 式 Router 只允许建立在达到场景 baseline 门槛的完整专家上；质量门不是可选日志字段。
- Manifest 的冗余元数据也必须一致，避免 checkpoint 手工拼装后只靠单个 hash 字段通过。

### 仍需做什么
- 运行 quality gate、bundle identity 与全量回归。
- 为 Router manifest architecture/PSNR tampering 增加直接测试。
- 清理临时目录并完成最终审计。

### 运行过哪些测试
- identity-start 定向：`10 passed`。
- identity 修复后全量：`151 passed, 2 warnings`。

### 下一步最小任务
- 增加 Router manifest 元数据篡改测试并回归。

## Update 2026-06-11 complete expert time-encoder training fix

### 已完成
- 审计发现 `tracking_time_encoder` 不在 CompleteExpertScheduler 白名单，Stage 2 中其 LR 会被 phase 置零。
- 三角色 scheduler 现在显式训练 `tracking_time_encoder`；Local/Contact 同时训练 role-specific refinement。
- 测试验证 time encoder 与 refinement 的 group-trainable 契约。

### 当前设计决策
- Global/Local/Contact 都需要可学习时间编码器驱动完整 HexPlane deformation。
- Base deformation/grid 继续通过通用 always-trainable 规则训练；Local/Contact refinement 使用显式白名单训练。

### 仍需做什么
- 运行 complete expert LR/identity 与全量回归。
- 检查 Joint 是否应同时允许 Global time encoder（当前完整 deformation 已包含）。
- 最终清理与 diff 审查。

### 运行过哪些测试
- manifest/quality 定向：`35 passed`。
- 上一全量：`151 passed, 2 warnings`。

### 下一步最小任务
- 验证时间编码器训练契约并重新跑全量。

## Update 2026-06-11 complete expert motion regularization wiring

### 已完成
- 审计确认新 complete experts 未输出旧 tracking loss 所需的 motion norm keys，导致 motion magnitude regularization 实际未生效。
- 所有角色现在输出 normalized backbone `global_motion_norm`。
- Local 额外输出 refinement `local_motion_norm`；Contact 额外输出 refinement `cut_graph_motion_norm`。
- 归一化统一使用 scene scale，保持不同 EndoNeRF 场景的量纲稳定。

### 当前设计决策
- 完整 backbone 承担全场动态，因此始终使用较强 global magnitude regularization。
- Local/Contact refinement 使用较弱角色专用 magnitude regularization，允许局部大形变但抑制无界漂移。

### 仍需做什么
- 运行 complete expert/tracking loss 与全量回归。
- 检查 TensorBoard 是否自动记录新增 motion metrics（现有 tracking scalar logger 会记录）。
- 最终清理临时文件。

### 运行过哪些测试
- time-encoder 定向：`27 passed`。
- time-encoder 修复后全量：`151 passed, 2 warnings`。

### 下一步最小任务
- 验证 motion norm 输出与训练回归。

## Update 2026-06-11 straight-through sparse Router

### 已完成
- 审计发现 hard top-k 在 sparse 阶段切断未选专家梯度，contact 掉出 top-2 后 anti-starvation 无法恢复。
- Router 改为 straight-through top-k：forward 保持严格 top-k 稀疏，backward 使用 dense soft weights 梯度。
- 新增测试验证 top-1 forward 仅一个非零专家，但三个 logits 均收到梯度。
- `render.py` Router 依赖改为惰性导入，消除普通渲染与测试 mock 的顺序相关 import 脆弱性。

### 当前设计决策
- 稀疏路由只改变前向计算预算与组合支持集，不应彻底剥夺未选专家的恢复梯度。
- Inference 无梯度，仍得到严格 top-k 权重；training 使用 straight-through estimator。

### 仍需做什么
- 运行 Router、motion wiring 与全量回归。
- 清理 pytest 临时目录。
- 最终 `git diff --check` 与状态汇总。

### 运行过哪些测试
- lazy render 定向：`105 passed`。
- lazy render 修复后全量：`151 passed, 2 warnings`。

### 下一步最小任务
- 验证 straight-through Router 并完成最终回归。

## Update 2026-06-11 complete EndoMoeGaussian implementation ready

### 已完成
- 完成 Static Canonical → Independent Complete Experts → Frozen Volume-aware Router → Optional Controlled Joint 全流水线。
- 最终审计额外修复三个直接影响训练效果的问题：
  - complete expert backbone 未 identity 初始化；
  - complete expert time encoder 被 phase 冻结；
  - sparse top-k 切断未选专家梯度。
- Motion magnitude regularization 已按 backbone/global、local refinement、contact refinement 真实接入。
- Router/Joint 质量门、full-state identity、camera-only train/render、Joint 新 assembly 与 README 命令全部闭环。
- `py_compile` 与 `git diff --check` 通过。

### 当前设计决策
- 主方法是三阶段：静态 canonical、三个独立完整专家、冻结专家训练 Router。
- Controlled Joint 是可选第四阶段，采用严格白名单、anchor 与整体/单专家双重质量门。
- Expert/Router bundle version 2 不兼容旧 residual component checkpoint；必须重新训练生成。

### 仍需做什么
- 用户在服务器按 README 依次执行 Stage 1–3，并提供各 expert 与 Router TensorBoard/固定视角指标。
- 训练前将 `MIN_EXPERT_PSNR` 设置为原始 EndoGaussian fixed-view test PSNR 减去小容差。
- 如果 Stage 3 已优于 baseline，再决定是否执行可选 Joint。
- 尚未 commit/push；等待用户确认后执行。

### 运行过哪些测试
- 最终定向回归：`114 passed`。
- 最终完整回归：`152 passed, 2 warnings`。
- Python runtime modules `py_compile`：passed。
- `git diff --check`：passed，仅存在 LF/CRLF 提示。
- 独立 Codex 审核调用失败：账户 usage limit，未获得外部审核结论；本地只读审计已完成并发现/修复上述三项问题。

### 下一步最小任务
- 用户确认后 commit 并 push，随后在服务器运行 README 的 Stage 1 canonical 命令。
