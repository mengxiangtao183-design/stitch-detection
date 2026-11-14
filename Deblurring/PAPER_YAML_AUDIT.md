# 去模糊 YAML 清理记录（2026-11-14）

## 当前论文相关配置

- 论文主模型：MBDNet-P+，实际结果运行标识 MultiSourceV2_MBDNetPPlusR12_S30_S42。训练参数快照已复制到 `NAFNet/experiment_configs/mbdnet_p_plus_r12_s30_s42_run_config.json`；原始运行记录位于 `NAFNet/checkpoints/MultiSourceV2_MBDNetPPlusR12_S30_S42/run_config.json`，并非 YAML。
- 对比模型 MPRNet：Deblurring/training_mprnet_original_multisource_synblur_v2_s30.yml，保留在活动目录。
- 其他九个对比模型使用 deblur_comparison_suite 的 Python 训练入口与运行记录；第三方仓库内的官方 YAML 模板没有作为本项目统一训练入口。
- 当前未发现 MBDNet-P+ 的正式结构消融 YAML。旧 MPRNet 改型与训练探索不能写作 MBDNet-P+ 消融。

## 本次处理

- 从 Deblurring 顶层归档 51 个非当前论文配置；现只留 1 个 MPRNet 对比 YAML。
- 本地归档路径：_archive/deblurring_nonpaper_yaml_2026-11-14/。manifest.json 含原文件名、SHA-256 与修正的历史入口清单。归档由 `.gitignore` 排除，不上传 GitHub；历史入口需要该本地归档才能运行，公开仓库只保留正式配置。需要在新克隆中恢复旧实验时，也可从清理前的 Git 提交 `db7dda8` 提取原 YAML。
- 旧 Python/PowerShell 入口改指向归档配置；Deblurring/train.py 无参数默认配置改为保留的 MPRNet 对比配置。
- 未移动 NAFNet/options 与 deblur_comparison_suite/third_party 的上游模板，以保持第三方源码完整。这些模板不应列为论文主模型配置。

## 归档分类

### 旧 MPRNet 结构与训练探索（22）

- training_lmprnet_m_edge_kd.yml
- training_mprnet_baseline_fast100.yml
- training_mprnet_edge_fast100.yml
- training_mprnet_edge_ft.yml
- training_mprnet_edge_tristripe_cons000_ft10.yml
- training_mprnet_edge_tristripe_cons010_ft10.yml
- training_mprnet_edge_tristripe_fast100.yml
- training_mprnet_edge_tristripe_frequency_fast100.yml
- training_mprnet_edge_tristripe_frequency_v2_fast100.yml
- training_mprnet_edge_tristripe_frequency_v3_bandgate_fast100.yml
- training_mprnet_edge_tristripe_frequency_v4_disagreement_fast100.yml
- training_mprnet_edge_tristripe_pairaligned_fast100.yml
- training_mprnet_edge_tristripe_pairaligned_irr_fast100.yml
- training_mprnet_edge_tristripe_stage2transformer_fast100.yml
- training_mprnet_edge_tristripe_stripetransformer_fast100.yml
- training_mprnet_m2_freqcons_dualdomain_ors_fast100.yml
- training_mprnet_m2_freqcons_gated_csff_fast100.yml
- training_mprnet_m2_frequency_consistency_fast100.yml
- training_mprnet_m2_generalization_pair_fast100.yml
- training_mprnet_m5v3_generalization_pair_fast100.yml
- training_mprnet_tripaired_sgors_fast100.yml
- training_mprnet_tristripe_pairaligned_gatedcsff_fast100.yml

### 诊断配置（2）

- diagnosis_frequency_causal_m5v3.yml
- diagnosis_frequency_hybrid_m2_m5v3.yml

### 历史默认配置（1）

- training.yml

### 其他早期候选网络与融合实验（26）

- training_evidence_canvas_fast100.yml
- training_hmtnet_fast100.yml
- training_hmtnet_full150.yml
- training_quality_fusion_fast100.yml
- training_quality_fusion_v2_fast100.yml
- training_tfr_net_fast100_scratch.yml
- training_tfr_net_full150_scratch.yml
- training_tfr_net_v2_controlled_synblur_v1_fast15_memsafe_nokd.yml
- training_tfr_net_v2_controlled_synblur_v1_s30_nokd.yml
- training_tfr_net_v2_fast100_tptteacher.yml
- training_tfr_net_v2_lite_controlled_synblur_v1_fast15_nokd.yml
- training_tpt_net_bandgate_fast100.yml
- training_tpt_net_bandgate_stage2_tmt_fast100.yml
- training_tpt_net_bandgate_stage2tmt_hsgorsv2_fast100.yml
- training_tpt_net_bandgate_stage2tmt_stage3trr_fast100.yml
- training_tpt_net_bandgate_strd_fast100.yml
- training_tpt_net_stam_fast100.yml
- training_tpt_net_tepr_pure_fast100.yml
- training_tpt_net_tepr_transition_fast100.yml
- training_tptrnet_bandgate_fast100.yml
- training_tptrnet_tcfm_bandgate_fast100.yml
- training_tptrnet_tcfm_bandgate_full150.yml
- training_tptrnet_tcp_scratch_full150.yml
- training_tristripe_ors_edge_kd.yml
- training_tristripe_ors_nokd.yml
- training_tristripe_ors_nokd_fast100.yml
