# Workflow, Rules & Goals

Tài liệu hướng dẫn cộng tác cho dự án vaipe-contextual-pill-recognition.

## 1. Mục tiêu

Vượt qua bài báo PIKA (arXiv:2208.02432, ACIIDS 2022) trên cùng giao thức đánh giá.

Số liệu paper (76 lớp, dataset tác giả):
- ResNet-50 baseline F1 = 0.5215
- ResNet-50 PIKA F1 = 0.8101

Dataset của repo: VAIPE clean_paper_like_split_v2 — 108 lớp, train 23189 / val 4974 / test 4550.

Đích ngắn hạn: vượt 0.5215 trên subset paper-like (đã đạt 0.544 trên 73 lớp với M17 v2).

Đích dài hạn: chạm 0.81 trên subset 38 lớp (hiện 0.659) qua ensemble.

## 2. Quy trình cộng tác

```
USER ↔ AI Cursor               AI → GitHub                  USER → Colab
1. User mô tả ý tưởng     2. AI commit code + push     3. User git pull
4. User chạy cell Colab                                 5. User gửi kết quả
6. AI phân tích, đề xuất phase tiếp                          
```

Quy tắc:
- AI sửa code, commit và push trực tiếp.
- User chỉ chạy cell train/eval trên Colab GPU.
- Mỗi Phase hoặc mỗi Run là một commit riêng để dễ rollback.
- Luôn git pull trên Colab trước khi chạy cell.

## 3. Quy ước path Drive

| Loại | Path |
|---|---|
| Dataset Kaggle | /root/.cache/kagglehub/... (Colab, không lên Drive) |
| Split CSV | /content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2/ |
| Audit | /content/drive/MyDrive/model/audit_pika_protocol_v1/ |
| Graph PPMI+SVD | model/M17_faithful_pika/graph_artifacts/ |
| Graph walk | model/M17_faithful_pika/walk_graph_artifacts/ |
| M17 train | model/M17_faithful_pika/train_run_<config>/ |
| M17 eval | model/M17_faithful_pika/test_eval_<config>/ |
| Candidate eval | audit_pika_protocol_v1/candidate_eval_<config>/ |
| Visual stage1 | model/M23_stage1_visual_pretraining/visual_init_resnet50/ |

## 4. Metrics

| Metric | Khi nào dùng |
|---|---|
| macro_f1_present | Báo cáo nội bộ, lớp xuất hiện trong y_true |
| macro_f1_all_candidate | Chính, dùng so paper trên subset |
| accuracy, weighted_f1 | Phụ trợ |

## 5. Decision gates

| Gate | Điều kiện | Hành động |
|---|---|---|
| 1 | Val F1 >= 0.42 trên M17 paper-faithful | Triển khai Phase 2 Run B/C/D |
| 2 | Test F1 Present >= 0.50 | Bắt đầu Phase 3 ensemble |
| 3 | Subset 38 lớp F1 >= 0.75 | Phase 4 writeup |

## 6. Presets comparison_track

| Preset | Backbone | Epoch | LR | Status |
|---|---|---|---|---|
| default | EfficientNetV2-S | 15 | 1e-4 | Notebook baseline_new.ipynb |
| paper_faithful | ResNet-50 | 15 | 1e-4 | Phase 1, undertrained, Val 0.313 |
| paper_faithful_v2 | ResNet-50 | 25 | 5e-4 | Phase 2, Val 0.4975, Test 0.431 |

Flags Phase 2:
- --visual_init_ckpt PATH (Run D)
- --graph_artifacts_dir <walk_graph> (Run B)

## 7. Common mistakes

1. Sai path candidate CSV: dùng audit_candidate_subsets.csv (sai) thay vì audit_candidate_benchmarks.csv (đúng).
2. Sai path split: /MyDrive/data/clean_split (sai) vs /MyDrive/vaipe_splits/clean_paper_like_split_v2 (đúng).
3. tee Colab lỗi vì dir chưa tồn tại, os.makedirs(exist_ok=True) trước.
4. PATH chưa nạp git/gh sau winget, refresh bằng [Environment]::GetEnvironmentVariable.
5. %%bash mất context cd, dùng %cd + !python ... thay thế.
6. Colab hết runtime, script luôn save best ckpt mỗi epoch.
7. ResNet-50 cần lr cao hơn EffNet (5e-4 thay 1e-4) để hội tụ.

## 8. Status hiện tại

| Phase | Run | Val F1 | Test F1 Present | Test F1 38-class | Commit |
|---|---|---|---|---|---|
| 1 | M17 ResNet-50 v1 | 0.313 | 0.322 | 0.491 | 3332ede |
| 2 | M17 v2 paper_faithful_v2 | 0.4975 | 0.431 | 0.659 | 88a5663 |
| 2 | Run D visual init | đang chạy | - | - | 4941681 |
| 2 | Run B walk graph | đang chạy | - | - | 4941681 |
| 3 | Ensemble paper-track | - | - | - | TBD |

## 9. Next steps

1. Chạy hết Run D và Run B, so sánh với baseline v2.
2. Nếu một trong hai vượt baseline, commit Run B+D kết hợp.
3. Phase 3: M19/M21 ResNet-50 với heavy augmentation.
4. Phase 3.5: M26-paper calibrated ensemble (val-only tuning).
5. Phase 4: writeup với 3 subset 108/73/38.

## 10. Reference docs

- README.md — overview
- PROGRESS_AND_PLAN.md — chi tiết roadmap
- WORKFLOW.md — file này
- baseline_new.ipynb — notebook gốc M17 đến M30

## 11. Quick commands

Cursor Windows:
```
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
cd C:\Users\haban\Documents\vaipe-contextual-pill-recognition
git log -5 --oneline
```

## 12. AI prompt cheat-sheet

| Mục tiêu | Câu nói với AI |
|---|---|
| Commit phase mới | Commit Phase N Run X |
| Hỏi kế hoạch | Kế hoạch tiếp theo sau Run X? |
| Phân tích kết quả | (paste log Colab) |
| Sửa lỗi cell | (paste error) |
| Update doc | Cập nhật PROGRESS_AND_PLAN.md với kết quả mới |

## 13. Commit history

- 3332ede — feat(phase1): paper-faithful M17 ResNet-50, PMKG graph fallback
- 88a5663 — feat(phase2): paper_faithful_v2 preset (25 epoch, lr=5e-4)
- 5e39283 — docs(plan): progress report and roadmap
- 4941681 — feat(phase2): visual stage1 init (Run D) and walk graph (Run B)