# RepresentativeSelection_ — 獨立式正常影像代表性選樣（v1）

**SubspaceAD + AnomalyVFM 專案的獨立離線選樣工具。** 不會 Import、執行或修改 Server 上的 `SubspaceAD_` 或 `AnomalyVFM_` 兩個專案。它只使用設定檔指定的外部 DINOv2 原始 Repository／模型權重與 AnomalyVFM adapted checkpoint。`offline_models/dinov2_offline.py` 是使用者提供的離線載入程式；本專案包含獨立的 DoRA 相容實作（`peft_local/`），不在執行時依賴外部 AnomalyVFM 的 Python 程式。

## 目前已實作

- 依 Class 分開處理：每個 Class 各有獨立訓練集合、候選池、R_good、Coverage 與 PCA。
- 原始 DINOv2-L/14-Reg + A1 AnomalyVFM DoRA rank 64；最後一層 `x_norm_patchtokens`；672x672 → 約 2304 x 1024。
- Feature Cache：模型每張影像只需要萃取一次特徵，可跨實驗重用（需保持模型和前處理一致）。
- Stage 1：完整 Patch、shifted cosine、逐影像等權的 Mean Coverage Gain，逐張 Greedy 提案，Self/Other Gain 診斷。
- H100：GPU Chunked Matrix Multiplication / Max Reduction；有容量估算的預計算 Max-Similarity Cache、CPU/Disk Memory Mapping、可選 Stream。
- PCA：浮點 64 位統計量，逐張可加總的 `n, sum(x), sum(xxᵀ)`；Fixed Dimension 或 Explained Variance（EV）。
- Stage 2：在**相同獨立 R_good** 上比較 PCA Before/After，支援 Mean、P90 與 High-error Ratio 改善指標（預設 Mean）；`accept/reject/pending`。
- Random 或 Coverage Bootstrap、不同 Proposal Baseline（Random / Coverage / Residual Ranking）、每次結果落盤、Resume、MVTec-compatible Selected/Merged 輸出。
- 不會啟動既有三篩、也不會自己聲稱測得 FPR／Recall。最終異常偵測效能要用原三篩**另外執行**評估。

**暫未實作：** Rotation Augmentation（正式部署一致性實驗應加上 Stage 2 Rotate × 30）；Stage 1 Top-K/Tail-aware 聚合；三篩 anomaly-map denoise/threshold；多 GPU 並行；Approximate Neighbor 搜尋。這些功能會在設定檔明確拒絕，**不會默默偽裝成已支援**。

## 0. Server 環境

已參考 Server 版本：`torch==2.5.1`、`torchvision==0.20.1`、`numpy==1.26.4`、`Pillow==11.3.0`、`scipy==1.15.3`、`pandas==2.2.0`。若既有環境可正常使用，不必為本專案重新安裝整套 PyTorch。另需 `PyYAML`；可用 `python -c 'import yaml'` 確認。測試則需 `pytest`。

不包含模型權重，也不包含 Meta DINOv2 Source Tree；請先確認下列檔案存在：

- `/workspace/Venessa/dinov2/hubconf.py`（Meta 官方 Repo）
- `/workspace/model_weight/dinov2_vitl14_reg4_pretrain.pth`
- `/workspace/Venessa/AnomalyVFM_/pretrained_models/anomalyvfm_dinov2.pkl`

**注意：Server 的資料夾名稱是 `AnomalyVFM_`、`SubspaceAD_`（有底線）。** 此專案不會引用 `SubspaceAD_` 原始碼，也不會從 `AnomalyVFM_` Import Python 模組；`AnomalyVFM_` 只出現在權重檔的 Config 路徑中。

## 1. 資料夾格式

```text
/workspace/data/develop/
  class1/train/good/a.jpg        # A：原本三篩訓練影像
  class1/val/good/ref.jpg        # R_good：獨立正常參考影像
  class2/train/good/...
  class2/val/good/...

/workspace/data/candidates/
  class1/001.jpg                 # 候選 U：不需 train/good 子目錄
  class1/002.png
  class2/001.jpg
```

Class 名稱自 `candidate_root` 第一層自動探索；每個 Class 完全獨立選樣、建 PCA。支援 PNG / JPG / JPEG / BMP / TIFF / WEBP，可在 Class 內遞迴掃描。`initial_train_root`、`reference_good_root` 及 `candidate_root` 可以是三個不同的根目錄。

### 標籤安全

候選池的二篩／三篩正常預測**不等於**可信 Ground Truth。預設 `data.assume_candidates_verified_good: false`，沒有 Manifest 時候選是 `unknown`、Gate 回 `pending`，不能自動當正常加入 PCA。若 Class 沒有可信 Bootstrap 圖片，會**在耗時的 DINO 特徵萃取之前**明確報 `bootstrap_insufficient_verified_good`。

請提供 `label_manifest.csv`：

```csv
image_path,label_status,source
class1/001.jpg,verified_good,stage2_pred_normal
class1/002.png,unknown,stage3_pred_normal
class2/001.jpg,dataset_good,curated_normal_set
```

可選欄位 `label_status`：`verified_good`、`dataset_good`、`unknown`、`confirmed_abnormal`。`image_path` 相對 `candidate_root`，包含 Class 名。`confirmed_abnormal` 在計算 Coverage 前即排除，避免污染 Coverage Pool。若**已透過其他程序確認候選資料夾全部正常**，才可設定 `assume_candidates_verified_good: true`，這是明確的使用者信任決策，請勿拿未確認資料直接設定 true。

生成待填的 CSV 模板：

```bash
python scripts/make_manifest.py --candidate-root /workspace/data/candidates --output /workspace/data/candidate_labels.csv
```

選樣會對 A、U、R_good 做 **SHA256 檔案內容重複檢查**；跨 Split 有相同檔案時會報資料洩漏錯誤（不是悄悄移除）。同 Split 的重複檔保留最先掃描的一份。若不同編碼或裁切但視覺上相同，仍須自行人工檢查。

## 2. 你只要修改哪些設定檔？

**主要修改 `configs/paths.yaml`：**

```yaml
model:
  dino_repo_path: /workspace/Venessa/dinov2
  dino_weight_path: /workspace/model_weight/dinov2_vitl14_reg4_pretrain.pth
  anomalyvfm_ckpt: /workspace/Venessa/AnomalyVFM_/pretrained_models/anomalyvfm_dinov2.pkl

data:
  initial_train_root: /workspace/data/develop
  candidate_root: /workspace/data/candidates
  reference_good_root: /workspace/data/develop
  initial_subdir: train/good
  reference_subdir: val/good
  classes: auto
  label_manifest: /workspace/data/candidate_labels.csv
  assume_candidates_verified_good: false

output:
  root: ./outputs
  cache_root: ./cache
  image_export: copy     # copy / symlink
```

**演算法參數修改 `configs/selection.yaml`：** 預設 `two_stage`、新增上限 8（含 Bootstrap）、Fixed PCA 600、Stage 1 Mean、Stage 2 Mean、無 Augmentation。若需要與三篩 EV=0.99 設定比較，將 `pca.mode` 改為 `explained_variance`。實際三篩的 Rotate × 30 並未包含在 v1，不能宣稱 v1 最終訓練空間與完整三篩完全相同。

## 3. 正式上 H100 前，照順序執行

以下指令請在 `RepresentativeSelection_/` 根目錄執行：

```bash
# 1) 檢查環境與模型檔案路徑
python scripts/check_environment.py

# 2) CPU 的核心演算法／Controller 測試，不需要模型權重
python scripts/smoke_test.py

# 2b) 獨立 PCA 數學診斷，不需 H100 或 DINO 模型權重
python scripts/test_pca.py

# 3) 只檢查資料和 Class / split；不萃取特徵
python main.py --dry-run

# 4) 使用一張真實影像檢查完整 Offline A1 DINO 載入
python scripts/test_extractor.py --image /workspace/data/develop/class1/train/good/example.jpg

# 可選：對已儲存的 DINO [P,1024] 特徵測試 Fixed 600 PCA（可在 H100 執行）
python scripts/test_pca.py --features-npy /path/to/feature1.npy /path/to/feature2.npy --fixed-dim 600 --device cuda

# 5) H100 Patch GEMM benchmark（先做小規模測試）
python scripts/benchmark_coverage.py --device cuda --iterations 10

# 6) 只跑一個 Class，確認 2-stage 能完整輸出
python main.py --classes class1

# 7) 跑所有 Class；每個 Class 都獨立建立 PCA 和輸出結果
python main.py
```

請將 `class1` 和 `example.jpg` 改為實際值。**執行之前一定要先設定 `paths.yaml`，而且要有可信 R_good 與可信正常候選資料。**

## 4. Stage 1 的 Exact Coverage 和 5,000 張影像

數學定義：

```text
k(p,q) = (1 + cos(p,q)) / 2
m_T(p) = max_{q in patches(T)} k(p,q)
m_I(p) = max_{q in patches(I)} k(p,q)
Gain(I) = mean_{x in U} mean_{p in patches(x)} max(m_I(p) - m_T(p), 0)
```

- **全部有效 Patch 都參與**（不降維、不抽樣、不採 ANN），不對齊 Patch 空間座標。
- Patch-to-Patch 相似度使用 GPU Block Matrix Multiplication，但立即對候選 Patches 做 Max Reduction，並不保存龐大的全部 Patch-Pair Matrix。
- 可預算空間時，快取 `max_similarity[候選影像, U 的每個 Patch]`；Greedy Accept 時只更新 Baseline Coverage，Reject 時絕不更新。
- `compute_dtype: float16` 對 H100 加速，但數值是**有限精度**；如需數值更嚴格，改用 `float32` 並對照排序差異。Exact 表示**不近似搜尋／不丟 Patch**，不是 FP16 位元完全精確。
- U 的 Patches 若可在顯示卡剩餘 VRAM 的約 65% 內存下，會預載入 GPU，否則流式讀取 Feature Cache。

假設一個 Class 有 N 張、每張 2,304 Patches，FP32 Max-Similarity Cache 估計約 `4 * N² * 2304` bytes：

| 單一 Class 張數 | Max-Sim Cache（十進位） |
|---:|---:|
| 100 | 0.092 GB |
| 500 | 2.30 GB |
| 1,000 | 9.22 GB |
| 5,000 | 230.4 GB |

如果 **5,000 張是分散在很多 Class**，逐 Class 處理能大幅降低成本。如果**單一 Class 有 5,000 張**，有可能需要數百 GB NVMe 空間及很長的運算時間；`cache_mode: auto` 遇到快取空間不足且候選達 750 張以上，會**拒絕自動切換到極慢的 streaming**，避免無意中啟動多日重算。這時需在 H100 進行 benchmark 後，明確提高 `disk_budget_gib` 並使用 `cache_mode: precompute`（足夠磁碟），或自行決定使用 `stream`（重算代價極高）。

## 5. PCA Gate 的演算法與限制

- **每個 Class 各自一個 PCA**；A 與通過 Gate 的候選 Patch Features 共同建立 PCA。v1 沒有 Augmentation。
- PCA 支援 `fixed_dim: 600` 或 `explained_variance: 0.99`，建議 Fixed 600 用於控制 PCA 容量，EV=0.99 用於與目前三篩對照；若 rank 不足以支援固定維度則明確報錯。
- Stage 2 透過 `n, sum(x), sum(xxᵀ)` 統計量組合訓練集合後的 Covariance，再對 1024 × 1024 對稱矩陣進行 Eigen Decomposition。並非不重新 Fit PCA；而是**不用每次重新讀取所有訓練影像的統計量**。
- `R_good` 僅用來評估候選加入前後的 Reconstruction Error，不參與 PCA Fit；Final Test 不能參與參數校準。
- Gate 預設 `mean`：先算每張 R_good 的平均 Patch Residual，再對影像等權平均。可切換 `p90` 或 `high_error_ratio`；High-error Ratio 以**本 Class 第一個 Gate 模型**在 R_good 的 Q95 校準固定門檻 τ，Before/After 不可變動 τ。
- `delta=(Loss_before - Loss_after) / max(Loss_before, 1e-12)`；預設 **嚴格** `delta > 0.0001` 才 Accept，並非 `>=0`。門檻屬探索設定，後續校準。
- `accept` 才正式提交 PCA 和 Coverage；`reject/pending` 只排除本輪候選，不修改 PCA。未知標籤為 Pending，若從 Manifest 排除 confirmed_abnormal 才是 Coverage 正確行為。

## 6. Experiment Mode

編輯 `configs/selection.yaml` 的 `selection.mode`：

| mode | Proposal | Acceptance | 備註 |
|---|---|---|---|
| `two_stage` | Mean Coverage | PCA Gate | 主要方法 |
| `coverage_only` | Mean Coverage | 直接接受 | 要求可信正常 |
| `random_only` | Random | 直接接受 | Random Baseline |
| `random_gate` | Random | PCA Gate | 檢查 Stage 1 的增益 |
| `residual_only` | 固定初始 PCA 的 Residual Score Top-K | 直接接受 | Stage 2 原文件 v1 獨立 Baseline |

**不同實驗請修改 `experiment.run_name`。** Residual-only 的初始 PCA 只用於選樣排名，最後才建立最終 PCA；它不是對每張候選的 Trial PCA Gate。`Random/coverage_only` 會在選樣完成後建最終 PCA，不會為每張提案執行 Stage 2 Gate。

## 7. 輸出與 Resume

```text
outputs/<run_name>/
  selected_only/<class>/train/good/*.jpg
  merged_training_set/<class>/train/good/*.jpg
  logs/<class>/
    controller_state.json   # 可恢復的 Bootstrap / Accepted / Attempted / decisions
    selection_decisions.csv # 每張被實際提案的 image_id/score/decision/delta
    selected_ids.csv
    summary.json
  summary.csv
  run_config.json
```

`merged_training_set` 會包含原始 A + 新增 Bootstrap B + 正式接受 S。新影像輸出檔名帶 SHA 前綴，避免與原始 A 同名。正式執行不會修改任何原始資料。`output.image_export: symlink` 可節省硬碟容量，但搬去其他 Server 時連結可能失效；預設 `copy` 更可攜。

`controller_state.json` 在每次 Bootstrap 和 Proposal 後寫入；重新使用相同 `run_name`、相同資料/Config 可 Resume。**已完成的 Class 恢復時，直接重新輸出既有結果，不重跑 DINO 或 Coverage**。**若要重頭比較實驗，必須改用新的 `experiment.run_name`（或先由使用者自行整理舊輸出）。為保護舊結果，現在 `--no-resume` 遇到同名 Checkpoint 會明確報錯，不會覆寫既有結果。**

## 8. 已知的研究／工程界限

1. **不含 Stage 2 Rotate × 30**：這是下一版正式和你原本三篩 PCA 對照時要補的功能。
2. **真實 H100 未驗證**：本交付可完成 CPU Synthetic Unit/Integration Tests，但仍需在實際 Server 上跑 `test_extractor.py`、Cache Benchmark，以及與舊三篩同圖特徵／PCA 數值對照。
3. **PCA Gate 不是最終 FPR/Recall**：需要你獨立使用三篩在同樣的 D1/D2、Top 0.5% 等設定下跑最終驗證。
4. **來源可靠度**：二篩判正常／三篩判正常皆不等於 Verified Good。未知候選可被 Coverage 評估，但不能自動接受；大量未知或異常未被辨識時會污染 Coverage 目標分布。
5. **選樣效率**：最壞情況的 Full Exact Coverage 非常昂貴；不保證一張 H100 可快速完成單 Class 5k，全流程時間需要實測。
6. **資料重新編碼但內容相同**：SHA256 只能識別位元完全相同的檔案，無法自動發現經重新編碼、旋轉或裁切的語意重複圖。
7. **SubspaceAD 一致性**：離線 DINO 來源及數學策略參考你提供的程式與 GitHub；但尚未在你的 Server 進行逐像素特徵與 PCA 參數比對，不應將此視為已完成部署驗證。

## 9. 技術依據

- Stage 1: `Stage1_Greedy_Patch_Coverage_Implementation_Guide_zhTW.pdf`（2026-11-20）
- Stage 2: `Stage2_PCA_Reconstruction_Error_Guided_Selection_Implementation_Guide.pdf`（2026-11-25）
- 使用者提供的 `dinov2_offline.py`
- 公開程式作為**參考**：https://github.com/yuting8902/SubspaceAD 、 https://github.com/yuting8902/AnomalyVFM

完整專案不依賴上述其他兩個 Python 專案的程式呼叫。

## 10. CPU 測試與實際 H100 的驗證狀態

更多詳情見 `VALIDATION.md`。交付時 CPU 的數學、資料與 Controller 測試通過；真實 A1 權重載入、GPU Feature Extraction、數值對照、H100 最佳 Chunk Size 和 5k 真實資料規模都必須到 Server 再確認。

需要比對新舊特徵時，可以從兩個**獨立**程式各輸出 `.npy`，再執行：

```bash
python scripts/test_extractor.py --image /your/sample.jpg --save-npy /tmp/new_features.npy
python scripts/compare_feature_arrays.py --old /tmp/old_features.npy --new /tmp/new_features.npy
```

不要求在本專案中 Import 現有 `SubspaceAD_`。

## 11. 開發與移植補充（2026-11-26）

- 即使使用 CPU 執行 Synthetic Tests，也可以驗證所有主要數學公式與狀態更新；**這不代表**已經在 H100、實際 A1 Checkpoint 或 5,000 張資料上驗證。
- `bootstrap.enabled: false` 可在**既有 A 能直接建立 PCA**時停用自動補足；若 A 為空且停用 Bootstrap，會明確報錯。
- `random_only`、`random_gate` 無需先萃取全部候選特徵，只會在提案後按需取得；`coverage` 與 `residual_only` 則依定義需要對整個候選池計算特徵。
- 如果每個 Class 的候選池很大，請先執行 `python main.py --dry-run`、再執行 H100 Coverage Benchmark，並確保 Disk Cache 的空間足夠；本程式**不會**因 GPU 記憶體不足而隱性啟用 Patch Sampling。
- 本專案只使用外部 `AnomalyVFM_` Checkpoint 與 Meta DINOv2 原始 Repository／預訓練權重；不引用其他專案的 Python 模組。`offline_models/dinov2_offline.py` 來源為使用者提供的相容程式，`peft_local/` 是僅支援 A1 DoRA 的獨立相容封裝，應在公司 Server 上執行實際 Checkpoint 的 `strict=True` 載入測試。
