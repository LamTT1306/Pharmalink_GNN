# Pharmalink GNN — Drug-Disease Association Prediction

Hệ thống dự đoán liên kết thuốc-bệnh sử dụng Graph Neural Network (GNN) và Fuzzy Logic.

## Yêu cầu môi trường

| Package | Version đã test |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.3.0+cu121 |
| DGL | 2.2.1+cu121 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| networkx | 3.4.2 |
| flask | 3.1.3 |
| google-genai | >=1.0 |
| replicate | >=0.25.0 |

> Cần **GPU CUDA 12.1+** để chạy model. CPU-only sẽ rất chậm.

### Cài đặt

**Bước 1 — PyTorch + DGL (CUDA 12.1):**
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3.0/cu121/repo.html
```

**Bước 2 — Các thư viện còn lại:**
```bash
pip install numpy==1.26.4 pandas scikit-learn networkx
pip install -r web_app/requirements.txt
```

> Nếu dùng **Google Colab**, cần thêm `torchdata==0.7.1` trước khi cài DGL:
> ```bash
> pip install torchdata==0.7.1
> ```

## Dữ liệu

Thư mục `data/` chứa 3 bộ dữ liệu: **B-dataset**, **C-dataset**, **F-dataset**.

Mỗi bộ gồm:
- `Drug_mol2vec` — mol2vec embeddings cho thuốc
- `DrugFingerprint`, `DrugGIP` — độ tương đồng thuốc
- `DiseaseFeature`, `DiseasePS`, `DiseaseGIP` — đặc trưng và độ tương đồng bệnh
- `Protein_ESM` — ESM-2 embeddings cho protein
- `DrugDiseaseAssociationNumber`, `DrugProteinAssociationNumber`, `ProteinDiseaseAssociationNumber` — ma trận liên kết đã biết

## Download Model (.pt)

Các file model **không được lưu trong repo** (quá lớn). Tải về tại:

> **[Google Drive — Pretrained Models](https://drive.google.com/drive/folders/1uzS_gWNGv1hipYGFQ5fzrhK2ThI1YDJy?usp=drive_link)**

Sau khi tải, đặt vào thư mục `web_app/models/`:
```
web_app/models/best_model_fuzzy_c_dataset.pt   ← C-dataset GNN+Fuzzy (AUC 0.9602)
web_app/models/best_model_fuzzy_b_dataset.pt   ← B-dataset GNN+Fuzzy
web_app/models/best_model_fuzzy_f_dataset.pt   ← F-dataset GNN+Fuzzy
```

## Train lại model

```bash
# C-dataset (mặc định)
python train_DDA.py --model gnn_fuzzy --dataset C-dataset --epochs 1000

# B-dataset (giảm dim để tránh OOM)
python train_DDA.py --model gnn_fuzzy --dataset B-dataset --epochs 1000 \
    --gt_out_dim 64 --hgt_head 4 --hgt_head_dim 16 --neighbor 10

# F-dataset
python train_DDA.py --model gnn_fuzzy --dataset F-dataset --epochs 1000
```

## Chạy Web App

```bash
cd web_app
python app.py
```

Truy cập: http://127.0.0.1:5000

## Cấu trúc dự án

```
data/               ← Dữ liệu (B/C/F-dataset)
model/              ← Kiến trúc GNN (AMNTDDA, GraphTransformer, HGT)
web_app/            ← Flask web application
  models/           ← File model .pt (tải riêng)
  templates/        ← HTML templates
  static/           ← CSS, JS
train_DDA.py        ← Script huấn luyện
data_preprocess.py  ← Tiền xử lý dữ liệu
metric.py           ← Tính toán metrics
```
