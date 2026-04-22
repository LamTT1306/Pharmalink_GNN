"""
Bước 3: Drug Repurposing — AI phát hiện công dụng mới

Sau khi train xong best_model_fuzzy_c_dataset.pt (đã loại bỏ cặp nghi vấn),
script này:
  1. Load lại mô hình và dữ liệu
  2. Đưa toàn bộ cặp nghi vấn (từ suspect_pairs.csv) vào GNN dự đoán
  3. Lọc ra những cặp có xác suất cao (prob >= prob_thr)
  4. Lưu vào Result/<dataset>/drug_repurposing_candidates.csv
     → dùng cho web app SmartPharmacy "AI phát hiện công dụng mới"

Usage:
  python drug_repurposing.py --dataset C-dataset --prob_thr 0.7
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# Cho phép import từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocess import get_data, data_processing, dgl_similarity_graph, dgl_heterograph
from model.AMNTDDA_Fuzzy import AMNTDDA_Fuzzy
from model.AMNTDDA import AMNTDDA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   default='C-dataset')
    parser.add_argument('--model',     default='gnn_fuzzy', choices=['gnn', 'gnn_fuzzy'])
    parser.add_argument('--prob_thr',  type=float, default=0.7,
                        help='ngưỡng xác suất để lọc ứng viên (0-1)')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='chunk size khi inference (tránh OOM)')
    # Hyper-params mặc định (dùng để khởi tạo model đúng kích thước)
    parser.add_argument('--gt_layer',    type=int, default=2)
    parser.add_argument('--gt_head',     type=int, default=2)
    parser.add_argument('--gt_out_dim',  type=int, default=200)
    parser.add_argument('--hgt_layer',   type=int, default=2)
    parser.add_argument('--hgt_head',    type=int, default=8)
    parser.add_argument('--hgt_in_dim',  type=int, default=64)
    parser.add_argument('--hgt_head_dim', type=int, default=25)
    parser.add_argument('--hgt_out_dim', type=int, default=200)
    parser.add_argument('--tr_layer',    type=int, default=2)
    parser.add_argument('--tr_head',     type=int, default=4)
    parser.add_argument('--dropout',     type=float, default=0.2)
    parser.add_argument('--neighbor',    type=int, default=20)
    parser.add_argument('--negative_rate', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--fuzzy_rules',    type=int, default=32)
    parser.add_argument('--fuzzy_dim',      type=int, default=256)
    parser.add_argument('--fuzzy_proj_dim', type=int, default=64)
    parser.add_argument('--fuzzy_dropout',  type=float, default=0.05)
    args = parser.parse_args()

    args.data_dir   = f'data/{args.dataset}/'
    args.result_dir = f'Result/{args.dataset}/drug_repurposing/'
    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[repurposing] Dataset: {args.dataset} | Device: {device}')

    # ── Xác định checkpoint ───────────────────────────────────────────────────
    dataset_tag = args.dataset.replace('-', '_').lower()
    ckpt_path   = (
        f'web_app/models/best_model_fuzzy_{dataset_tag}.pt'
        if args.model == 'gnn_fuzzy'
        else f'web_app/models/best_model_{dataset_tag}.pt'
    )
    if not os.path.exists(ckpt_path):
        print(f'[repurposing] Không tìm thấy checkpoint: {ckpt_path}')
        print('[repurposing] Hãy train mô hình trước: python train_DDA.py --model gnn_fuzzy --dataset ...')
        return

    # ── Kiểm tra suspect_pairs.csv ────────────────────────────────────────────
    suspect_path = Path(args.data_dir) / 'suspect_pairs.csv'
    if not suspect_path.exists():
        print(f'[repurposing] Không tìm thấy: {suspect_path}')
        print('[repurposing] Hãy chạy Bước 1 trước: python scan_fake_negatives.py --dataset ...')
        return

    suspect_df = pd.read_csv(suspect_path)
    print(f'[repurposing] Tổng cặp nghi vấn: {len(suspect_df):,}')

    # ── Load dữ liệu ─────────────────────────────────────────────────────────
    print('[repurposing] Đang load dữ liệu...')
    data = get_data(args)
    args.drug_number    = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)

    import random; random.seed(args.random_seed)
    from data_preprocess import k_fold, dgl_similarity_graph, dgl_heterograph
    data = k_fold(data, args)
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    # Dùng toàn bộ drdi để build heterograph đầy đủ nhất
    all_train = data['X_train'][0]   # fold 0 — chỉ cần graph topology
    drdipr_graph, _ = dgl_heterograph(data, all_train, args)

    drdr_graph   = drdr_graph.to(device)
    didi_graph   = didi_graph.to(device)
    drdipr_graph = drdipr_graph.to(device)

    drug_feature    = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'[repurposing] Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    # Nếu checkpoint lưu args gốc, ưu tiên dùng để đảm bảo kích thước khớp
    if 'args' in ckpt:
        saved_args = ckpt['args']
        for attr in ['drug_number', 'disease_number', 'protein_number',
                     'gt_layer', 'gt_head', 'gt_out_dim',
                     'hgt_layer', 'hgt_head', 'hgt_in_dim', 'hgt_head_dim',
                     'fuzzy_rules', 'fuzzy_dim', 'fuzzy_proj_dim']:
            if hasattr(saved_args, attr):
                setattr(args, attr, getattr(saved_args, attr))

    if args.model == 'gnn_fuzzy':
        model = AMNTDDA_Fuzzy(args)
    else:
        model = AMNTDDA(args)

    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print('[repurposing] Model loaded.')

    # ── Lấy embedding toàn bộ (GNN chạy 1 lần) ──────────────────────────────
    print('[repurposing] Đang trích xuất embedding...')
    with torch.no_grad():
        if args.model == 'gnn_fuzzy':
            dr_emb, di_emb = model._get_embeddings(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature)
        else:
            # AMNTDDA gốc không có _get_embeddings — chạy forward với dummy sample
            dummy = torch.zeros(1, 2, dtype=torch.long).to(device)
            dr_emb, _ = model(drdr_graph, didi_graph, drdipr_graph,
                              drug_feature, disease_feature, protein_feature, dummy)
            # di_emb cần tính riêng từ drug_trans của model gốc
            # (fallback: dùng cách chunk với model.forward)
            di_emb = None

    # ── Inference theo chunk ──────────────────────────────────────────────────
    suspect_pairs = suspect_df[['drug_idx', 'disease_idx']].to_numpy(dtype=int)
    all_probs = []

    print(f'[repurposing] Inference {len(suspect_pairs):,} cặp (chunk={args.chunk_size})...')
    with torch.no_grad():
        chunks = np.array_split(suspect_pairs, max(1, len(suspect_pairs) // args.chunk_size))
        for c_idx, chunk in enumerate(chunks):
            sample_t = torch.LongTensor(chunk).to(device)

            if di_emb is not None:
                # Fuzzy model: dùng embedding đã tính
                interact   = dr_emb[sample_t[:, 0]] * di_emb[sample_t[:, 1]]
                fuzzy_out  = model.fuzzy_layer(interact)
                correction = model.fuzzy_correction(fuzzy_out)
                logits     = model.mlp(interact + correction)
            else:
                # AMNTDDA gốc: cần full forward
                _, logits = model(drdr_graph, didi_graph, drdipr_graph,
                                  drug_feature, disease_feature, protein_feature,
                                  sample_t)

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

            if c_idx % 20 == 0:
                print(f'  chunk {c_idx}/{len(chunks)} done')

    # ── Lọc + lưu kết quả ────────────────────────────────────────────────────
    suspect_df['repurposing_prob'] = all_probs
    candidates = (
        suspect_df[suspect_df['repurposing_prob'] >= args.prob_thr]
        .sort_values('repurposing_prob', ascending=False)
        .reset_index(drop=True)
    )

    print(f'\n[repurposing] Tổng ứng viên (prob >= {args.prob_thr}): {len(candidates):,}')

    # Thêm tên thuốc/bệnh nếu có
    try:
        drug_info = pd.read_csv(f'{args.data_dir}DrugInformation.csv', header=None)
        dis_info  = pd.read_csv(f'{args.data_dir}ProteinInformation.csv', header=None)  # fallback
        drug_names = {i: str(drug_info.iloc[i, 0]) for i in range(len(drug_info))}
        candidates['drug_name'] = candidates['drug_idx'].map(drug_names)
    except Exception:
        pass
    try:
        dis_info = pd.read_csv(f'{args.data_dir}ProteinInformation.csv', header=None)
    except Exception:
        pass

    out_path = Path(args.result_dir) / 'drug_repurposing_candidates.csv'
    candidates.to_csv(out_path, index=False)
    print(f'[repurposing] Đã lưu: {out_path}')

    # Thống kê phân phối xác suất
    probs_arr = np.array(all_probs)
    print(f'\n[repurposing] Phân phối xác suất của {len(suspect_pairs):,} cặp nghi vấn:')
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cnt = (probs_arr >= thr).sum()
        print(f'  prob >= {thr:.2f}: {cnt:,} cặp  ({cnt/len(suspect_pairs)*100:.1f}%)')

    print('\n[repurposing] Xong! Các ứng viên Drug Repurposing đã sẵn sàng.')
    print(f'[repurposing] Tích hợp vào web app: đọc {out_path} trong endpoint /repurposing')


if __name__ == '__main__':
    main()
