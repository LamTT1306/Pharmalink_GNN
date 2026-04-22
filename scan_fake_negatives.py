"""
Bước 1: Tầm soát "Fake Negatives" (Positive-Unlabeled Learning)

Nguyên lý:
  Một cặp (drug_i, disease_j) bị gán nhãn 0 được xem là "nghi vấn" nếu:
    ① Drug_i có cấu trúc hóa học giống cao (DrugFingerprint Tanimoto) với
       ít nhất một thuốc đã biết điều trị disease_j.
    ② Disease_j có profile tương đồng (DiseasePS semantic) với ít nhất
       một bệnh đã được điều trị bởi drug_i.
    ③ GIP similarity (tương đồng tương tác) dùng để xác nhận thêm,
       KHÔNG dùng làm tiêu chí chính (tránh circular reasoning).

Lưu ý về circular reasoning:
  DrugGIP[i, k] = 1 khi drug_i và drug_k có hồ sơ tương tác giống nhau.
  Nếu drug_i và drug_k đều chữa disease_j → GIP cao vì chính liên kết đó
  tạo ra độ tương đồng → dùng GIP làm tiêu chí chính là suy luận vòng.
  → Sử dụng DrugFingerprint (đặc trưng hóa học độc lập) + DiseasePS
    (semantic similarity từ ontology, độc lập với ma trận tương tác).

Kết quả:
  data/<dataset>/suspect_pairs.csv  — danh sách cặp nghi vấn
  data/<dataset>/suspect_summary.txt — thống kê

Usage:
  python scan_fake_negatives.py --dataset C-dataset --drug_thr 0.5 --dis_thr 0.3 --gip_confirm 0.2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ── Tanimoto similarity cho binary fingerprint ───────────────────────────────

def tanimoto_matrix(fp: np.ndarray) -> np.ndarray:
    """
    fp : (N, D) binary fingerprint matrix
    Returns (N, N) Tanimoto similarity:
      T(a, b) = |a ∩ b| / |a ∪ b| = dot(a,b) / (|a|+|b|-dot(a,b))
    """
    dot    = fp @ fp.T                              # (N, N)
    norms  = fp.sum(axis=1)                         # (N,)
    denom  = norms[:, None] + norms[None, :] - dot  # (N, N)
    denom  = np.where(denom == 0, 1e-8, denom)
    return dot / denom


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     default='C-dataset')
    parser.add_argument('--drug_thr',    type=float, default=0.5,
                        help='ngưỡng Tanimoto fingerprint giữa drug_i và drug đã điều trị disease_j')
    parser.add_argument('--dis_thr',     type=float, default=0.3,
                        help='ngưỡng cosine DiseasePS giữa disease_j và bệnh đã được drug_i điều trị')
    parser.add_argument('--gip_confirm', type=float, default=0.2,
                        help='ngưỡng GIP để xác nhận thêm (mức độ thấp, chỉ loại bỏ cặp hoàn toàn không liên quan)')
    parser.add_argument('--topk',        type=int,   default=5,
                        help='lấy top-k thuốc/bệnh liên quan để tính điểm trung bình')
    args = parser.parse_args()

    data_dir = Path('data') / args.dataset
    out_dir  = data_dir

    print(f'[scan] Dataset: {args.dataset}')
    print(f'[scan] Thresholds — drug_fp: {args.drug_thr}, dis_ps: {args.dis_thr}, gip_confirm: {args.gip_confirm}')

    # ── Load dữ liệu ─────────────────────────────────────────────────────────
    adj_csv = 'adj.csv'
    adj = pd.read_csv(data_dir / adj_csv, index_col=0).to_numpy().astype(int)   # (D, I)
    n_drug, n_dis = adj.shape
    print(f'[scan] adj: {n_drug} drugs × {n_dis} diseases  |  positives: {adj.sum()}')

    # DrugFingerprint — đặc trưng hóa học (binary), cột đầu là index
    fp_df = pd.read_csv(data_dir / 'DrugFingerprint.csv')
    drug_fp = fp_df.iloc[:, 1:].to_numpy().astype(float)                        # (D, fp_dim)

    # DiseasePS — semantic similarity từ ontology (độc lập với adj)
    dips_df = pd.read_csv(data_dir / 'DiseasePS.csv')
    dis_ps  = dips_df.iloc[:, 1:].to_numpy().astype(float)                      # (I, I)

    # DrugGIP + DiseaseGIP — dùng xác nhận phụ (không làm tiêu chí chính)
    drg = pd.read_csv(data_dir / 'DrugGIP.csv').iloc[:, 1:].to_numpy()          # (D, D)
    dig = pd.read_csv(data_dir / 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()       # (I, I)

    # ── Tính ma trận Tanimoto giữa các thuốc ─────────────────────────────────
    print('[scan] Tính Tanimoto similarity giữa các thuốc...')
    drug_sim = tanimoto_matrix(drug_fp)                                          # (D, D)
    np.fill_diagonal(drug_sim, 0.0)   # bỏ self-similarity

    # ── Tính cosine similarity giữa các bệnh (DiseasePS) ─────────────────────
    print('[scan] Tính cosine similarity giữa các bệnh...')
    # dis_ps đã là similarity matrix (ontology-based), không cần tính thêm
    dis_sim = dis_ps.copy()
    np.fill_diagonal(dis_sim, 0.0)

    # ── Tìm danh sách thuốc/bệnh liên kết theo adj ───────────────────────────
    # drug_for_dis[j] = list of drug indices that treat disease j
    drug_for_dis = [np.where(adj[:, j] == 1)[0] for j in range(n_dis)]
    # dis_for_drug[i] = list of disease indices treated by drug i
    dis_for_drug = [np.where(adj[i, :] == 1)[0] for i in range(n_drug)]

    # ── Quét từng cặp âm tính ────────────────────────────────────────────────
    print('[scan] Đang quét các cặp âm tính...')
    neg_i, neg_j = np.where(adj == 0)
    total_neg = len(neg_i)
    print(f'[scan] Tổng cặp âm tính: {total_neg:,}')

    suspect_records = []

    for idx in range(total_neg):
        i = neg_i[idx]
        j = neg_j[idx]

        # Thuốc đã biết điều trị disease_j
        pos_drugs = drug_for_dis[j]
        # Bệnh đã biết được điều trị bởi drug_i
        pos_dis   = dis_for_drug[i]

        if len(pos_drugs) == 0 or len(pos_dis) == 0:
            continue   # bỏ qua nếu không có positive anchor

        # ① Drug chemical similarity (Tanimoto fingerprint)
        drug_scores = drug_sim[i, pos_drugs]
        topk_drug   = np.sort(drug_scores)[::-1][:args.topk]
        drug_score  = topk_drug.mean()

        if drug_score < args.drug_thr:
            continue   # cắt sớm để tăng tốc

        # ② Disease semantic similarity (DiseasePS)
        dis_scores = dis_sim[j, pos_dis]
        topk_dis   = np.sort(dis_scores)[::-1][:args.topk]
        dis_score  = topk_dis.mean()

        if dis_score < args.dis_thr:
            continue

        # ③ GIP xác nhận phụ — loại bỏ nếu GIP quá thấp (cặp hoàn toàn không liên quan)
        gip_drug_score = drg[i, pos_drugs].mean() if len(pos_drugs) > 0 else 0.0
        gip_dis_score  = dig[j, pos_dis].mean()   if len(pos_dis)   > 0 else 0.0

        if gip_drug_score < args.gip_confirm or gip_dis_score < args.gip_confirm:
            continue

        # Điểm tổng hợp (kết hợp 3 tín hiệu)
        combined = 0.5 * drug_score + 0.3 * dis_score + 0.1 * gip_drug_score + 0.1 * gip_dis_score

        suspect_records.append({
            'drug_idx':        i,
            'disease_idx':     j,
            'drug_fp_score':   round(drug_score,  4),
            'dis_ps_score':    round(dis_score,   4),
            'gip_drug_score':  round(gip_drug_score, 4),
            'gip_dis_score':   round(gip_dis_score,  4),
            'combined_score':  round(combined,    4),
        })

        if idx % 50000 == 0:
            print(f'  {idx:,}/{total_neg:,} quét xong | nghi vấn hiện tại: {len(suspect_records):,}')

    print(f'\n[scan] Tổng cặp nghi vấn tìm thấy: {len(suspect_records):,}')

    if not suspect_records:
        print('[scan] Không tìm thấy cặp nghi vấn với ngưỡng hiện tại.')
        print('[scan] Thử giảm --drug_thr hoặc --dis_thr xuống.')
        return

    # ── Lưu kết quả ─────────────────────────────────────────────────────────
    df = pd.DataFrame(suspect_records).sort_values('combined_score', ascending=False)
    out_path = out_dir / 'suspect_pairs.csv'
    df.to_csv(out_path, index=False)
    print(f'[scan] Đã lưu: {out_path}')

    # Thống kê
    summary_path = out_dir / 'suspect_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Dataset       : {args.dataset}\n')
        f.write(f'Tổng âm tính  : {total_neg:,}\n')
        f.write(f'Cặp nghi vấn  : {len(df):,}  ({len(df)/total_neg*100:.2f}% âm tính)\n')
        f.write(f'Threshold drug: {args.drug_thr} (Tanimoto fingerprint)\n')
        f.write(f'Threshold dis : {args.dis_thr} (cosine DiseasePS)\n')
        f.write(f'GIP confirm   : {args.gip_confirm}\n\n')
        f.write('Top 20 cặp nghi vấn nhất:\n')
        f.write(df.head(20).to_string(index=False))
    print(f'[scan] Tóm tắt: {summary_path}')

    print(f'\n[scan] Phân phối combined_score:')
    print(df['combined_score'].describe().round(4))


if __name__ == '__main__':
    main()
