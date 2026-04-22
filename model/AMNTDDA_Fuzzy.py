"""
AMNTDDA_Fuzzy – Mô hình cải tiến, kế thừa từ AMNTDDA gốc và thêm lớp Fuzzy Inference.

Thiết kế
--------
  Backbone  : GIỮ NGUYÊN toàn bộ từ lớp AMNTDDA gốc (model/AMNTDDA.py).
              Không sao chép lại code – dùng kế thừa: class AMNTDDA_Fuzzy(AMNTDDA).

  Cải tiến (các thành phần mới, KHÔNG có trong AMNTDDA gốc):
    ① LearnableFuzzyLayer  (TSK-type, Gaussian MFs, fully differentiable)
         in       = 400-dim  (gt_out_dim × 2, giống AMNTDDA)
         proj_dim = 64       (chiếu xuống trước T-norm để tránh tích vanish)
         out      = 256-dim  fuzzy-activated features + residual
    ② fuzzy_correction  (Linear 256 → 400): cộng residual vào interaction vector
    ③ MLP giữ nguyên kích thước 400-dim input — kết quả so sánh công bằng với gốc

  Luồng forward (AMNTDDA_Fuzzy):
    [Backbone AMNTDDA] → interact (400) → fuzzy_layer → correction (400)
    → enhanced = interact + correction → mlp → 2 logits

  Tham số thêm (có giá trị mặc định nếu không truyền vào):
    --fuzzy_rules 32  --fuzzy_dim 256  --fuzzy_proj_dim 64

Training:
  Run train_DDA.py với --model gnn_fuzzy
  Checkpoint lưu tại web_app/models/best_model_fuzzy_{dataset}.pt
"""

import dgl
import torch
import torch.nn as nn
from model.AMNTDDA import AMNTDDA  # Import mô hình gốc — không sửa file đó


# ════════════════════════════════════════════════════════════════
#  Learnable Fuzzy Layer  (TSK type-0 + Gaussian MFs)
# ════════════════════════════════════════════════════════════════

class LearnableFuzzyLayer(nn.Module):
    """
    Improved Differentiable TSK Fuzzy Inference Layer (v3).

    Changes from v2:
      – Temperature scaling T (learnable): sharpens/softens Wang-Mendel weights.
          w_k = exp(−mean((x−c)²/(σ²·T)))  →  T<1 sharpens, T>1 softens.
      – Fuzzy Dropout (p=fuzzy_dropout): randomly zeros firing strengths during
          training to prevent over-reliance on a few dominant rules.
      – raw_w (pre-norm firing strengths) exposed via _firing_strengths so
          train_DDA can apply L1 sparsity loss on them directly.

    Parameters
    ----------
    in_features    : input dimensionality   (400 in AMNTDDA_Fuzzy)
    n_rules        : number of fuzzy rules  (default 32)
    out_features   : output dimensionality  (default 256)
    proj_dim       : projected dim before T-norm (default 64)
    fuzzy_dropout  : dropout rate on normalised firing strengths (default 0.1)
    """

    def __init__(self, in_features: int, n_rules: int = 32,
                 out_features: int = 256, proj_dim: int = 64,
                 fuzzy_dropout: float = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.n_rules      = n_rules
        self.out_features = out_features
        self.proj_dim     = proj_dim

        # ① Project to lower dim BEFORE T-norm (prevents prod collapse over 400 dims)
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        # Gaussian MF parameters in projected space (learnable)
        self.centers    = nn.Parameter(torch.randn(n_rules, proj_dim) * 0.1)
        self.log_sigmas = nn.Parameter(torch.zeros(n_rules, proj_dim))

        # ③ Temperature scaling — learnable, clamped to (0.1, 10) during forward
        # init = log(1) = 0  →  T_init = 1.0 (neutral starting point)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # ④ Learnable blend: α·Gaussian + (1-α)·Triangular
        # Gaussian: smooth gradient; Triangular: kháng outlier cứng hơn
        # α_init = sigmoid(0) = 0.5  →  cân bằng ban đầu
        self.log_alpha_blend = nn.Parameter(torch.zeros(1))

        # Fuzzy Dropout — only active during training
        self.rule_dropout = nn.Dropout(p=fuzzy_dropout)

        # TSK consequent + residual connection
        self.consequent = nn.Linear(n_rules, out_features, bias=True)
        self.res_proj   = nn.Linear(proj_dim, out_features, bias=False)
        self.out_norm   = nn.LayerNorm(out_features)

    # ── internal helpers ─────────────────────────────────────────

    def _firing_strengths(self, x: torch.Tensor):
        """
        x : (B, in_features) → (raw_w: (B,R), w_norm: (B,R), xp: (B,P))

        Returns raw_w (pre-normalisation) so the caller can compute
        L1 sparsity loss on it — after Wang-Mendel normalisation
        sum(w̃)=1 always, making L1(w̃) a useless constant.

        Temperature T < 1  →  sharpens firing (luật nào mạnh càng mạnh)
        Temperature T > 1  →  softens  firing (trải đều hơn)
        """
        xp   = self.input_proj(x)                                        # (B, P)
        diff = xp.unsqueeze(1) - self.centers.unsqueeze(0)               # (B, R, P)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0).clamp(min=1e-3) # (1, R, P)

        # ③ Temperature scaling applied inside the exponent
        T      = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)  # scalar
        neg_sq = -(diff / sigma).pow(2) / T                              # (B, R, P)

        # Cải tiến B: Gaussian + Triangular MF blend (học từ FuzzyGCN)
        # Gaussian: smooth, gradient liên tục
        gauss_w     = torch.exp(neg_sq.mean(dim=-1))                          # (B, R)
        # Triangular: cứng hơn, không bị nhiễu xa tâm → kháng outlier tốt hơn
        tri_w       = (1.0 - (diff.abs() / sigma).mean(dim=-1)).clamp(min=0.0) # (B, R)
        # α tự học: α·Gaussian + (1-α)·Triangular
        alpha_blend = torch.sigmoid(self.log_alpha_blend)
        raw_w       = alpha_blend * gauss_w + (1.0 - alpha_blend) * tri_w     # (B, R) pre-norm

        # Wang-Mendel normalisation
        denom  = raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = raw_w / denom                                           # (B, R)

        return raw_w, w_norm, xp

    # ── forward ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_features) → (B, out_features)"""
        raw_w, w_norm, xp = self._firing_strengths(x)       # (B,R), (B,R), (B,P)

        # Cache raw_w so train_DDA can compute L1 sparsity loss without
        # an extra forward pass (no new computation, just a reference).
        self._last_raw_w = raw_w

        # ① Fuzzy Dropout: randomly silence rules during training
        w_drop = self.rule_dropout(w_norm)                  # (B, R)

        # Residual: consequent(fuzzy) + direct projection of input
        out = self.consequent(w_drop) + self.res_proj(xp)   # (B, out)
        return self.out_norm(out)

    # ── interpretability accessor ────────────────────────────────

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return normalised firing strengths without gradient tracking.
        Used by the web app for per-rule visualisation.
        Shape: (B, n_rules)
        """
        with torch.no_grad():
            _, w, _ = self._firing_strengths(x)
            return w.cpu()


# ════════════════════════════════════════════════════════════════
#  FuzzyInputGate  — Lọc nhiễu đặc trưng đầu vào (LopFuzzy style)
# ════════════════════════════════════════════════════════════════

class FuzzyInputGate(nn.Module):
    """
    Cải tiến A: Input-level fuzzy denoising gate (lấy cảm hứng từ LopFuzzy).

    Với mỗi chiều i của đặc trưng đầu vào:
        output_i = x_i * m_i(x_i)
        m_i = α · gauss_i + (1-α) · tri_i

    - Gaussian:   m_gauss_i = exp(-(x_i - μ_i)² / (2σ_i²))
    - Triangular: m_tri_i   = max(0, 1 - |x_i - μ_i| / σ_i)
    - α = sigmoid(log_alpha) — tỉ lệ pha trộn học được

    Ý nghĩa: Chiều nào lệch xa tâm μ (outlier / nhiễu) bị giảm trọng số
    trước khi vào GNN backbone → mạng ít bị nhiễu chi phối.

    Init: log_sigma = 2.0  (σ ≈ 7.4 >> khoảng giá trị input) → gate ≈ 1.0 ban đầu
    → Hành xử như identity lúc đầu, dần học ra cổng lọc phù hợp.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.mu        = nn.Parameter(torch.zeros(in_features))
        # log_sigma = 2.0 → σ ≈ 7.4 → gate ≈ 1 với input chuẩn hóa → near-identity init
        self.log_sigma = nn.Parameter(torch.full((in_features,), 2.0))
        self.log_alpha = nn.Parameter(torch.zeros(1))   # α_init = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, in_features) → (N, in_features)
        Áp dụng fuzzy gate per-dimension.
        """
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)   # (F,)
        diff  = x - self.mu                                   # (N, F)
        # Gaussian gate
        gauss = torch.exp(-(diff / sigma).pow(2) / 2)        # (N, F)
        # Triangular gate
        tri   = (1.0 - diff.abs() / sigma).clamp(min=0.0)    # (N, F)
        # Learnable blend
        alpha = torch.sigmoid(self.log_alpha)                 # scalar
        m     = alpha * gauss + (1.0 - alpha) * tri           # (N, F)
        return x * m


# ════════════════════════════════════════════════════════════════
#  AMNTDDA_Fuzzy  — kế thừa AMNTDDA, chỉ thêm Fuzzy layer mới
# ════════════════════════════════════════════════════════════════

class AMNTDDA_Fuzzy(AMNTDDA):
    """
    Mô hình cải tiến kế thừa toàn bộ backbone từ AMNTDDA gốc.

    __init__ gọi super().__init__(args) để lấy tất cả các lớp backbone,
    sau đó bổ sung thêm LearnableFuzzyLayer và fuzzy_correction.

    forward tái sử dụng các lớp của parent (gt_drug, gt_disease, hgt,
    drug_trans, disease_trans, mlp) và chèn bước xử lý Fuzzy vào giữa
    interaction vector và MLP.

    Tham số thêm (có giá trị mặc định):
      args.fuzzy_rules    = 32   số luật TSK
      args.fuzzy_dim      = 256  chiều output của fuzzy layer
      args.fuzzy_proj_dim = 64   chiều proj bên trong LearnableFuzzyLayer
    """

    def __init__(self, args):
        # ── Khởi tạo toàn bộ backbone từ AMNTDDA gốc ──────────────
        super().__init__(args)

        # ── Vấn đề 2: Xoá Dead Code (không sửa file gốc) ──────────
        # AMNTDDA.__init__ tạo drug_tr và disease_tr (2 nn.Transformer full)
        # nhưng AMNTDDA.forward() không bao giờ dùng chúng.
        # Xoá ngay sau super().__init__ để giải phóng ~84MB VRAM.
        del self.drug_tr
        del self.disease_tr

        interact_dim = args.gt_out_dim * 2          # 400 (giống baseline)

        # ── Thành phần mới: Fuzzy layer + residual correction ─────
        fuzzy_rules    = getattr(args, 'fuzzy_rules',    32)
        fuzzy_dim      = getattr(args, 'fuzzy_dim',     256)
        fuzzy_proj_dim = getattr(args, 'fuzzy_proj_dim', 64)
        fuzzy_dropout  = getattr(args, 'fuzzy_dropout', 0.05)
        self.fuzzy_layer = LearnableFuzzyLayer(
            interact_dim, n_rules=fuzzy_rules,
            out_features=fuzzy_dim, proj_dim=fuzzy_proj_dim,
            fuzzy_dropout=fuzzy_dropout)

        # Residual correction: chiếu fuzzy_out về interact_dim rồi cộng vào.
        # Init std nhỏ → ban đầu correction ≈ 0, model hành xử như baseline.
        self.fuzzy_correction = nn.Linear(fuzzy_dim, interact_dim, bias=False)
        nn.init.normal_(self.fuzzy_correction.weight, std=0.01)

        # ── Cải tiến A: FuzzyInputGate (LopFuzzy — lọc nhiễu feature đầu vào) ──
        # Áp dụng trước drug_linear / protein_linear / HGT, lọc outlier per-dim.
        # drug: 300-dim (mol2vec), disease: 64-dim (DiseaseFeature), protein: 320-dim (ESM)
        self.drug_gate    = FuzzyInputGate(300)
        self.disease_gate = FuzzyInputGate(args.hgt_in_dim)   # 64-dim
        self.protein_gate = FuzzyInputGate(320)

        # ── Cải tiến C: Bilinear Pair Interaction (học từ FuzzyGCN) ────────────
        # Thay vì interact = dr * di (Hadamard, 400-dim),
        # dùng [dr ‖ di ‖ dr⊙di] (1200-dim) → proj 400 → phong phú hơn.
        # Xavier init → bắt đầu ổn định, nhanh hội tụ.
        self.pair_proj = nn.Linear(interact_dim * 3, interact_dim, bias=False)
        nn.init.xavier_uniform_(self.pair_proj.weight)

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature, sample,
                is_finetuning: bool = False):
        """
        is_finetuning=False (mặc định): giống baseline, .detach() chặn gradient
            đi qua fuzzy layer về backbone (ổn định, dùng trong warmup).
        is_finetuning=True:  tháo .detach() để fuzzy layer và backbone
            co-adapt với nhau — bật sau fuzzy_warmup epochs.
        """
        # ── Backbone + interaction vector ────────────────────────
        dr, interact = self._get_interact(
            drdr_graph, didi_graph, drdipr_graph,
            drug_feature, disease_feature, protein_feature, sample)  # (B, 400)

        # ── Phần cải tiến: Fuzzy residual correction ─────────────
        # Vấn đề 1: is_finetuning=True → tháo .detach() để gradient từ
        # fuzzy layer chạy ngược về backbone (co-adaptation).
        fuzzy_in   = interact if is_finetuning else interact.detach()
        fuzzy_out  = self.fuzzy_layer(fuzzy_in)                  # (B, 256)
        correction = self.fuzzy_correction(fuzzy_out)            # (B, 400)
        enhanced   = interact + correction                       # (B, 400)

        # ── MLP từ AMNTDDA gốc — input dim 400 giữ nguyên ────────
        output = self.mlp(enhanced)                              # (B, 2)

        return dr, output

    # ── Helpers ──────────────────────────────────────────────────

    def _get_embeddings(self, drdr_graph, didi_graph, drdipr_graph,
                        drug_feature, disease_feature, protein_feature):
        """
        Chạy toàn bộ GNN backbone (GT-Drug, GT-Disease, HGT, Transformer fusion)
        và trả về embedding đầy đủ:
          dr : (N_drug,    gt_out_dim*2)
          di : (N_disease, gt_out_dim*2)

        Tách riêng khỏi _get_interact để train_DDA có thể gọi một lần duy nhất
        rồi chunk phần sample-level (interact → fuzzy → MLP) → giải quyết
        Vấn đề 3: tránh OOM khi is_finetuning=True.
        """
        # Cải tiến A: FuzzyInputGate lọc nhiễu trước khi vào backbone
        drug_feature    = self.drug_gate(drug_feature)
        disease_feature = self.disease_gate(disease_feature)
        protein_feature = self.protein_gate(protein_feature)

        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature    = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature, 'disease': disease_feature, 'protein': protein_feature
        }
        drdipr_graph.ndata['h'] = feature_dict
        g       = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:
                         self.args.disease_number + self.args.drug_number, :]

        dr = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))
        di = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1))
        dr = dr.view(self.args.drug_number,    2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)
        return dr, di

    def _get_interact(self, drdr_graph, didi_graph, drdipr_graph,
                      drug_feature, disease_feature, protein_feature,
                      sample):
        """Trả về (dr, interact) — dùng cho forward() và get_firing_strengths()."""
        dr, di   = self._get_embeddings(drdr_graph, didi_graph, drdipr_graph,
                                        drug_feature, disease_feature, protein_feature)
        # Cải tiến C: Bilinear pair interaction [dr ‖ di ‖ dr⊙di] → proj 400
        dr_s = dr[sample[:, 0]]                                # (B, 400)
        di_s = di[sample[:, 1]]                                # (B, 400)
        interact = self.pair_proj(
            torch.cat([dr_s, di_s, dr_s * di_s], dim=-1))     # (B, 400)
        return dr, interact

    # ── Interpretability ─────────────────────────────────────────

    def get_firing_strengths(self, drdr_graph, didi_graph, drdipr_graph,
                             drug_feat, disease_feat, protein_feat,
                             sample) -> torch.Tensor:
        """
        Return normalised fuzzy firing strengths (no grad) for given samples.
        Used by the web app to render the animated rule-activation heatmap.
        Shape: (n_samples, n_rules)
        """
        with torch.no_grad():
            _, interact = self._get_interact(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feat, disease_feat, protein_feat, sample)
            return self.fuzzy_layer.firing_strengths(interact)
