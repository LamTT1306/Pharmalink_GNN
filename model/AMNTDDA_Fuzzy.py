"""
AMNTDDA_Fuzzy v2 – AMNTDDA augmented with an Enhanced Learnable Fuzzy Inference Layer.

Architecture
------------
  Backbone  : identical to AMNTDDA
                 GT-Drug  (drug–drug similarity graph)
               + GT-Disease (disease–disease similarity graph)
               + HGTConv  (heterogeneous drug–disease–protein knowledge graph)
               + TransformerEncoder fusion

  Interaction: Rich vector cat(dr*di, |dr-di|) → Linear(800→400) + GELU + LN
                 Element-wise product captures co-activation;
                 absolute difference captures asymmetric/contrastive signal.

  New layer : LearnableFuzzyLayer v2  (TSK-type, Gaussian MFs, fully differentiable)
                 in       = 400-dim rich interaction vector
                 proj_dim = 64   (projects input BEFORE T-norm to prevent collapse)
                 out      = 256-dim fuzzy-activated features + residual from projection

  Classifier: MLP input = cat(fuzzy_out:256, interaction:400) = 656-dim

Key improvements over v1
  ① Input projection 400→proj_dim BEFORE T-norm.
      Problem solved: product T-norm over 400 dims collapses to ~0 for all rules,
      meaning the fuzzy layer learned nothing.  Projecting to 64 dims first keeps
      firing strengths in a healthy range (0.1–0.9 per rule).
  ② RBF mean T-norm: w_k = exp(−mean_i((x_i−c_ki)²/σ_ki²))
      Numerically stable; still a valid similarity measure.  Replaces product T-norm.
  ③ Residual connection: out = LN(consequent(w̃) + res_proj(x_proj))
      Prevents gradient vanishing through the fuzzy layer.
  ④ Rich interaction vector cat(mul, diff) replaces plain element-wise product.
  ⑤ GELU activations and reduced dropout in classifier MLP.

Training
  Run train_DDA.py with --model gnn_fuzzy  (see that file for all flags).
  Extra args: --fuzzy_rules 32  --fuzzy_dim 256  --fuzzy_proj_dim 64
  Checkpoint saved to web_app/models/best_model_fuzzy_{dataset}.pt
"""

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease


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
        T       = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)  # scalar
        neg_sq  = -(diff / sigma).pow(2) / T                             # (B, R, P)
        raw_w   = torch.exp(neg_sq.mean(dim=-1))                         # (B, R) pre-norm

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
#  AMNTDDA_Fuzzy  (full model)
# ════════════════════════════════════════════════════════════════

class AMNTDDA_Fuzzy(nn.Module):
    """
    AMNTDDA_Fuzzy v2 with improved fuzzy layer and richer interaction vector.

    The fuzzy layer sits between the interaction vector and the MLP classifier:
      rich_interact (400) → LearnableFuzzyLayer v2 → fuzzy_out (256)
      cat(fuzzy_out, rich_interact) → MLP → 2 logits

    Interaction vector (④):
      Instead of plain dr*di, use cat(dr*di, |dr-di|)→ Linear(800→400).
      Element-wise product captures co-activation;
      absolute difference captures asymmetric/contrastive signal.

    Extra args (with defaults used if absent from checkpoint):
      args.fuzzy_rules    = 32   number of TSK fuzzy rules
      args.fuzzy_dim      = 256  output dimension of the fuzzy layer
      args.fuzzy_proj_dim = 64   projection dim inside LearnableFuzzyLayer
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Backbone (identical to AMNTDDA) ──────────────────────
        self.drug_linear    = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        self.gt_drug = gt_net_drug.GraphTransformer(
            _device, args.gt_layer, args.drug_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(
            _device, args.gt_layer, args.disease_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head),
            args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim, args.hgt_head_dim,
            args.hgt_head, 3, 3, args.dropout)
        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer - 1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer      = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans    = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        interact_dim = args.gt_out_dim * 2                           # 400

        # ── Fuzzy Layer (v3) ─────────────────────────────────────
        fuzzy_rules    = getattr(args, 'fuzzy_rules',    32)
        fuzzy_dim      = getattr(args, 'fuzzy_dim',     256)
        fuzzy_proj_dim = getattr(args, 'fuzzy_proj_dim', 64)
        fuzzy_dropout  = getattr(args, 'fuzzy_dropout', 0.05)
        self.fuzzy_layer = LearnableFuzzyLayer(
            interact_dim, n_rules=fuzzy_rules,
            out_features=fuzzy_dim, proj_dim=fuzzy_proj_dim,
            fuzzy_dropout=fuzzy_dropout)

        # ── Fuzzy residual correction (fuzzy_dim → interact_dim) ─
        # Projects fuzzy features back to interaction space as a residual
        # correction.  Small-std init so correction ≈ 0 at start (model
        # behaves like baseline until fuzzy layer warms up).
        self.fuzzy_correction = nn.Linear(fuzzy_dim, interact_dim, bias=False)
        nn.init.normal_(self.fuzzy_correction.weight, std=0.01)

        # ── MLP — identical structure to AMNTDDA baseline ────────
        # Input dim kept at 400 (same as baseline) so parameter count
        # and regularisation exactly match the proven configuration.
        self.mlp = nn.Sequential(
            nn.Linear(interact_dim, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 1024),         nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256),          nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,  2),
        )

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature, sample):

        # Similarity graph encoders
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        # HGT over heterogeneous graph
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

        # Self-fusion — identical to AMNTDDA baseline
        dr = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))        # (N_drug, 2, d)
        di = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1))     # (N_disease, 2, d)
        dr = dr.view(self.args.drug_number,    2 * self.args.gt_out_dim)  # (N_drug, 400)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)  # (N_disease, 400)

        # Interaction: element-wise product — identical to AMNTDDA baseline
        interact = dr[sample[:, 0]] * di[sample[:, 1]]                    # (B, 400)

        # Fuzzy residual correction — gradient-decoupled.
        # detach() keeps backbone gradient identical to baseline;
        # fuzzy_lr_ratio=1.0 lets the fuzzy layer learn fast independently.
        fuzzy_out  = self.fuzzy_layer(interact.detach())                  # (B, 256)
        correction = self.fuzzy_correction(fuzzy_out)                     # (B, 400)
        enhanced   = interact + correction                                 # (B, 400)

        # Classifier: same 400-dim input as AMNTDDA baseline
        output = self.mlp(enhanced)                                        # (B, 2)

        return dr, output

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
            # Similarity encoders
            dr_sim = self.gt_drug(drdr_graph)
            di_sim = self.gt_disease(didi_graph)

            # HGT
            df = self.drug_linear(drug_feat)
            pf = self.protein_linear(protein_feat)
            drdipr_graph.ndata['h'] = {'drug': df, 'disease': disease_feat, 'protein': pf}
            g  = dgl.to_homogeneous(drdipr_graph, ndata='h')
            f  = torch.cat((df, disease_feat, pf), dim=0)
            for layer in self.hgt:
                hgt_out = layer(g, f, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
                f = hgt_out

            dr_hgt = hgt_out[:self.args.drug_number, :]
            di_hgt = hgt_out[self.args.drug_number:
                             self.args.disease_number + self.args.drug_number, :]

            dr = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))
            di = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1))
            dr = dr.view(self.args.drug_number,    2 * self.args.gt_out_dim)
            di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)
            interact = dr[sample[:, 0]] * di[sample[:, 1]]

            return self.fuzzy_layer.firing_strengths(interact)
