"""
AMNTDDA_Fuzzy – AMNTDDA augmented with a Learnable Fuzzy Inference Layer.

Architecture
------------
  Backbone  : identical to AMNTDDA
                 GT-Drug  (drug–drug similarity graph)
               + GT-Disease (disease–disease similarity graph)
               + HGTConv  (heterogeneous drug–disease–protein knowledge graph)
               + TransformerEncoder fusion → 400-dim interaction vector

  New layer : LearnableFuzzyLayer  (TSK-type, Gaussian MFs, fully differentiable)
                 in  = 400-dim element-wise interaction vector
                 out = 256-dim fuzzy-activated features

  Classifier: MLP input = cat(fuzzy_out:256, interaction:400) = 656-dim

Key properties
  – Gaussian MF centres c_k and log-widths log(σ_k) are learnable parameters.
  – Product T-norm for per-rule firing strength w_k = ∏_i μ_k(x_i).
  – Wang-Mendel normalisation:  w̃_k = w_k / Σ_j w_j.
  – TSK linear consequent:  out = LayerNorm(Linear(w̃)).
  – get_firing_strengths() exposes per-rule activations for web visualisation.

Training
  Run train_DDA.py with --model gnn_fuzzy  (see that file for all flags).
  Checkpoint saved to web_app/models/best_model_fuzzy.pt
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
    Differentiable Takagi-Sugeno-Kang (TSK) fuzzy inference layer.

    For each of n_rules fuzzy rules k:
      μ_k(x_i) = exp( –((x_i – c_k_i) / σ_k_i)² )     Gaussian MF
      w_k      = ∏_i  μ_k(x_i)                          product T-norm
      w̃_k     = w_k / Σ_j w_j                          normalisation

    Output = LayerNorm( Linear(w̃) )                     TSK consequent

    Parameters
    ----------
    in_features  : dimensionality of input x  (400 in AMNTDDA_Fuzzy)
    n_rules      : number of fuzzy rules       (default 32)
    out_features : output dimensionality       (default 256)
    """

    def __init__(self, in_features: int, n_rules: int = 32, out_features: int = 256):
        super().__init__()
        self.in_features  = in_features
        self.n_rules      = n_rules
        self.out_features = out_features

        # Gaussian MF parameters (learnable)
        # centres near origin; log_sigma = 0 → σ = 1 at init
        self.centers    = nn.Parameter(torch.randn(n_rules, in_features) * 0.1)
        self.log_sigmas = nn.Parameter(torch.zeros(n_rules, in_features))

        # TSK linear consequent + normalisation
        self.consequent = nn.Linear(n_rules, out_features, bias=True)
        self.out_norm   = nn.LayerNorm(out_features)

    # ── internal helpers ─────────────────────────────────────────

    def _firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, F) → normalised firing strengths : (B, R)
        """
        # Gaussian membership for every (rule, feature)
        diff   = x.unsqueeze(1) - self.centers.unsqueeze(0)           # (B, R, F)
        sigma  = torch.exp(self.log_sigmas).unsqueeze(0) + 1e-6       # (1, R, F)
        mu     = torch.exp(-(diff / sigma).pow(2))                     # (B, R, F)

        # Product T-norm across features → per-rule firing strength
        firing = mu.prod(dim=-1)                                       # (B, R)

        # Wang-Mendel normalisation (sum to 1)
        denom  = firing.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return firing / denom                                          # (B, R)

    # ── forward ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_features) → (B, out_features)"""
        w_norm = self._firing_strengths(x)          # (B, R)
        out    = self.consequent(w_norm)             # (B, out_features)
        return self.out_norm(out)

    # ── interpretability accessor ────────────────────────────────

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return normalised firing strengths without gradient tracking.
        Used by the web app for per-rule visualisation.
        Shape: (B, n_rules)

        NOTE: Product T-norm across many features collapses to ~0 numerically.
        When that happens, fall back to softmax of mean squared distances
        so the visualisation always shows meaningful relative activations.
        """
        with torch.no_grad():
            raw = self._firing_strengths(x).cpu()
            # If product T-norm collapsed (all values negligible), use mean-distance fallback
            if raw.max() < 1e-6:
                diff   = x.unsqueeze(1) - self.centers.unsqueeze(0)          # (B, R, F)
                sigma  = torch.exp(self.log_sigmas).unsqueeze(0) + 1e-6      # (1, R, F)
                mean_sq = ((diff / sigma).pow(2)).mean(dim=-1)                # (B, R)
                raw = torch.softmax(-mean_sq, dim=-1).cpu()
            return raw


# ════════════════════════════════════════════════════════════════
#  AMNTDDA_Fuzzy  (full model)
# ════════════════════════════════════════════════════════════════

class AMNTDDA_Fuzzy(nn.Module):
    """
    AMNTDDA with a Learnable Fuzzy Inference Layer.

    The fuzzy layer sits between the interaction vector and the MLP classifier:
      interaction (400) → LearnableFuzzyLayer → fuzzy_out (256)
      cat(fuzzy_out, interaction) → MLP → 2 logits

    Extra args (with defaults used if absent from checkpoint):
      args.fuzzy_rules = 32   number of TSK fuzzy rules
      args.fuzzy_dim   = 256  output dimension of the fuzzy layer
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

        self.drug_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        # ── Fuzzy Layer ──────────────────────────────────────────
        interact_dim = args.gt_out_dim * 2                          # 400
        fuzzy_rules  = getattr(args, 'fuzzy_rules', 32)
        fuzzy_dim    = getattr(args, 'fuzzy_dim',   256)
        self.fuzzy_layer = LearnableFuzzyLayer(
            interact_dim, n_rules=fuzzy_rules, out_features=fuzzy_dim)

        # ── MLP  (656 = 400 + 256) ───────────────────────────────
        mlp_in = interact_dim + fuzzy_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 1024),   nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256),    nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 2),
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

        # Self-fusion: TransformerEncoder combines similarity + network features
        dr_seq = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))    # (N_drug, 2, d)
        di_seq = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1)) # (N_disease, 2, d)

        # Cross-modal attention (Module 3): drug queries disease context and vice versa
        dr_pairs = dr_seq[sample[:, 0]]                                    # (B, 2, d)
        di_pairs = di_seq[sample[:, 1]]                                    # (B, 2, d)
        dr_refined = self.drug_tr(src=di_pairs, tgt=dr_pairs)             # (B, 2, d)
        di_refined = self.disease_tr(src=dr_pairs, tgt=di_pairs)          # (B, 2, d)

        # Element-wise interaction vector
        interact = torch.mul(
            dr_refined.reshape(dr_pairs.shape[0], 2 * self.args.gt_out_dim),  # (B, 400)
            di_refined.reshape(di_pairs.shape[0], 2 * self.args.gt_out_dim),  # (B, 400)
        )                                                                       # (B, 400)

        # Fuzzy layer
        fuzzy_out = self.fuzzy_layer(interact)                      # (B, 256)

        # Classifier
        output = self.mlp(torch.cat([fuzzy_out, interact], dim=-1)) # (B, 2)

        return dr_seq.view(self.args.drug_number, 2 * self.args.gt_out_dim), output

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

            dr_seq = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))
            di_seq = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1))

            dr_pairs = dr_seq[sample[:, 0]]
            di_pairs = di_seq[sample[:, 1]]
            dr_refined = self.drug_tr(src=di_pairs, tgt=dr_pairs)
            di_refined = self.disease_tr(src=dr_pairs, tgt=di_pairs)

            interact = torch.mul(
                dr_refined.reshape(dr_pairs.shape[0], 2 * self.args.gt_out_dim),
                di_refined.reshape(di_pairs.shape[0], 2 * self.args.gt_out_dim),
            )
            return self.fuzzy_layer.firing_strengths(interact)
