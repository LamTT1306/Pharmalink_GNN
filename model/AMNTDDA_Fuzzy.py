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
    Improved Differentiable TSK Fuzzy Inference Layer (v2).

    Changes from v1:
      – input_proj   : Linear(in_features→proj_dim) + GELU + LN  (prevents T-norm collapse)
      – RBF mean T-norm: w_k = exp(−mean((x−c)²/σ²))  [numerically stable, always ∈(e⁻¹,1]]
      – residual     : out = LN(consequent(w̃) + res_proj(x_proj))

    Parameters
    ----------
    in_features  : input dimensionality   (400 in AMNTDDA_Fuzzy)
    n_rules      : number of fuzzy rules  (default 32)
    out_features : output dimensionality  (default 256)
    proj_dim     : projected dim before T-norm to prevent collapse (default 64)
    """

    def __init__(self, in_features: int, n_rules: int = 32,
                 out_features: int = 256, proj_dim: int = 64):
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

        # TSK consequent + residual connection
        self.consequent = nn.Linear(n_rules, out_features, bias=True)
        self.res_proj   = nn.Linear(proj_dim, out_features, bias=False)
        self.out_norm   = nn.LayerNorm(out_features)

    # ── internal helpers ─────────────────────────────────────────

    def _firing_strengths(self, x: torch.Tensor):
        """
        x : (B, in_features) → (w_norm: (B, R), xp: (B, proj_dim))

        ② RBF mean T-norm (numerically stable):
           w_k = exp(−mean_i((x_i − c_ki)² / σ_ki²))
           This is always in (e⁻¹, 1] — no collapse even with high-dim input.
        """
        xp    = self.input_proj(x)                                           # (B, P)
        diff  = xp.unsqueeze(1) - self.centers.unsqueeze(0)                 # (B, R, P)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0).clamp(min=1e-3)    # (1, R, P)

        # Mean of negative squared distances (RBF-style), then exp
        neg_sq   = -(diff / sigma).pow(2)                # (B, R, P)
        firing   = torch.exp(neg_sq.mean(dim=-1))        # (B, R)  ∈ (e⁻¹, 1]

        # Wang-Mendel normalisation
        denom    = firing.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return firing / denom, xp                        # (B, R), (B, P)

    # ── forward ─────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_features) → (B, out_features)"""
        w_norm, xp = self._firing_strengths(x)                       # (B,R), (B,P)
        # ③ Residual: consequent(fuzzy) + direct projection of input
        out = self.consequent(w_norm) + self.res_proj(xp)            # (B, out)
        return self.out_norm(out)

    # ── interpretability accessor ────────────────────────────────

    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return normalised firing strengths without gradient tracking.
        Used by the web app for per-rule visualisation.
        Shape: (B, n_rules)
        """
        with torch.no_grad():
            w, _ = self._firing_strengths(x)
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

        self.drug_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        interact_dim = args.gt_out_dim * 2                           # 400

        # ④ Rich interaction projection: cat(mul, diff) 800 → 400
        self.interact_proj = nn.Sequential(
            nn.Linear(interact_dim * 2, interact_dim),
            nn.GELU(),
            nn.LayerNorm(interact_dim),
        )

        # ── Fuzzy Layer (v2) ─────────────────────────────────────
        fuzzy_rules    = getattr(args, 'fuzzy_rules',    32)
        fuzzy_dim      = getattr(args, 'fuzzy_dim',     256)
        fuzzy_proj_dim = getattr(args, 'fuzzy_proj_dim', 64)
        self.fuzzy_layer = LearnableFuzzyLayer(
            interact_dim, n_rules=fuzzy_rules,
            out_features=fuzzy_dim, proj_dim=fuzzy_proj_dim)

        # ── MLP  (656 = 400 + 256), GELU + reduced dropout ⑤ ────
        mlp_in = interact_dim + fuzzy_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),    nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512,  256),    nn.GELU(), nn.Dropout(0.2),
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

        # Self-fusion: TransformerEncoder combines similarity + network features
        dr_seq = self.drug_trans(torch.stack((dr_sim, dr_hgt), dim=1))    # (N_drug, 2, d)
        di_seq = self.disease_trans(torch.stack((di_sim, di_hgt), dim=1)) # (N_disease, 2, d)

        # Cross-modal attention: drug queries disease context and vice versa
        dr_pairs = dr_seq[sample[:, 0]]                                    # (B, 2, d)
        di_pairs = di_seq[sample[:, 1]]                                    # (B, 2, d)
        dr_refined = self.drug_tr(src=di_pairs, tgt=dr_pairs)             # (B, 2, d)
        di_refined = self.disease_tr(src=dr_pairs, tgt=di_pairs)          # (B, 2, d)

        B = dr_pairs.shape[0]
        d2 = 2 * self.args.gt_out_dim
        dr_flat = dr_refined.reshape(B, d2)                               # (B, 400)
        di_flat = di_refined.reshape(B, d2)                               # (B, 400)

        # ④ Rich interaction: element-wise product + absolute difference → project
        interact_mul  = dr_flat * di_flat                                  # (B, 400)
        interact_diff = (dr_flat - di_flat).abs()                          # (B, 400)
        interact = self.interact_proj(
            torch.cat([interact_mul, interact_diff], dim=-1))              # (B, 400)

        # Fuzzy layer (v2)
        fuzzy_out = self.fuzzy_layer(interact)                             # (B, 256)

        # Classifier
        output = self.mlp(torch.cat([fuzzy_out, interact], dim=-1))       # (B, 2)

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

            B  = dr_pairs.shape[0]
            d2 = 2 * self.args.gt_out_dim
            dr_flat = dr_refined.reshape(B, d2)
            di_flat = di_refined.reshape(B, d2)
            interact = self.interact_proj(
                torch.cat([dr_flat * di_flat, (dr_flat - di_flat).abs()], dim=-1))

            return self.fuzzy_layer.firing_strengths(interact)
