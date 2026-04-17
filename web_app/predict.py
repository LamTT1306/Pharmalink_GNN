"""
Prediction engine for Drug-Disease-Protein Association.

Three models available:
  1. Similarity CF  – always available, fast (matrix-based collaborative filtering)
  2. Fuzzy Logic    – always available, Mamdani FIS (pure NumPy, no deps)
  3. GNN (AMNTDDA)  – available when web_app/models/best_model.pt exists
"""
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ════════════════════════════════════════════════════════════════
#  Mamdani Fuzzy Inference System
# ════════════════════════════════════════════════════════════════
class FuzzyPredictor:
    """
    Mamdani FIS for drug-disease and drug/disease-protein association.

    Inputs (each 0-1):
      cf_score      : collaborative-filtering score
      src_nbr_score : max similarity of query entity to its neighbors
                      that are known to associate with the target
      tgt_nbr_score : max similarity of target entity to its neighbors
                      that are known to associate with the query

    Output (0-1): fuzzy association confidence
    """

    @staticmethod
    def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        left  = np.where(b > a, (x - a) / (b - a + 1e-10), (x >= b).astype(float))
        right = np.where(c > b, (c - x) / (c - b + 1e-10), (x <= b).astype(float))
        return np.clip(np.minimum(left, right), 0, 1)

    def _mf_low(self, x):  return self._trimf(x, 0.0,  0.0,  0.45)
    def _mf_mid(self, x):  return self._trimf(x, 0.25, 0.5,  0.75)
    def _mf_high(self, x): return self._trimf(x, 0.55, 1.0,  1.0)

    # Output singleton centroids: [very_low, low, medium, high, very_high]
    _OUT = np.array([0.10, 0.30, 0.50, 0.70, 0.90])

    def infer(self,
              cf: np.ndarray,
              src_nbr: np.ndarray,
              tgt_nbr: np.ndarray) -> np.ndarray:
        """Vectorised fuzzy inference. Inputs are arrays of shape (N,)."""
        cf_lo, cf_md, cf_hi = self._mf_low(cf),      self._mf_mid(cf),      self._mf_high(cf)
        sn_lo, sn_md, sn_hi = self._mf_low(src_nbr), self._mf_mid(src_nbr), self._mf_high(src_nbr)
        tn_lo, tn_md, tn_hi = self._mf_low(tgt_nbr), self._mf_mid(tgt_nbr), self._mf_high(tgt_nbr)

        AND = np.minimum
        rules = np.stack([
            AND(AND(cf_hi, sn_hi), tn_hi),    # → very_high (0.90)
            AND(cf_hi, sn_hi),                # → high      (0.70)
            AND(cf_hi, AND(sn_md, tn_md)),    # → high      (0.70)
            AND(cf_md, AND(sn_hi, tn_hi)),    # → high      (0.70)
            AND(cf_hi, sn_lo),                # → medium    (0.50)
            AND(cf_md, AND(sn_hi, tn_md)),    # → medium    (0.50)
            AND(cf_md, sn_md),                # → medium    (0.50)
            AND(cf_lo, sn_hi),                # → medium    (0.50)
            AND(cf_md, sn_lo),                # → low       (0.30)
            AND(cf_lo, sn_md),                # → low       (0.30)
            AND(cf_lo, AND(sn_lo, tn_lo)),    # → very_low  (0.10)
        ], axis=0)  # (11, N)

        out_idx = np.array([4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0])
        numer = np.einsum('rn,r->n', rules, self._OUT[out_idx])
        denom = rules.sum(axis=0)
        return np.where(denom < 1e-10, cf, numer / denom)


# ════════════════════════════════════════════════════════════════
#  GNN Inference wrapper (AMNTDDA)
# ════════════════════════════════════════════════════════════════
class GNNInference:
    """
    Wraps the AMNTDDA model for live inference.
    Graphs and features are built once on first call.
    Requires: PyTorch, DGL, and a saved checkpoint at model_path.
    """

    def __init__(self, data_dir: str, model_path: str):
        self.data_dir        = data_dir
        self.model_path      = model_path
        self.ready           = False
        self._auc            = 0.0
        self.dataset_name    = None   # auto-detected from checkpoint
        self._try_init()

    def _try_init(self):
        if not os.path.exists(self.model_path):
            print('[GNN] No saved model found. GNN unavailable.')
            return
        try:
            import torch
            import torch.nn.functional as F
            import dgl
            self._torch = torch
            self._F     = F

            ckpt = torch.load(self.model_path, map_location='cpu')
            args = ckpt['args']
            self._n_dr = args.drug_number
            self._n_di = args.disease_number
            self._n_pr = args.protein_number

            # Auto-detect dataset from checkpoint
            ckpt_data_dir = getattr(args, 'data_dir', '').replace('\\', '/').strip('/')
            detected = None
            for ds in ('B-dataset', 'C-dataset', 'F-dataset'):
                if ds.lower().replace('-', '_') in ckpt_data_dir.lower().replace('-', '_'):
                    detected = ds
                    break
            self.dataset_name = detected
            use_data_dir = os.path.join(BASE_DIR, 'data', detected) if detected else self.data_dir

            sys.path.insert(0, BASE_DIR)
            from data_preprocess import get_data, data_processing, dgl_similarity_graph, dgl_heterograph

            class _Args:
                pass
            a = _Args()
            a.__dict__.update(vars(args))
            a.data_dir      = use_data_dir + os.sep
            a.negative_rate = 1.0
            a.random_seed   = 1234
            a.neighbor      = getattr(args, 'neighbor', 20)

            raw = get_data(a)
            raw = data_processing(raw, a)
            self._drdr_graph, self._didi_graph, _ = dgl_similarity_graph(raw, a)
            self._drdipr_graph, _ = dgl_heterograph(raw, raw['all_drdi'], a)

            self._drug_feat    = torch.FloatTensor(raw['drugfeature'])
            self._disease_feat = torch.FloatTensor(raw['diseasefeature'])
            self._protein_feat = torch.FloatTensor(raw['proteinfeature'])

            from model.AMNTDDA import AMNTDDA
            self._model = AMNTDDA(args)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.eval()
            # Force all submodule device attributes to CPU for web inference
            _cpu = torch.device('cpu')
            for mod in self._model.modules():
                if isinstance(getattr(mod, 'device', None), torch.device):
                    mod.device = _cpu
                if isinstance(getattr(mod, '_device', None), torch.device):
                    mod._device = _cpu
            self._auc   = float(ckpt.get('auc', 0))
            self.ready  = True
            print(f'[GNN] AMNTDDA loaded (best AUC: {self._auc:.4f})')
        except Exception as e:
            print(f'[GNN] Cannot load model: {e}')

    def predict_drug(self, drug_idx: int) -> np.ndarray:
        """Return association probability for all diseases. Shape: (n_diseases,)"""
        T = self._torch
        pairs = T.tensor([[drug_idx, j] for j in range(self._n_di)], dtype=T.long)
        with T.no_grad():
            _, logits = self._model(
                self._drdr_graph, self._didi_graph, self._drdipr_graph,
                self._drug_feat, self._disease_feat, self._protein_feat, pairs)
        return self._F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    def predict_disease(self, disease_idx: int) -> np.ndarray:
        """Return association probability for all drugs. Shape: (n_drugs,)"""
        T = self._torch
        pairs = T.tensor([[i, disease_idx] for i in range(self._n_dr)], dtype=T.long)
        with T.no_grad():
            _, logits = self._model(
                self._drdr_graph, self._didi_graph, self._drdipr_graph,
                self._drug_feat, self._disease_feat, self._protein_feat, pairs)
        return self._F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


# ════════════════════════════════════════════════════════════════
#  GNN + Fuzzy Inference wrapper  (AMNTDDA_Fuzzy)
# ════════════════════════════════════════════════════════════════
class GNNFuzzyInference(GNNInference):
    """
    Wraps the AMNTDDA_Fuzzy model for live inference.
    Identical to GNNInference but loads best_model_fuzzy.pt
    and instantiates AMNTDDA_Fuzzy instead of AMNTDDA.

    Extra public method:
      get_firing_strengths(drug_idx, disease_idx) -> list[float]
        Returns the normalised fuzzy rule activations for a single pair.
        Used by the web app to render the animated rule-heatmap.
    """

    def _try_init(self):
        if not os.path.exists(self.model_path):
            print('[GNN+Fuzzy] No checkpoint found. GNN+Fuzzy unavailable.')
            return
        try:
            import torch
            import torch.nn.functional as F
            import dgl
            self._torch = torch
            self._F     = F

            ckpt = torch.load(self.model_path, map_location='cpu')
            args = ckpt['args']
            self._n_dr = args.drug_number
            self._n_di = args.disease_number
            self._n_pr = args.protein_number

            # Auto-detect dataset from checkpoint
            ckpt_data_dir = getattr(args, 'data_dir', '').replace('\\', '/').strip('/')
            detected = None
            for ds in ('B-dataset', 'C-dataset', 'F-dataset'):
                if ds.lower().replace('-', '_') in ckpt_data_dir.lower().replace('-', '_'):
                    detected = ds
                    break
            self.dataset_name = detected
            use_data_dir = os.path.join(BASE_DIR, 'data', detected) if detected else self.data_dir

            sys.path.insert(0, BASE_DIR)
            from data_preprocess import get_data, data_processing, dgl_similarity_graph, dgl_heterograph

            class _Args:
                pass
            a = _Args()
            a.__dict__.update(vars(args))
            a.data_dir      = use_data_dir + os.sep
            a.negative_rate = 1.0
            a.random_seed   = 1234
            a.neighbor      = getattr(args, 'neighbor', 20)

            raw = get_data(a)
            raw = data_processing(raw, a)
            self._drdr_graph, self._didi_graph, _ = dgl_similarity_graph(raw, a)
            self._drdipr_graph, _ = dgl_heterograph(raw, raw['all_drdi'], a)

            self._drug_feat    = torch.FloatTensor(raw['drugfeature'])
            self._disease_feat = torch.FloatTensor(raw['diseasefeature'])
            self._protein_feat = torch.FloatTensor(raw['proteinfeature'])

            from model.AMNTDDA_Fuzzy import AMNTDDA_Fuzzy
            self._model = AMNTDDA_Fuzzy(args)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.eval()
            # Force all submodule device attributes to CPU for web inference
            _cpu = torch.device('cpu')
            for mod in self._model.modules():
                if isinstance(getattr(mod, 'device', None), torch.device):
                    mod.device = _cpu
                if isinstance(getattr(mod, '_device', None), torch.device):
                    mod._device = _cpu
            self._auc   = float(ckpt.get('auc', 0))
            self._n_rules = getattr(args, 'fuzzy_rules', 32)
            self.ready  = True
            print(f'[GNN+Fuzzy] AMNTDDA_Fuzzy loaded (best AUC: {self._auc:.4f})')
        except Exception as e:
            print(f'[GNN+Fuzzy] Cannot load model: {e}')

    def get_firing_strengths(self, drug_idx: int, disease_idx: int) -> list:
        """
        Return normalised fuzzy firing strengths for a single (drug, disease) pair.
        Shape: (n_rules,) as a Python list, suitable for JSON serialisation.
        """
        T = self._torch
        pair = T.tensor([[drug_idx, disease_idx]], dtype=T.long)
        strengths = self._model.get_firing_strengths(
            self._drdr_graph, self._didi_graph, self._drdipr_graph,
            self._drug_feat, self._disease_feat, self._protein_feat, pair)
        return strengths[0].tolist()

    def predict_drug(self, drug_idx: int) -> np.ndarray:
        """Return association probability for all diseases. Shape: (n_diseases,)"""
        T = self._torch
        pairs = T.tensor([[drug_idx, j] for j in range(self._n_di)], dtype=T.long)
        with T.no_grad():
            _, logits = self._model(
                self._drdr_graph, self._didi_graph, self._drdipr_graph,
                self._drug_feat, self._disease_feat, self._protein_feat, pairs)
        return self._F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    def predict_disease(self, disease_idx: int) -> np.ndarray:
        """Return association probability for all drugs. Shape: (n_drugs,)"""
        T = self._torch
        pairs = T.tensor([[i, disease_idx] for i in range(self._n_dr)], dtype=T.long)
        with T.no_grad():
            _, logits = self._model(
                self._drdr_graph, self._didi_graph, self._drdipr_graph,
                self._drug_feat, self._disease_feat, self._protein_feat, pairs)
        return self._F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


# ════════════════════════════════════════════════════════════════
#  Main Prediction Engine
# ════════════════════════════════════════════════════════════════
class PredictionEngine:
    def __init__(self, dataset='C-dataset'):
        self.dataset  = dataset
        self.data_dir = os.path.join(BASE_DIR, 'data', dataset)

        self._fuzzy = FuzzyPredictor()
        self._drug_cf_cache    = None
        self._disease_cf_cache = None
        self._drug_protein_cf_cache    = None
        self._disease_protein_cf_cache = None

        self._load_data()

        _models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        dataset_tag = self.dataset.lower().replace('-', '_')  # e.g. "c_dataset"

        # Load dataset-specific model first, fall back to legacy file
        def _pick(base_name: str) -> str:
            specific = os.path.join(_models_dir, f'{base_name}_{dataset_tag}.pt')
            legacy   = os.path.join(_models_dir, f'{base_name}.pt')
            return specific if os.path.exists(specific) else legacy

        self._gnn       = GNNInference(self.data_dir, _pick('best_model'))
        self._gnn_fuzzy = GNNFuzzyInference(self.data_dir, _pick('best_model_fuzzy'))

        # Disable any model whose dataset doesn't match the requested dataset
        # (keeps main data correct; GNN inference uses its own internal graphs anyway)
        for inf_model, label in ((self._gnn, 'GNN'), (self._gnn_fuzzy, 'GNN+Fuzzy')):
            ds = getattr(inf_model, 'dataset_name', None)
            if inf_model.ready and ds and ds != self.dataset:
                print(f'[Engine] Disabling {label} for {self.dataset} (model trained on {ds})')
                inf_model.ready = False

    # ── Data loading ─────────────────────────────────────────
    def _load_data(self):
        d = self.data_dir

        # Drugs
        drug_info = pd.read_csv(os.path.join(d, 'DrugInformation.csv'))
        self.drug_ids   = drug_info['id'].tolist()
        self.drug_names = drug_info['name'].tolist()
        self.drug_smiles = drug_info['smiles'].tolist() if 'smiles' in drug_info.columns else [''] * len(drug_info)
        self.n_drugs = len(self.drug_ids)

        # Diseases
        df_dis = pd.read_csv(os.path.join(d, 'DiseaseFeature.csv'), header=None)
        raw_codes = [str(c) for c in df_dis.iloc[:, 0].tolist()]
        self.disease_codes = raw_codes
        # If code is OMIM-style (D followed by digits), label as "OMIM:Dxxxxxx"; else use name directly
        self.disease_names = [
            c if not (len(c) > 1 and c[0] == 'D' and c[1:].isdigit()) else f'OMIM:{c}'
            for c in raw_codes
        ]
        self.n_diseases = len(self.disease_codes)

        # Proteins
        prot_info = pd.read_csv(os.path.join(d, 'ProteinInformation.csv'))
        self.protein_ids  = prot_info['id'].tolist()
        self.protein_seqs = prot_info['sequence'].tolist() if 'sequence' in prot_info.columns else [''] * len(prot_info)
        self.n_proteins   = len(self.protein_ids)

        # Drug-Disease adjacency  (n_drugs × n_diseases)
        drdi = pd.read_csv(os.path.join(d, 'DrugDiseaseAssociationNumber.csv'), dtype=int).to_numpy()
        self.drdi_assoc = drdi
        self.adj = np.zeros((self.n_drugs, self.n_diseases), dtype=np.float32)
        for r in drdi:
            if r[0] < self.n_drugs and r[1] < self.n_diseases:
                self.adj[r[0], r[1]] = 1.0

        # Drug-Protein adjacency  (n_drugs × n_proteins)
        drpr = pd.read_csv(os.path.join(d, 'DrugProteinAssociationNumber.csv'), dtype=int).to_numpy()
        self.drpr_assoc = drpr
        self.drpr_adj = np.zeros((self.n_drugs, self.n_proteins), dtype=np.float32)
        for r in drpr:
            if r[0] < self.n_drugs and r[1] < self.n_proteins:
                self.drpr_adj[r[0], r[1]] = 1.0

        # Disease-Protein adjacency  (n_diseases × n_proteins)  columns: disease, protein
        dipr = pd.read_csv(os.path.join(d, 'ProteinDiseaseAssociationNumber.csv'), dtype=int).to_numpy()
        self.dipr_assoc = dipr
        self.dipr_adj = np.zeros((self.n_diseases, self.n_proteins), dtype=np.float32)
        for r in dipr:
            if r[0] < self.n_diseases and r[1] < self.n_proteins:
                self.dipr_adj[r[0], r[1]] = 1.0

        # Drug similarity (clip to n_drugs to guard against CSV size mismatch)
        drf = pd.read_csv(os.path.join(d, 'DrugFingerprint.csv')).iloc[:, 1:].to_numpy()
        drg = pd.read_csv(os.path.join(d, 'DrugGIP.csv')).iloc[:, 1:].to_numpy()
        nd = self.n_drugs
        drf = drf[:nd, :nd] if drf.shape[0] >= nd and drf.shape[1] >= nd else drf
        drg = drg[:nd, :nd] if drg.shape[0] >= nd and drg.shape[1] >= nd else drg
        # Align shapes before element-wise ops
        min_r = min(drf.shape[0], drg.shape[0], nd)
        min_c = min(drf.shape[1], drg.shape[1], nd)
        drf = drf[:min_r, :min_c]; drg = drg[:min_r, :min_c]
        self.drug_sim = np.where(drf == 0, drg, (drf + drg) / 2).astype(np.float32)

        # Disease similarity (clip to n_diseases)
        dip = pd.read_csv(os.path.join(d, 'DiseasePS.csv')).iloc[:, 1:].to_numpy()
        dig = pd.read_csv(os.path.join(d, 'DiseaseGIP.csv')).iloc[:, 1:].to_numpy()
        ndi = self.n_diseases
        min_r2 = min(dip.shape[0], dig.shape[0], ndi)
        min_c2 = min(dip.shape[1], dig.shape[1], ndi)
        dip = dip[:min_r2, :min_c2]; dig = dig[:min_r2, :min_c2]
        self.disease_sim = np.where(dip == 0, dig, (dip + dig) / 2).astype(np.float32)

    # ── CF score matrices (lazy, cached) ─────────────────────
    def _cf_drug_disease(self):
        if self._drug_cf_cache is None:
            raw = self.drug_sim @ self.adj
            mx  = raw.max(axis=1, keepdims=True); mx[mx == 0] = 1.0
            self._drug_cf_cache = (raw / mx).astype(np.float32)
        return self._drug_cf_cache

    def _cf_disease_drug(self):
        if self._disease_cf_cache is None:
            raw = (self.adj.T @ self.disease_sim).T
            mx  = raw.max(axis=1, keepdims=True); mx[mx == 0] = 1.0
            self._disease_cf_cache = (raw / mx).astype(np.float32)
        return self._disease_cf_cache

    def _cf_drug_protein(self):
        if self._drug_protein_cf_cache is None:
            raw = self.drug_sim @ self.drpr_adj
            mx  = raw.max(axis=1, keepdims=True); mx[mx == 0] = 1.0
            self._drug_protein_cf_cache = (raw / mx).astype(np.float32)
        return self._drug_protein_cf_cache

    def _cf_disease_protein(self):
        if self._disease_protein_cf_cache is None:
            raw = self.disease_sim @ self.dipr_adj
            mx  = raw.max(axis=1, keepdims=True); mx[mx == 0] = 1.0
            self._disease_protein_cf_cache = (raw / mx).astype(np.float32)
        return self._disease_protein_cf_cache

    # ── Fuzzy helper ─────────────────────────────────────────
    def _fuzzy_drug_disease(self, drug_idx: int) -> np.ndarray:
        cf      = self._cf_drug_disease()[drug_idx]
        sim_row = self.drug_sim[drug_idx]
        # src_nbr[j]: max drug similarity to drugs known for disease j
        weighted = sim_row[:, None] * self.adj
        known_cnt = self.adj.sum(axis=0)
        src_nbr = np.where(known_cnt > 0, weighted.max(axis=0), 0.0)
        # tgt_nbr[j]: max disease similarity to diseases known for drug_idx
        known_di = self.adj[drug_idx]
        tgt_nbr = (self.disease_sim * known_di[None, :]).max(axis=1) \
                  if known_di.sum() > 0 else np.zeros(self.n_diseases, dtype=np.float32)
        return self._fuzzy.infer(cf, src_nbr, tgt_nbr)

    def _fuzzy_disease_drug(self, disease_idx: int) -> np.ndarray:
        cf      = self._cf_disease_drug()[disease_idx]
        sim_row = self.disease_sim[disease_idx]
        weighted = sim_row[:, None] * self.adj.T
        known_cnt = self.adj.T.sum(axis=0)
        src_nbr = np.where(known_cnt > 0, weighted.max(axis=0), 0.0)
        known_dr = self.adj[:, disease_idx]
        tgt_nbr = (self.drug_sim * known_dr[None, :]).max(axis=1) \
                  if known_dr.sum() > 0 else np.zeros(self.n_drugs, dtype=np.float32)
        return self._fuzzy.infer(cf, src_nbr, tgt_nbr)

    # ── Model dispatcher ─────────────────────────────────────
    def _scores_drug(self, drug_idx: int, model: str) -> np.ndarray:
        # Explicit model requests — only use that model if ready
        if model == 'gnn':
            if self._gnn.ready:
                return self._gnn.predict_drug(drug_idx)
            # GNN not ready (wrong dataset) → fall back to Mamdani fuzzy CF
            return self._fuzzy_drug_disease(drug_idx)
        if model == 'gnn_fuzzy':
            if self._gnn_fuzzy.ready:
                return self._gnn_fuzzy.predict_drug(drug_idx)
            # GNN+Fuzzy not ready → fall back to Mamdani fuzzy CF
            return self._fuzzy_drug_disease(drug_idx)
        if model == 'similarity':
            return self._cf_drug_disease()[drug_idx]
        if model == 'fuzzy':
            return self._fuzzy_drug_disease(drug_idx)
        # Unknown model → best available
        if self._gnn_fuzzy.ready:
            return self._gnn_fuzzy.predict_drug(drug_idx)
        if self._gnn.ready:
            return self._gnn.predict_drug(drug_idx)
        return self._fuzzy_drug_disease(drug_idx)

    def _scores_disease(self, disease_idx: int, model: str) -> np.ndarray:
        if model == 'gnn':
            if self._gnn.ready:
                return self._gnn.predict_disease(disease_idx)
            return self._fuzzy_disease_drug(disease_idx)
        if model == 'gnn_fuzzy':
            if self._gnn_fuzzy.ready:
                return self._gnn_fuzzy.predict_disease(disease_idx)
            return self._fuzzy_disease_drug(disease_idx)
        if model == 'similarity':
            return self._cf_disease_drug()[disease_idx]
        if model == 'fuzzy':
            return self._fuzzy_disease_drug(disease_idx)
        if self._gnn_fuzzy.ready:
            return self._gnn_fuzzy.predict_disease(disease_idx)
        if self._gnn.ready:
            return self._gnn.predict_disease(disease_idx)
        return self._fuzzy_disease_drug(disease_idx)

    def get_fuzzy_firing_strengths(self, drug_idx: int, disease_idx: int) -> list:
        """Return fuzzy rule firing strengths from GNN+Fuzzy model for a pair."""
        if not self._gnn_fuzzy.ready:
            return []
        return self._gnn_fuzzy.get_firing_strengths(drug_idx, disease_idx)

    # ── Public prediction API ────────────────────────────────
    def predict_from_drug(self, drug_idx: int, top_k: int = 10, model: str = 'fuzzy'):
        if drug_idx < 0 or drug_idx >= self.n_drugs:
            return []
        scores = self._scores_drug(drug_idx, model)
        known  = set(np.where(self.adj[drug_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for di in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'disease_idx': int(di),
                'disease_code': self.disease_codes[di], 'disease_name': self.disease_names[di],
                'score': float(round(float(scores[di]), 4)), 'is_known': int(di) in known,
            })
        return results

    def predict_from_disease(self, disease_idx: int, top_k: int = 10, model: str = 'fuzzy'):
        if disease_idx < 0 or disease_idx >= self.n_diseases:
            return []
        scores = self._scores_disease(disease_idx, model)
        known  = set(np.where(self.adj[:, disease_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for dr in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'drug_idx': int(dr),
                'drug_id': self.drug_ids[dr], 'drug_name': self.drug_names[dr],
                'score': float(round(float(scores[dr]), 4)), 'is_known': int(dr) in known,
            })
        return results

    def predict_from_drug_to_proteins(self, drug_idx: int, top_k: int = 10):
        if drug_idx < 0 or drug_idx >= self.n_drugs: return []
        scores = self._cf_drug_protein()[drug_idx]
        known  = set(np.where(self.drpr_adj[drug_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for pr in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'protein_idx': int(pr),
                'protein_id': self.protein_ids[pr],
                'score': float(round(float(scores[pr]), 4)), 'is_known': int(pr) in known,
            })
        return results

    def predict_from_disease_to_proteins(self, disease_idx: int, top_k: int = 10):
        if disease_idx < 0 or disease_idx >= self.n_diseases: return []
        scores = self._cf_disease_protein()[disease_idx]
        known  = set(np.where(self.dipr_adj[disease_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for pr in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'protein_idx': int(pr),
                'protein_id': self.protein_ids[pr],
                'score': float(round(float(scores[pr]), 4)), 'is_known': int(pr) in known,
            })
        return results

    def predict_from_protein_to_drugs(self, protein_idx: int, top_k: int = 10):
        if protein_idx < 0 or protein_idx >= self.n_proteins: return []
        raw = self.drug_sim @ self.drpr_adj[:, protein_idx]
        mx  = raw.max(); scores = (raw / mx) if mx > 0 else raw
        known  = set(np.where(self.drpr_adj[:, protein_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for dr in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'drug_idx': int(dr),
                'drug_id': self.drug_ids[dr], 'drug_name': self.drug_names[dr],
                'score': float(round(float(scores[dr]), 4)), 'is_known': int(dr) in known,
            })
        return results

    def predict_from_protein_to_diseases(self, protein_idx: int, top_k: int = 10):
        if protein_idx < 0 or protein_idx >= self.n_proteins: return []
        raw = self.disease_sim @ self.dipr_adj[:, protein_idx]
        mx  = raw.max(); scores = (raw / mx) if mx > 0 else raw
        known  = set(np.where(self.dipr_adj[:, protein_idx] == 1)[0].tolist())
        order  = np.argsort(-scores)
        results = []
        for di in order:
            if len(results) >= top_k: break
            results.append({
                'rank': len(results) + 1, 'disease_idx': int(di),
                'disease_code': self.disease_codes[di], 'disease_name': self.disease_names[di],
                'score': float(round(float(scores[di]), 4)), 'is_known': int(di) in known,
            })
        return results

    # ── Search ───────────────────────────────────────────────
    def search_drugs(self, query: str, limit: int = 20):
        q = query.lower().strip()
        results = []
        for i, (did, dname) in enumerate(zip(self.drug_ids, self.drug_names)):
            if q in dname.lower() or q in did.lower():
                smiles = self.drug_smiles[i] if i < len(self.drug_smiles) else ''
                results.append({'idx': i, 'id': did, 'name': dname, 'smiles': smiles})
            if len(results) >= limit: break
        return results

    def search_diseases(self, query: str, limit: int = 20):
        q = query.lower().strip()
        results = []
        for i, (code, name) in enumerate(zip(self.disease_codes, self.disease_names)):
            if q in code.lower() or q in name.lower():
                results.append({'idx': i, 'code': code, 'name': name})
            if len(results) >= limit: break
        return results

    def search_proteins(self, query: str, limit: int = 20):
        q = query.lower().strip()
        results = []
        for i, pid in enumerate(self.protein_ids):
            if q in pid.lower():
                results.append({'idx': i, 'id': pid})
            if len(results) >= limit: break
        return results

    # ── Info helpers ─────────────────────────────────────────
    def get_drug_info(self, drug_idx: int):
        if drug_idx < 0 or drug_idx >= self.n_drugs: return None
        return {
            'idx': drug_idx, 'id': self.drug_ids[drug_idx],
            'name': self.drug_names[drug_idx],
            'smiles': self.drug_smiles[drug_idx] if drug_idx < len(self.drug_smiles) else '',
            'known_diseases': list(map(int, np.where(self.adj[drug_idx] == 1)[0])),
            'known_proteins': list(map(int, np.where(self.drpr_adj[drug_idx] == 1)[0])),
        }

    def get_disease_info(self, disease_idx: int):
        if disease_idx < 0 or disease_idx >= self.n_diseases: return None
        return {
            'idx': disease_idx, 'code': self.disease_codes[disease_idx],
            'name': self.disease_names[disease_idx],
            'known_drugs':    list(map(int, np.where(self.adj[:, disease_idx] == 1)[0])),
            'known_proteins': list(map(int, np.where(self.dipr_adj[disease_idx] == 1)[0])),
        }

    def get_protein_info(self, protein_idx: int):
        if protein_idx < 0 or protein_idx >= self.n_proteins: return None
        seq = self.protein_seqs[protein_idx] if protein_idx < len(self.protein_seqs) else ''
        return {
            'idx': protein_idx, 'id': self.protein_ids[protein_idx],
            'sequence': seq[:80] + '…' if len(seq) > 80 else seq,
            'known_drugs':    list(map(int, np.where(self.drpr_adj[:, protein_idx] == 1)[0])),
            'known_diseases': list(map(int, np.where(self.dipr_adj[:, protein_idx] == 1)[0])),
        }

    def get_dataset_info(self):
        return {
            'n_drugs':        self.n_drugs,
            'n_diseases':     self.n_diseases,
            'n_proteins':     self.n_proteins,
            'n_associations': len(self.drdi_assoc),
            'n_drpr':         len(self.drpr_assoc),
            'n_dipr':         len(self.dipr_assoc),
            'dataset':        self.dataset,
            'gnn_ready':      self._gnn.ready,
            'gnn_auc':        self._gnn._auc,
            'models_available': self.available_models(),
        }

    def available_models(self):
        m = []
        if self._gnn.ready:
            m.append('gnn')
        if self._gnn_fuzzy.ready:
            m.append('gnn_fuzzy')
        return m

    def get_gnn_fuzzy_auc(self) -> float:
        return self._gnn_fuzzy._auc if self._gnn_fuzzy.ready else 0.0

    def get_gnn_fuzzy_n_rules(self) -> int:
        return getattr(self._gnn_fuzzy, '_n_rules', 32) if self._gnn_fuzzy.ready else 32

    # ── DB export helpers ────────────────────────────────────
    def as_drugs_list(self):
        return [{'idx': i, 'id': self.drug_ids[i], 'name': self.drug_names[i],
                 'smiles': self.drug_smiles[i] if i < len(self.drug_smiles) else ''}
                for i in range(self.n_drugs)]

    def as_diseases_list(self):
        return [{'idx': i, 'code': self.disease_codes[i], 'name': self.disease_names[i]}
                for i in range(self.n_diseases)]

    def as_proteins_list(self):
        return [{'idx': i, 'id': self.protein_ids[i],
                 'sequence': self.protein_seqs[i] if i < len(self.protein_seqs) else ''}
                for i in range(self.n_proteins)]

    # ── Matrix prediction ─────────────────────────────────────
    def predict_matrix(self, drug_idxs: list, disease_idxs: list, model: str = 'fuzzy') -> dict:
        """Compare N drugs × M diseases. Returns matrix + bipartite graph data."""
        dr_idxs = [i for i in drug_idxs if 0 <= i < self.n_drugs]
        di_idxs = [i for i in disease_idxs if 0 <= i < self.n_diseases]

        score_cache = {dr: self._scores_drug(dr, model) for dr in dr_idxs}

        matrix = []
        for dr in dr_idxs:
            row = []
            for di in di_idxs:
                sc = float(score_cache[dr][di]) if dr in score_cache else 0.0
                row.append({'score': round(sc, 4), 'is_known': bool(self.adj[dr, di] == 1.0)})
            matrix.append(row)

        threshold = 0.25
        nodes, edges = [], []
        for dr in dr_idxs:
            nodes.append({'id': f'd{dr}', 'label': self.drug_names[dr], 'type': 'drug', 'idx': dr})
        for di in di_idxs:
            nodes.append({'id': f'dis{di}', 'label': self.disease_names[di], 'type': 'disease', 'idx': di})
        for dr in dr_idxs:
            for di in di_idxs:
                sc = float(score_cache[dr][di]) if dr in score_cache else 0.0
                known = bool(self.adj[dr, di] == 1.0)
                if sc >= threshold or known:
                    edges.append({'source': f'd{dr}', 'target': f'dis{di}',
                                  'score': round(sc, 4), 'is_known': known})

        return {
            'drugs':    [{'idx': i, 'name': self.drug_names[i], 'id': self.drug_ids[i]} for i in dr_idxs],
            'diseases': [{'idx': i, 'name': self.disease_names[i], 'code': self.disease_codes[i]} for i in di_idxs],
            'matrix':   matrix,
            'graph':    {'nodes': nodes, 'edges': edges},
            'model':    model,
        }

    # ── Fuzzy step-by-step explanation ───────────────────────
    def fuzzy_explain(self, drug_idx: int, disease_idx: int) -> dict:
        """Return step-by-step Mamdani FIS details for one drug-disease pair."""
        if drug_idx < 0 or drug_idx >= self.n_drugs: return {}
        if disease_idx < 0 or disease_idx >= self.n_diseases: return {}

        cf = float(self._cf_drug_disease()[drug_idx][disease_idx])

        sim_row  = self.drug_sim[drug_idx]
        weighted = sim_row[:, None] * self.adj
        known_cnt = self.adj.sum(axis=0)
        src_nbr = float(np.where(known_cnt > 0, weighted.max(axis=0), 0.0)[disease_idx])

        known_di = self.adj[drug_idx]
        tgt_nbr_arr = (self.disease_sim * known_di[None, :]).max(axis=1) \
                      if known_di.sum() > 0 else np.zeros(self.n_diseases, dtype=np.float32)
        tgt_nbr = float(tgt_nbr_arr[disease_idx])

        fuzzy_score = float(self._fuzzy.infer(
            np.array([cf]), np.array([src_nbr]), np.array([tgt_nbr]))[0])

        fz = self._fuzzy
        def _mf(fn, v): return round(float(fn(np.array([v]))[0]), 4)

        cf_lo,  cf_md,  cf_hi  = _mf(fz._mf_low, cf),      _mf(fz._mf_mid, cf),      _mf(fz._mf_high, cf)
        sn_lo,  sn_md,  sn_hi  = _mf(fz._mf_low, src_nbr), _mf(fz._mf_mid, src_nbr), _mf(fz._mf_high, src_nbr)
        tn_lo,  tn_md,  tn_hi  = _mf(fz._mf_low, tgt_nbr), _mf(fz._mf_mid, tgt_nbr), _mf(fz._mf_high, tgt_nbr)

        rules = [
            {'desc': 'IF cf_high ∧ src_high ∧ tgt_high → Very High (0.90)',
             'activation': round(min(cf_hi, sn_hi, tn_hi), 4), 'output': 0.90},
            {'desc': 'IF cf_high ∧ src_high → High (0.70)',
             'activation': round(min(cf_hi, sn_hi), 4),        'output': 0.70},
            {'desc': 'IF cf_high ∧ src_mid ∧ tgt_mid → High (0.70)',
             'activation': round(min(cf_hi, sn_md, tn_md), 4), 'output': 0.70},
            {'desc': 'IF cf_mid ∧ src_high ∧ tgt_high → High (0.70)',
             'activation': round(min(cf_md, sn_hi, tn_hi), 4), 'output': 0.70},
            {'desc': 'IF cf_high ∧ src_low → Medium (0.50)',
             'activation': round(min(cf_hi, sn_lo), 4),        'output': 0.50},
            {'desc': 'IF cf_mid ∧ src_high ∧ tgt_mid → Medium (0.50)',
             'activation': round(min(cf_md, sn_hi, tn_md), 4), 'output': 0.50},
            {'desc': 'IF cf_mid ∧ src_mid → Medium (0.50)',
             'activation': round(min(cf_md, sn_md), 4),        'output': 0.50},
            {'desc': 'IF cf_low ∧ src_high → Medium (0.50)',
             'activation': round(min(cf_lo, sn_hi), 4),        'output': 0.50},
            {'desc': 'IF cf_mid ∧ src_low → Low (0.30)',
             'activation': round(min(cf_md, sn_lo), 4),        'output': 0.30},
            {'desc': 'IF cf_low ∧ src_mid → Low (0.30)',
             'activation': round(min(cf_lo, sn_md), 4),        'output': 0.30},
            {'desc': 'IF cf_low ∧ src_low ∧ tgt_low → Very Low (0.10)',
             'activation': round(min(cf_lo, sn_lo, tn_lo), 4), 'output': 0.10},
        ]

        x = np.linspace(0, 1, 51)
        mf_chart = {
            'x':    [round(float(v), 2) for v in x],
            'low':  [round(float(fz._mf_low(np.array([v]))[0]),  4) for v in x],
            'mid':  [round(float(fz._mf_mid(np.array([v]))[0]),  4) for v in x],
            'high': [round(float(fz._mf_high(np.array([v]))[0]), 4) for v in x],
        }

        return {
            'drug_name':    self.drug_names[drug_idx],
            'disease_name': self.disease_names[disease_idx],
            'is_known':     bool(self.adj[drug_idx, disease_idx] == 1),
            'inputs':  {'cf_score': round(cf, 4), 'src_nbr_score': round(src_nbr, 4),
                        'tgt_nbr_score': round(tgt_nbr, 4)},
            'memberships': {
                'cf':      {'low': cf_lo,  'medium': cf_md,  'high': cf_hi},
                'src_nbr': {'low': sn_lo,  'medium': sn_md,  'high': sn_hi},
                'tgt_nbr': {'low': tn_lo,  'medium': tn_md,  'high': tn_hi},
            },
            'rules':       rules,
            'fuzzy_score': round(fuzzy_score, 4),
            'mf_chart':    mf_chart,
        }

    # ── Candidate molecule generation ─────────────────────────
    _FRAGMENTS = [
        ('Methyl',          'C'),  ('Hydroxyl',        'O'),
        ('Amino',           'N'),  ('Fluoro',          'F'),
        ('Chloro',          'Cl'), ('Carboxyl',        'C(=O)O'),
        ('Ketone',          'C(=O)C'), ('Trifluoromethyl', 'C(F)(F)F'),
        ('Cyano',           'C#N'), ('Sulfonamide',    'S(=O)(=O)N'),
    ]
    _SCAFFOLDS = [
        ('Benzene',         'c1ccccc1'),  ('Pyridine',       'c1ccncc1'),
        ('Imidazole',       'c1cnc[nH]1'), ('Pyrimidine',    'c1cnccn1'),
        ('Thiophene',       'c1ccsc1'),   ('Piperidine',     'C1CCNCC1'),
        ('Morpholine',      'C1CCOCC1'),  ('Indole',         'c1ccc2[nH]ccc2c1'),
        ('Quinoline',       'c1ccc2ncccc2c1'), ('Benzimidazole', 'c1ccc2[nH]cnc2c1'),
    ]

    def generate_candidates(self, disease_idx: int, n: int = 6) -> list:
        """Fragment-based candidate generation for a target disease."""
        from difflib import SequenceMatcher
        import random
        rng = random.Random(42 + disease_idx)

        if disease_idx < 0 or disease_idx >= self.n_diseases:
            return []

        known_idxs = np.where(self.adj[:, disease_idx] == 1)[0]
        cf_col = self._cf_drug_disease()[:, disease_idx]

        def score_smiles(smi: str) -> float:
            if not known_idxs.size:
                return round(rng.uniform(0.15, 0.35), 4)
            sims = [SequenceMatcher(None, smi, self.drug_smiles[ki]).ratio()
                    for ki in known_idxs[:15]
                    if ki < len(self.drug_smiles) and self.drug_smiles[ki]]
            sim  = max(sims) if sims else 0.0
            base = float(cf_col[known_idxs[:15]].mean()) if known_idxs.size else 0.3
            return round(0.35 * sim + 0.65 * base, 4)

        candidates = []

        # Strategy 1 – modify known drugs
        for ki in known_idxs[:4]:
            base_smi = self.drug_smiles[ki] if ki < len(self.drug_smiles) else ''
            if not base_smi:
                continue
            for fr_name, fr_smi in rng.sample(self._FRAGMENTS, min(4, len(self._FRAGMENTS))):
                new_smi = base_smi + fr_smi
                candidates.append({
                    'smiles': new_smi,
                    'name':   f'Cand-{self.drug_names[ki][:8]}-{fr_name[:3]}',
                    'base':   self.drug_names[ki],
                    'strategy': 'Fragment Addition',
                    'modification': fr_name,
                    'score': score_smiles(new_smi),
                })

        # Strategy 2 – scaffold + fragment assembly
        for sc_name, sc_smi in rng.sample(self._SCAFFOLDS, min(5, len(self._SCAFFOLDS))):
            for fr_name, fr_smi in rng.sample(self._FRAGMENTS, min(3, len(self._FRAGMENTS))):
                new_smi = sc_smi + fr_smi
                candidates.append({
                    'smiles': new_smi,
                    'name':   f'{sc_name[:5]}-{fr_name[:3]}',
                    'base':   sc_name,
                    'strategy': 'Scaffold Assembly',
                    'modification': fr_name,
                    'score': score_smiles(new_smi),
                })

        seen, unique = set(), []
        for c in sorted(candidates, key=lambda x: x['score'], reverse=True):
            if c['smiles'] not in seen:
                seen.add(c['smiles'])
                unique.append(c)
        return unique[:n]

    # ── Graph explorer data ───────────────────────────────────
    def get_entity_graph(self, entity_type: str, entity_idx: int,
                         top_k: int = 10, model: str = 'fuzzy') -> dict:
        """Return Cytoscape-ready graph for one entity + its top associations."""
        nodes, edges = [], []

        if entity_type == 'drug':
            if entity_idx < 0 or entity_idx >= self.n_drugs: return {}
            nodes.append({'id': f'd{entity_idx}', 'label': self.drug_names[entity_idx],
                          'type': 'drug', 'center': True})
            scores = self._scores_drug(entity_idx, model)
            order  = np.argsort(-scores)
            for di in order[:top_k]:
                sc = float(scores[di])
                nodes.append({'id': f'dis{di}', 'label': self.disease_names[di],
                              'type': 'disease', 'score': round(sc, 4)})
                edges.append({'source': f'd{entity_idx}', 'target': f'dis{di}',
                              'score': round(sc, 4),
                              'is_known': bool(self.adj[entity_idx, di] == 1)})

        elif entity_type == 'disease':
            if entity_idx < 0 or entity_idx >= self.n_diseases: return {}
            nodes.append({'id': f'dis{entity_idx}', 'label': self.disease_names[entity_idx],
                          'type': 'disease', 'center': True})
            scores = self._scores_disease(entity_idx, model)
            order  = np.argsort(-scores)
            for dr in order[:top_k]:
                sc = float(scores[dr])
                nodes.append({'id': f'd{dr}', 'label': self.drug_names[dr],
                              'type': 'drug', 'score': round(sc, 4)})
                edges.append({'source': f'd{dr}', 'target': f'dis{entity_idx}',
                              'score': round(sc, 4),
                              'is_known': bool(self.adj[dr, entity_idx] == 1)})
        else:
            return {}

        return {'nodes': nodes, 'edges': edges}

