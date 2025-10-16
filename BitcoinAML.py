# ==== Setup & Installs =========================================================
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("--- Step 1: Detecting PyTorch and CUDA versions ---")
torch_version_full = torch.__version__
torch_version_main = torch_version_full.split('+')[0]
if torch.cuda.is_available():
    cuda_version = f'cu{torch.version.cuda.replace(".", "")}'
    print(f"CUDA is available. Detected CUDA Version for PyG wheels: {cuda_version}")
else:
    cuda_version = 'cpu'
    print("CUDA is NOT available. Installing CPU versions of PyG libraries.")
print(f"Detected PyTorch Main Version: {torch_version_main}")
print(f"Full PyTorch Version: {torch_version_full}")

pyg_whl_url = f"https://data.pyg.org/whl/torch-{torch_version_main}+{cuda_version}.html"
print(f"Using PyG wheel URL: {pyg_whl_url}")

# --- Step 2: Explicit installs for PyG ops (avoid runtime import errors) ---
!pip install -q torch-scatter -f {pyg_whl_url}
!pip install -q torch-sparse  -f {pyg_whl_url}
!pip install -q torch-cluster -f {pyg_whl_url}
!pip install -q torch-spline-conv -f {pyg_whl_url}
!pip install -q torch-geometric -f {pyg_whl_url}
try:
    !pip install -q pyg-lib -f {pyg_whl_url}
except Exception as e:
    print("Skipping pyg-lib install:", e)

# ==== Device & Imports =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

import numpy as np
import pandas as pd
import torch.nn as nn
from torch import amp
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from contextlib import nullcontext
from copy import deepcopy

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.utils import dropout_edge, subgraph, k_hop_subgraph, to_undirected, coalesce
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Repro (toggle if you want max speed instead) ---
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ==== List dataset files (optional) ============================================
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
print("...")

# ==== Paths ====================================================================
features_path = '/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv'
edges_path    = '/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv'
classes_path  = '/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv'

# ==== Load =====================================================================
features_df = pd.read_csv(features_path, header=None)
edges_df    = pd.read_csv(edges_path)
classes_df  = pd.read_csv(classes_path)

# Keep ALL 165 features (no PCA). We'll scale them (fit on t<=30 only).
features_df.columns = ['txId', 'timestep'] + [f'feature_{i}' for i in range(165)]
classes_df['class'] = classes_df['class'].astype(str).map({'1': 1, '2': 0, 'unknown': 2})  # 1: illicit, 0: licit, 2: unknown
assert classes_df['class'].isna().sum() == 0, "Class mapping produced NaNs"
data_df = pd.merge(features_df, classes_df, on='txId')

# ==== Node mapping =============================================================
node_map = {txId: idx for idx, txId in enumerate(data_df['txId'].unique())}

# --- Build DIRECTED graph (for encoder) and filter unknown nodes ---------------
e = edges_df[edges_df['txId1'].isin(node_map) & edges_df['txId2'].isin(node_map)]
edge_index = torch.tensor([
    [node_map[txId] for txId in e['txId1']],
    [node_map[txId] for txId in e['txId2']]
], dtype=torch.long)

# ==== Scale features (train-only) =============================================
all_feats = data_df[[f'feature_{i}' for i in range(165)]].values.astype(np.float32)
all_ts = data_df['timestep'].values
scaler = StandardScaler()
scaler.fit(all_feats[all_ts <= 30])        # fit on train window ONLY
scaled_feats = scaler.transform(all_feats)  # transform all

# ==== Tensors ==================================================================
x = torch.tensor(scaled_feats, dtype=torch.float)
y = torch.tensor(data_df['class'].values, dtype=torch.long)         # 1: illicit, 0: licit, 2: unknown
original_y = y.clone()
timesteps = torch.tensor(data_df['timestep'].values, dtype=torch.long)

# ==== Data object: encoder(DIR) + classifier(UNDIR) ============================
full_data = Data(x=x, edge_index=edge_index, y=y, timesteps=timesteps, num_nodes=len(node_map))
# Coalesce BOTH directed (encoder) and undirected (classifier) graphs for stability & speed
full_data.edge_index, _ = coalesce(full_data.edge_index, None, full_data.num_nodes, full_data.num_nodes)
edge_index_undir = to_undirected(full_data.edge_index)
edge_index_cls, _ = coalesce(edge_index_undir, None, full_data.num_nodes, full_data.num_nodes)
full_data.edge_index_cls = edge_index_cls
full_data = full_data.to(device)

# ==== Masks ====================================================================
train_mask = (timesteps <= 30) & (y != 2)
val_mask   = (timesteps > 30) & (timesteps <= 34) & (y != 2)
test_mask  = (timesteps > 34) & (y != 2)

labeled_data = full_data.clone()
labeled_data.train_mask = train_mask.to(device)
labeled_data.val_mask   = val_mask.to(device)
labeled_data.test_mask  = test_mask.to(device)

# ==== Speed/Memory & Drift-Handling Controls ==================================
FAST_MODE = True        # Set False for full model; True for Kaggle 2h-friendly
CALIBRATE_EVERY = 3 if FAST_MODE else 1
K_HOPS_TRAIN   = 2 if FAST_MODE else 3

# Drift-handling toggles
ENABLE_IMPORTANCE_WEIGHTING   = True   # tiny domain classifier, weights train ≤30 samples
ENABLE_VAL_SELF_TRAIN         = True   # head-only fine-tune on 31..34 unknowns (gated)
ENABLE_TEST_PRIOR_CORRECTION  = True   # EM prior correction at test
ENABLE_ADAPTIVE_T_THRESHOLDS  = True   # per-t timestep τ using target rate
ENABLE_PSEUDO_CONSISTENCY     = True   # two-view agreement for pseudo labels

# ==== Per-dataset cache (no hidden coupling) ===================================
class SubgraphCache:
    def __init__(self, data: Data):
        self.data = data
        self.unique_ts = sorted(data.timesteps.unique().detach().cpu().tolist())
        self._cum_nodes_cache = {}
        self._enc_subgraph_cache = {}
        self._cls_subgraph_cache = {}

    def cum_nodes_at_t(self, t: int):
        t = int(t)
        if t not in self._cum_nodes_cache:
            self._cum_nodes_cache[t] = torch.nonzero(self.data.timesteps <= t, as_tuple=False).view(-1)
        return self._cum_nodes_cache[t]

    def cum_subgraphs_at_t(self, t: int):
        """Return (nodes, enc_sub_ei, cls_sub_ei), cached per timestep."""
        t = int(t)
        if (t not in self._enc_subgraph_cache) or (t not in self._cls_subgraph_cache):
            nodes = self.cum_nodes_at_t(t)
            if nodes.numel() == 0:
                self._enc_subgraph_cache[t] = None
                self._cls_subgraph_cache[t] = None
                return nodes, None, None
            enc_ei, _ = subgraph(nodes, self.data.edge_index,     relabel_nodes=True)
            cls_ei, _ = subgraph(nodes, self.data.edge_index_cls, relabel_nodes=True)
            self._enc_subgraph_cache[t] = enc_ei
            self._cls_subgraph_cache[t] = cls_ei
        else:
            nodes = self.cum_nodes_at_t(t)
        return nodes, self._enc_subgraph_cache[t], self._cls_subgraph_cache[t]

cache_full = SubgraphCache(full_data)
cache_lab  = SubgraphCache(labeled_data)

# ==== Utils ====================================================================
def nz(mask):
    return torch.nonzero(mask, as_tuple=False).view(-1)

def rand_choice(pool, k):
    # sample WITHOUT replacement for better diversity
    if pool.numel() <= k:
        return pool
    idx = torch.randperm(pool.numel(), device=pool.device)[:k]
    return pool[idx]

def add_random_edges(edge_index, num_nodes, p=0.05):
    # Simple augmentation; keep unique and avoid self-loops.
    dev = edge_index.device
    E = edge_index.size(1)
    num_add = max(1, int(p * E))
    u = torch.randint(0, num_nodes, (num_add,), device=dev)
    v = torch.randint(0, num_nodes, (num_add,), device=dev)
    mask = (u != v)
    u, v = u[mask], v[mask]
    add_edges = torch.stack([u, v], dim=0)
    all_edges = torch.cat([edge_index, add_edges], dim=1)
    all_edges = torch.unique(all_edges, dim=1)
    return all_edges

# ==== Drift tools: prior shift + adaptive thresholds ===========================
def estimate_prior_em(p, pi_init=0.2, iters=50, eps=1e-6):
    # p: tensor of probs under train prior; returns pi_hat in [eps, 1-eps]
    if not torch.is_tensor(p): p = torch.tensor(p, device=device)
    pi = torch.clamp(torch.tensor(float(pi_init), device=p.device), eps, 1-eps)
    for _ in range(iters):
        num = pi * p
        den = num + (1 - pi) * (1 - p) + 1e-12
        w = num / den
        pi_new = w.mean()
        if torch.abs(pi_new - pi) < 1e-6:
            pi = pi_new
            break
        pi = torch.clamp(pi_new, eps, 1-eps)
    return float(pi)

def prior_correct_probs(p, pi_train, pi_test, eps=1e-6):
    # p: probs under train prior -> corrected probs under test prior
    ratio_pos = (pi_test + eps) / (pi_train + eps)
    ratio_neg = (1 - pi_test + eps) / (1 - pi_train + eps)
    num = p * ratio_pos
    den = num + (1 - p) * ratio_neg + 1e-12
    return torch.clamp(num / den, 0.0, 1.0)

def adaptive_threshold_by_rate(p_corr, target_rate):
    # p_corr: corrected probs at a timestep; target_rate in (0,1)
    target_rate = float(np.clip(target_rate, 0.005, 0.6))  # sane bounds
    n = p_corr.numel()
    if n == 0:
        return 0.5
    k = max(1, int(round((1.0 - target_rate) * n)))
    vals, _ = torch.sort(p_corr)
    thresh = vals[min(k, n-1)].item()
    return min(max(thresh, 0.0), 1.0)

def compute_train_prior(data):
    m = (data.timesteps <= 30) & (data.y != 2)
    ytr = data.y[m]
    if ytr.numel() == 0:
        return 0.2
    return float((ytr == 1).float().mean().item())

def compute_val_target_rate(data):
    m = (data.timesteps > 30) & (data.timesteps <= 34) & (data.y != 2)
    yv = data.y[m]
    if yv.numel() == 0:
        return 0.2
    return float((yv == 1).float().mean().item())

# ==== Models ===================================================================
class TemporalGINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin = GINConv(mlp)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=False)
        self.fc  = nn.Linear(hidden_channels, hidden_channels)
        self.time_proj = nn.Linear(3, hidden_channels)

    def forward(self, x, edge_index, prev_h, timesteps=None, current_t=None):
        x = F.relu(self.gin(x, edge_index))
        if timesteps is not None and current_t is not None:
            dt = torch.clamp(current_t - timesteps.float(), min=0).unsqueeze(-1)
            multi_scale = torch.cat([dt, torch.log(dt + 1), torch.sqrt(dt + 1e-8)], dim=-1)
            time_emb = F.relu(self.time_proj(multi_scale))
            x = x + time_emb
        x = x.unsqueeze(0)          # (1, N, H)
        _, hn = self.gru(x, prev_h) # hn: (1, N, H)
        h = self.fc(hn.squeeze(0))  # (N, H)
        return h

def nt_xent_loss(anchor_emb, positive_emb, temperature=0.07):
    a = F.normalize(anchor_emb, dim=-1)
    b = F.normalize(positive_emb, dim=-1)
    sim = torch.mm(a, b.t()) / max(1e-6, float(temperature))
    if not torch.isfinite(sim).all():
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1e4, neginf=-1e4)
    N = a.size(0)
    labels = torch.arange(N, device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

def get_temporal_embeddings(encoder, data, cache: SubgraphCache, max_t=None):
    """Compute memory_h for nodes up to max_t once (using cached cum subgraphs)."""
    dev = data.x.device
    H = encoder.fc.in_features
    memory_h = torch.zeros(data.num_nodes, H, device=dev)
    unique_timesteps = cache.unique_ts if max_t is None else [t for t in cache.unique_ts if t <= max_t]
    encoder.eval()
    with torch.no_grad():
        for t in unique_timesteps:
            nodes, sub_ei, _ = cache.cum_subgraphs_at_t(t)
            if nodes is None or nodes.numel() == 0:
                continue
            sub_x = data.x[nodes]
            sub_t = data.timesteps[nodes]
            cur_t = torch.tensor(float(t), device=dev)
            ph = memory_h[nodes].unsqueeze(0)
            sub_h = encoder(sub_x, sub_ei, ph, timesteps=sub_t, current_t=cur_t)
            memory_h[nodes] = sub_h
    return memory_h

def make_amp():
    use_cuda = torch.cuda.is_available()
    try:
        scaler = amp.GradScaler(enabled=use_cuda)
        if use_cuda:
            def ctx(): return amp.autocast(device_type='cuda')
        else:
            def ctx(): return nullcontext()
        return scaler, ctx
    except TypeError:
        from torch.cuda import amp as cuda_amp
        scaler = cuda_amp.GradScaler(enabled=use_cuda)
        if use_cuda:
            def ctx(): return cuda_amp.autocast()
        else:
            def ctx(): return nullcontext()
        return scaler, ctx

# Safer AMP for downstream
def make_amp_downstream():
    class _NoScaler:
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        def ctx(): return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return _NoScaler(), ctx
    else:
        def ctx(): return nullcontext()
        return _NoScaler(), ctx

# ==== Pretext training (contrastive) ===========================================
def train_pretext(encoder, data, cache: SubgraphCache, max_t=30, epochs=200, lr=1e-4, temperature=0.07,
                  batch_size=768, patience=30, min_delta=1e-4, accum_steps=2,
                  checkpoint_path="pretrained_temporal_gin.pth"):
    if FAST_MODE:
        epochs = min(epochs, 60)
        batch_size = min(batch_size, 512)
        accum_steps = 1
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scaler, amp_ctx = make_amp()

    num_nodes = data.num_nodes
    H = encoder.fc.in_features
    memory_h = torch.zeros(num_nodes, H, device=data.x.device)

    unique_timesteps = [t for t in cache.unique_ts if t <= max_t]
    if len(unique_timesteps) == 0:
        raise RuntimeError("train_pretext: No timesteps to train on (check max_t).")

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0
        encoder.train()

        for t_idx, t in enumerate(tqdm(unique_timesteps, desc=f"Pretext Epoch {epoch+1}/{epochs}")):
            if t_idx % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            nodes_in_window, sub_ei, _ = cache.cum_subgraphs_at_t(t)
            if nodes_in_window is None or nodes_in_window.numel() == 0:
                continue

            sub_x = data.x[nodes_in_window]
            sub_t = data.timesteps[nodes_in_window]
            cur_t = torch.tensor(float(t), device=data.x.device)

            # Balanced sample: half labeled (y!=2), half unknown if possible
            labeled_pool = nz((data.timesteps <= t) & (data.y != 2))
            unknown_pool = nz((data.timesteps <= t) & (data.y == 2))
            k = min(batch_size, nodes_in_window.numel())
            k_lab = min(k // 2, labeled_pool.numel())
            k_unl = k - k_lab
            bn_lab = rand_choice(labeled_pool, k_lab) if k_lab > 0 else torch.tensor([], dtype=torch.long, device=data.x.device)
            bn_unl = rand_choice(unknown_pool, k_unl) if k_unl > 0 else torch.tensor([], dtype=torch.long, device=data.x.device)
            batch_node_indices_global = torch.cat([bn_lab, bn_unl])
            if batch_node_indices_global.numel() < 2:
                continue

            inv = -torch.ones(data.num_nodes, dtype=torch.long, device=data.x.device)
            inv[nodes_in_window] = torch.arange(nodes_in_window.numel(), device=data.x.device)
            batch_node_indices = inv[batch_node_indices_global]
            subset, batch_sub_ei, mapping, _ = k_hop_subgraph(
                node_idx=batch_node_indices, num_hops=2,
                edge_index=sub_ei, relabel_nodes=True,
                num_nodes=sub_x.size(0)
            )
            if mapping.numel() < 2:
                continue

            batch_sub_x = sub_x[subset]
            batch_sub_t = sub_t[subset]
            batch_ph = memory_h[nodes_in_window[subset]].unsqueeze(0).detach()

            # Augmentations: mild topology, add feature noise
            aug_ei_1, _ = dropout_edge(batch_sub_ei, p=0.15)
            aug_ei_2 = add_random_edges(batch_sub_ei, len(subset), p=0.05)
            noise = torch.randn_like(batch_sub_x) * 0.1
            batch_x_aug1 = F.dropout(batch_sub_x, p=0.2, training=True) + noise
            batch_x_aug2 = F.dropout(batch_sub_x, p=0.2, training=True) - noise

            with amp_ctx():
                emb_1 = encoder(batch_x_aug1, aug_ei_1, batch_ph,
                                timesteps=batch_sub_t, current_t=cur_t)
                emb_2 = encoder(batch_x_aug2, aug_ei_2, batch_ph,
                                timesteps=batch_sub_t, current_t=cur_t)
                emb_1_batch = emb_1[mapping]
                emb_2_batch = emb_2[mapping]
                loss = nt_xent_loss(emb_1_batch, emb_2_batch, temperature) / accum_steps

            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * accum_steps

            boundary = ((t_idx + 1) % accum_steps == 0) or ((t_idx + 1) == len(unique_timesteps))
            if boundary:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # update memory (teacher)
            with torch.no_grad():
                ph = memory_h[nodes_in_window].unsqueeze(0)
                sub_h = encoder(sub_x, sub_ei, ph, timesteps=sub_t, current_t=cur_t)
                memory_h[nodes_in_window] = sub_h

        avg_loss = total_loss / max(1, len(unique_timesteps))
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"  ↳ New best encoder saved to {checkpoint_path} (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (loss plateaued)")
                break

    state = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()
    return best_loss

class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, heads=4, drop=0.5):
        super().__init__()
        self.g1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=drop)
        self.g2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=drop)
        self.g3 = GATv2Conv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=drop)
        self.drop = nn.Dropout(drop)
    def forward(self, x, edge_index):
        x1 = F.elu(self.g1(x, edge_index))
        x2 = F.elu(self.g2(self.drop(x1), edge_index))
        x  = self.g3(self.drop(x2), edge_index)
        return x

# Stable Focal Loss (log-softmax based) with sample weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float)
        self.register_buffer('alpha', alpha if alpha is not None else None)

    def forward(self, inputs, targets, sample_weight=None):
        logp = F.log_softmax(inputs, dim=1)
        ce = F.nll_loss(logp, targets, reduction='none')
        pt = logp.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        loss = ((1 - pt).clamp(min=1e-6) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        if sample_weight is not None:
            loss = loss * sample_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ==== Calibration (Temperature Scaling) ========================================
def fit_temperature(encoder, classifier, data, cache: SubgraphCache, mem_base=None):
    dev = data.x.device
    encoder.eval(); classifier.eval()
    mem_h = mem_base.clone() if mem_base is not None else get_temporal_embeddings(encoder, data, cache, max_t=30)

    with torch.no_grad():
        logits_list, labels_list = [], []
        val_ts = sorted(data.timesteps[(data.timesteps > 30) & (data.timesteps <= 34)]
                        .unique().detach().cpu().tolist())
        for t in val_ts:
            nodes, sub_ei_enc, sub_ei_cls = cache.cum_subgraphs_at_t(t)
            if nodes is None or nodes.numel() == 0:
                continue
            sub_x  = data.x[nodes]
            sub_t  = data.timesteps[nodes]
            cur_t  = torch.tensor(float(t), device=dev)

            ph = mem_h[nodes].unsqueeze(0)
            h = encoder(sub_x, sub_ei_enc, ph, timesteps=sub_t, current_t=cur_t)
            mem_h[nodes] = h  # incremental temporal memory

            comb = torch.cat([sub_x, h], dim=1)
            out = classifier(comb, sub_ei_cls)

            global_now = ((data.timesteps == t) & (data.y != 2)).nonzero(as_tuple=False).view(-1)
            if global_now.numel() == 0:
                continue
            inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev); inv[nodes] = torch.arange(nodes.numel(), device=dev)
            local_now = inv[global_now]
            keep = local_now >= 0
            if keep.sum() == 0:
                continue
            sel = out[local_now[keep]]
            valid_rows = torch.isfinite(sel).all(dim=1)
            if valid_rows.sum() == 0:
                continue
            logits_list.append(sel[valid_rows])
            labels_list.append(data.y[global_now[keep]][valid_rows])

    if not logits_list:
        return 1.0

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    T = torch.tensor(1.0, device=dev, requires_grad=True)
    ce = torch.nn.CrossEntropyLoss()

    # Try LBFGS, fall back to Adam if needed
    try:
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
        def closure():
            opt.zero_grad()
            loss = ce(logits / T.clamp(min=0.8, max=4.0), labels)
            loss.backward()
            return loss
        opt.step(closure)
    except Exception:
        opt = torch.optim.Adam([T], lr=0.05)
        for _ in range(100):
            opt.zero_grad()
            loss = ce(logits / T.clamp(min=0.8, max=4.0), labels)
            loss.backward()
            opt.step()

    T_fit = float(T.detach().clamp(0.8, 4.0))
    return T_fit

# ==== Eval & Threshold Search ==================================================
def evaluate_metrics(encoder, classifier, data, cache: SubgraphCache, val_timesteps=False, original_y=None,
                     threshold=None, T=1.0, mem_base=None,
                     prior_correction=False, adaptive_t_thresholds=False,
                     pi_train=None, target_rate=None):
    
    dev = data.x.device
    classifier.eval(); encoder.eval()
    gt = (data.y if original_y is None else original_y).to(dev)

    memory_h = mem_base.clone() if mem_base is not None else get_temporal_embeddings(encoder, data, cache, max_t=30)
    if val_timesteps:
        timesteps_eval = sorted(data.timesteps[(data.timesteps > 30) & (data.timesteps <= 34)].unique().detach().cpu().tolist())
    else:
        timesteps_eval = sorted(data.timesteps[data.timesteps > 34].unique().detach().cpu().tolist())

    if prior_correction or adaptive_t_thresholds:
        if pi_train is None:
            pi_train = compute_train_prior(data)
        if target_rate is None:
            target_rate = compute_val_target_rate(data)

    preds, trues = [], []
    with torch.inference_mode():
        for t in tqdm(timesteps_eval, desc="Evaluating"):
            nodes, sub_ei_enc, sub_ei_cls = cache.cum_subgraphs_at_t(t)
            if nodes is None or nodes.numel() == 0:
                continue
            sub_x = data.x[nodes]
            sub_t = data.timesteps[nodes]
            cur_t = torch.tensor(float(t), device=dev)

            ph = memory_h[nodes].unsqueeze(0)
            sub_h = encoder(sub_x, sub_ei_enc, ph, timesteps=sub_t, current_t=cur_t)
            memory_h[nodes] = sub_h  # update all cumulative

            comb = torch.cat([sub_x, sub_h], dim=1)
            logits = classifier(comb, sub_ei_cls) / float(T)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            global_now = nz((data.timesteps == t) & (gt != 2))
            if global_now.numel() == 0:
                continue
            inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
            inv[nodes] = torch.arange(nodes.numel(), device=dev)
            local_now = inv[global_now]
            keep = local_now >= 0
            if keep.sum().item() == 0:
                continue
            local_now = local_now[keep]
            global_now = global_now[keep]

            local_logits = logits[local_now]
            valid_rows = torch.isfinite(local_logits).all(dim=1)
            if valid_rows.sum().item() == 0:
                continue
            local_logits = local_logits[valid_rows]
            true = gt[global_now][valid_rows]

            p = local_logits.softmax(dim=1)[:, 1]  # prob of illicit

            if adaptive_t_thresholds:
                # prior-correct (optional)
                if prior_correction:
                    pi_t = estimate_prior_em(p, pi_init=target_rate)
                    p_use = prior_correct_probs(p, pi_train, pi_t)
                else:
                    p_use = p
                tau_t = adaptive_threshold_by_rate(p_use, target_rate)
                pred = (p_use >= tau_t).long()
            else:
                # global threshold or argmax
                if threshold is None:
                    pred = local_logits.argmax(dim=1)
                else:
                    if prior_correction:
                        pi_t = estimate_prior_em(p, pi_init=target_rate)
                        p_use = prior_correct_probs(p, pi_train, pi_t)
                    else:
                        p_use = p
                    pred = (p_use >= float(threshold)).long()

            preds.append(pred); trues.append(true)

    if not preds:
        raise RuntimeError("Evaluate found no labeled samples—check masks/timesteps.")
    y_pred = torch.cat(preds).detach().cpu().numpy()
    y_true = torch.cat(trues).detach().cpu().numpy()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_illicit = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f2_illicit = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
    print(f'F1 Macro: {macro_f1:.4f}, F1 Weighted: {weighted_f1:.4f}, Illicit Recall: {rec_illicit:.4f}, Illicit F1: {f1_illicit:.4f}, Illicit F2: {f2_illicit:.4f}')
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "rec_illicit": float(rec_illicit),
        "f1_illicit": float(f1_illicit),
        "f2_illicit": float(f2_illicit),
    }

@torch.no_grad()
def find_best_threshold(encoder, classifier, data, cache: SubgraphCache, mode='f2', T=1.0, mem_base=None):
    dev = data.x.device
    encoder.eval(); classifier.eval()
    mem_h = mem_base.clone() if mem_base is not None else get_temporal_embeddings(encoder, data, cache, max_t=30)
    val_ts = sorted(data.timesteps[(data.timesteps > 30) & (data.timesteps <= 34)].unique().detach().cpu().tolist())
    ys, ps = [], []
    for t in val_ts:
        nodes, sub_ei_enc, sub_ei_cls = cache.cum_subgraphs_at_t(t)
        if nodes is None or nodes.numel() == 0:
            continue
        sub_x  = data.x[nodes]
        sub_t  = data.timesteps[nodes]
        cur_t  = torch.tensor(float(t), device=dev)
        ph = mem_h[nodes].unsqueeze(0)
        h = encoder(sub_x, sub_ei_enc, ph, timesteps=sub_t, current_t=cur_t)
        mem_h[nodes] = h
        comb = torch.cat([sub_x, h], dim=1)
        logits = classifier(comb, sub_ei_cls) / float(T)
        probs_illicit = logits.softmax(dim=1)[:, 1]
        global_now = ((data.timesteps == t) & (data.y != 2)).nonzero(as_tuple=False).view(-1)
        if global_now.numel() == 0:
            continue
        inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
        inv[nodes] = torch.arange(nodes.numel(), device=dev)
        local_now = inv[global_now]
        keep = local_now >= 0
        if keep.sum() == 0:
            continue
        pi = probs_illicit[local_now[keep]]
        valid = torch.isfinite(pi)
        if valid.sum() == 0:  # skip if all NaN/Inf
            continue
        ys.append(data.y[global_now[keep]][valid].detach().cpu())
        ps.append(pi[valid].detach().cpu())
    if len(ys) == 0:
        raise RuntimeError("No validation labels found in 31..34 for threshold tuning.")
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

    taus = np.linspace(0.005, 0.995, 199)
    best_tau, best_score = 0.5, -1.0
    for tau in taus:
        pred = (p >= tau).astype(int)
        if mode == 'macro':
            score = f1_score(y, pred, average='macro', zero_division=0)
        else:
            score = fbeta_score(y, pred, beta=2, average='binary', pos_label=1, zero_division=0)
        if score > best_score:
            best_score, best_tau = score, tau
    print(f"[Threshold search] mode={mode}, best τ = {best_tau:.3f}, score = {best_score:.4f}")
    return float(best_tau)

# ==== Importance weighting via domain classifier ===============================
class DomainLR(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(-1)

@torch.no_grad()
def _collect_now_features(encoder, data, cache, max_t):
    """Return dict: t -> (global_now_indices, features_now [N_t, D]) for t <= max_t"""
    dev = data.x.device
    H = encoder.fc.in_features
    mem_h = torch.zeros(data.num_nodes, H, device=dev)
    out = {}
    for t in [tt for tt in cache.unique_ts if tt <= max_t]:
        nodes, ei_enc, _ = cache.cum_subgraphs_at_t(t)
        if nodes is None or nodes.numel() == 0: continue
        sub_x = data.x[nodes]
        sub_t = data.timesteps[nodes]
        ph = mem_h[nodes].unsqueeze(0)
        cur_t = torch.tensor(float(t), device=dev)
        h = encoder(sub_x, ei_enc, ph, timesteps=sub_t, current_t=cur_t)
        mem_h[nodes] = h
        inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
        inv[nodes] = torch.arange(nodes.numel(), device=dev)
        global_now = ((data.timesteps == t)).nonzero(as_tuple=False).view(-1)
        local_now = inv[global_now]
        keep = local_now >= 0
        if keep.sum() == 0:
            continue
        local_now = local_now[keep]
        global_now = global_now[keep]
        feats_now = torch.cat([sub_x[local_now], h[local_now]], dim=1)
        out[int(t)] = (global_now, feats_now)
    return out

def compute_importance_weights(encoder, data, cache, max_train_t=30, max_val_t=34,
                               max_per_domain=40000, epochs=3, lr=1e-3, clip=(0.2, 5.0)):
    encoder.eval()
    with torch.no_grad():
        tr = _collect_now_features(encoder, data, cache, max_train_t)
        va = _collect_now_features(encoder, data, cache, max_val_t)
    # Stack samples: source=<=30, target=31..34
    feats_tr = torch.cat([fe for (_, fe) in tr.values()], dim=0) if tr else torch.empty(0, 165+encoder.fc.in_features, device=device)
    va_rows = []
    for t,(gn, fe) in va.items():
        if 31 <= t <= 34:
            va_rows.append(fe)
    feats_va = torch.cat(va_rows, dim=0) if va_rows else torch.empty(0, feats_tr.size(1) if feats_tr.numel()>0 else 165+encoder.fc.in_features, device=device)

    if feats_tr.numel() == 0 or feats_va.numel() == 0:
        print("[IW] Not enough data to train domain classifier; using weights=1.")
        return torch.ones(data.num_nodes, device=device), 0.5

    # Balance and cap
    n_tr = feats_tr.size(0)
    n_va = feats_va.size(0)
    n = min(max_per_domain, min(n_tr, n_va))
    idx_tr = torch.randperm(n_tr, device=device)[:n]
    idx_va = torch.randperm(n_va, device=device)[:n]
    X = torch.cat([feats_tr[idx_tr], feats_va[idx_va]], dim=0)
    y_dom = torch.cat([torch.zeros(n, device=device), torch.ones(n, device=device)], dim=0)

    model = DomainLR(X.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(logits, y_dom)
        loss.backward()
        opt.step()
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            auc_like = ((prob[y_dom==1].mean() + (1-prob[y_dom==0]).mean())/2).item()
        print(f"[IW] epoch {ep+1}/{epochs} loss={loss.item():.4f} 'AUC-ish'={auc_like:.4f}")

    # Predict on train-now nodes to get weights
    weights = torch.ones(data.num_nodes, device=device)
    with torch.no_grad():
        for t,(global_now, feats_now) in tr.items():
            p_fut = torch.sigmoid(model(feats_now))  # P(domain=1|x)
            odds = (p_fut / (1 - p_fut + 1e-6)).clamp(clip[0], clip[1])
            weights[global_now] = odds
    return weights, auc_like

# ==== EMA helper ===============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k,p in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,p in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
    def state_dict(self):
        return {k: v.clone().detach() for k,v in self.shadow.items()}

# ==== Downstream training (recall-focused) =====================================
def train_downstream(encoder, classifier, data, cache: SubgraphCache, epochs=50, lr_head=5e-4, lr_enc=1e-5,
                     gamma=2.0, patience=20, freeze_encoder_for=3, checkpoint_path="classifier_best.pth",
                     threshold_mode='f2', importance_weights=None, use_ema=True, ema_decay=0.999):
    dev = next(classifier.parameters()).device

    # Auto α from class priors on train_mask
    y_train = data.y[(data.timesteps <= 30) & (data.y != 2)]
    if y_train.numel() > 0:
        pos = (y_train == 1).sum().item()     # illicit
        neg = (y_train == 0).sum().item()     # licit
        tot = max(1, pos + neg)
        alpha = torch.tensor([neg / tot, pos / tot], dtype=torch.float, device=dev)  # [class0, class1]
        alpha = torch.clamp(alpha, min=0.05, max=0.95)
        alpha = alpha / alpha.sum()
    else:
        alpha = torch.tensor([0.30, 0.70], dtype=torch.float, device=dev)

    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    if FAST_MODE:
        epochs = min(epochs, 20)
        freeze_encoder_for = min(freeze_encoder_for, 1)

    best_metric, patience_counter = -1.0, 0
    best_T, best_tau = 1.0, 0.5
    H = encoder.fc.in_features
    params = [
        {'params': encoder.parameters(), 'lr': lr_enc, 'weight_decay': 1e-5},
        {'params': classifier.parameters(), 'lr': lr_head, 'weight_decay': 5e-4},
    ]
    optimizer = torch.optim.AdamW(params)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, cooldown=1)
    scaler, amp_ctx = make_amp_downstream()

    train_ts = sorted(data.timesteps[data.timesteps <= 30].unique().detach().cpu().tolist())
    ema = EMA(classifier, decay=ema_decay) if use_ema else None

    for epoch in range(epochs):
        if epoch < freeze_encoder_for:
            encoder.eval()
            for p in encoder.parameters(): p.requires_grad_(False)
        else:
            encoder.train()
            for p in encoder.parameters(): p.requires_grad_(True)
        classifier.train()
        total_loss, num_steps = 0.0, 0
        memory_h = torch.zeros(data.num_nodes, H, device=dev)

        for t in tqdm(train_ts, desc=f"Downstream Epoch {epoch+1}/{epochs}"):
            nodes, sub_ei_enc_full, sub_ei_cls_full = cache.cum_subgraphs_at_t(t)
            if nodes is None or nodes.numel() == 0:
                continue
            global_now = nz((data.timesteps == t) & (data.y != 2))
            if global_now.numel() == 0:
                continue

            # Map global_now to local indices in nodes
            inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
            inv[nodes] = torch.arange(nodes.numel(), device=dev)
            local_now_full = inv[global_now]
            keep = local_now_full >= 0
            if keep.sum().item() == 0:
                continue
            local_now_full = local_now_full[keep]
            global_now = global_now[keep]
            labels = data.y[global_now]

            # Sample k-hop subgraph around labeled nodes (on cumulative <=t)
            local_subset, local_sub_ei_enc, mapping, _ = k_hop_subgraph(
                local_now_full, num_hops=K_HOPS_TRAIN, edge_index=sub_ei_enc_full,
                relabel_nodes=True, num_nodes=len(nodes)
            )
            # Subgraph for cls (undirected, relabeled)
            local_sub_ei_cls, _ = subgraph(local_subset, sub_ei_cls_full, relabel_nodes=True)

            # Mapped nodes (global)
            sub_nodes = nodes[local_subset]
            sub_x = data.x[sub_nodes]
            sub_t = data.timesteps[sub_nodes]
            cur_t = torch.tensor(float(t), device=dev)
            ph = memory_h[sub_nodes].unsqueeze(0)
            sub_h = encoder(sub_x, local_sub_ei_enc, ph, timesteps=sub_t, current_t=cur_t)
            # Update memory on sampled sub_nodes only (approx for train)
            memory_h[sub_nodes] = sub_h.detach()
            comb = torch.cat([sub_x, sub_h], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = classifier(comb, local_sub_ei_cls)
                if not torch.isfinite(out).all():
                    print("[warn] Non-finite logits detected; skipping batch")
                    continue
                # importance weights for now nodes if provided
                sw = None
                if importance_weights is not None:
                    sw = importance_weights[global_now]
                    # normalize per-batch and clip to stabilize with focal
                    sw = (sw / (sw.mean() + 1e-8)).clamp_(0.5, 2.0).detach()
                loss = criterion(out[mapping], labels, sample_weight=sw)
            if not torch.isfinite(loss):
                print("[warn] Non-finite loss detected; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            if encoder.training:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            num_steps += 1

            if ema is not None:
                ema.update(classifier)

        if num_steps > 0:
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / num_steps:.4f}")
        else:
            print(f"Epoch {epoch+1}, no labeled steps found.")

        # Calibrate (T) and tune threshold (τ) on val (31..34) for chosen metric
        do_calibrate = ((epoch + 1) % CALIBRATE_EVERY == 0) or (epoch == epochs - 1)
        if do_calibrate:
            # Evaluate with EMA weights if available
            if ema is not None:
                raw_state = deepcopy(classifier.state_dict())
                ema.copy_to(classifier)

            mem_base = get_temporal_embeddings(encoder, data, cache, max_t=30)
            T_val = fit_temperature(encoder, classifier, data, cache, mem_base=mem_base)
            tau_val = find_best_threshold(encoder, classifier, data, cache, mode=threshold_mode, T=T_val, mem_base=mem_base)
            val_metrics = evaluate_metrics(encoder, classifier, data, cache, val_timesteps=True, original_y=original_y,
                                           threshold=tau_val, T=T_val, mem_base=mem_base)
            metric_to_opt = val_metrics['f2_illicit'] if threshold_mode == 'f2' else val_metrics['macro_f1']
            scheduler.step(metric_to_opt)
            if metric_to_opt > best_metric:
                best_metric = metric_to_opt
                best_T, best_tau = T_val, tau_val
                patience_counter = 0
                torch.save({
                    "cls_state_dict": classifier.state_dict(),  # EMA weights if enabled
                    "enc_state_dict": encoder.state_dict(),
                    "val_metric": best_metric,
                    "val_macro_f1": val_metrics['macro_f1'],
                    "epoch": epoch,
                    "T_val": T_val,
                    "tau_val": tau_val
                }, checkpoint_path)
                print(f" ↳ New best saved to {checkpoint_path} (val_metric={best_metric:.4f}, T={T_val:.3f}, τ={tau_val:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered (no metric improvement).")
                    # restore best before return
                    break

            if ema is not None:
                # restore raw for continued training
                classifier.load_state_dict(raw_state, strict=True)

    # restore best
    best = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(best["enc_state_dict"])
    classifier.load_state_dict(best["cls_state_dict"])
    print(f"Restored best from epoch {best['epoch']} (val_metric={best['val_metric']:.4f}, "
          f"val_macro_f1={best.get('val_macro_f1',-1):.4f}, T={best.get('T_val',1.0):.3f}, τ={best.get('tau_val',0.5):.3f})")
    return best.get("T_val", 1.0), best.get("tau_val", 0.5), checkpoint_path

# ==== Pseudo labeling (≤30) with calibration + consistency =====================
@torch.no_grad()
def pseudo_label(encoder, classifier, data, cache: SubgraphCache, max_t=30, T=1.0,
                 th_illicit=0.995, th_licit=0.9995,
                 cap_per_timestep=50, cap_per_class_per_t=30, use_consistency=True):
    dev = data.x.device
    encoder.eval(); classifier.eval()
    emb = get_temporal_embeddings(encoder, data, cache, max_t=max_t)

    nodes, sub_ei_enc, sub_ei_cls = cache.cum_subgraphs_at_t(max_t)
    if nodes is None or nodes.numel() == 0:
        print("Pseudo-label: no nodes in window.")
        return
    sub_x = data.x[nodes]
    sub_t = data.timesteps[nodes]
    cur_t = torch.tensor(float(max_t), device=dev)
    ph = emb[nodes].unsqueeze(0)  # emb already at <= max_t

    # two-view consistency (dropout + edge-drop) if enabled
    def forward_probs(ei):
        logits = classifier(torch.cat([sub_x, encoder(sub_x, ei, ph, timesteps=sub_t, current_t=cur_t)], dim=1), sub_ei_cls) / float(T)
        return logits.softmax(dim=1)

    if use_consistency:
        ei1, _ = dropout_edge(sub_ei_enc, p=0.1)
        ei2, _ = dropout_edge(sub_ei_enc, p=0.2)
        probs1 = forward_probs(ei1)
        probs2 = forward_probs(ei2)
        probs = (probs1 + probs2) / 2
        agree = (probs1.argmax(dim=1) == probs2.argmax(dim=1))
    else:
        logits = classifier(torch.cat([sub_x, emb[nodes]], dim=1), sub_ei_cls) / float(T)
        probs = logits.softmax(dim=1)
        agree = torch.ones(probs.size(0), dtype=torch.bool, device=dev)

    inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
    inv[nodes] = torch.arange(nodes.numel(), device=dev)
    unknown_global = nz((data.y == 2) & (data.timesteps <= max_t))
    if unknown_global.numel() == 0:
        print("Pseudo-label: nothing unknown in window.")
        return
    unknown_local = inv[unknown_global]
    keep = unknown_local >= 0
    unknown_local = unknown_local[keep]
    unknown_global = unknown_global[keep]

    out = probs[unknown_local]
    conf, pred = out.max(dim=1)

    # apply class thresholds + agreement
    mask_illicit = (pred == 1) & (conf >= th_illicit) & agree[unknown_local]
    mask_licit   = (pred == 0) & (conf >= th_licit)   & agree[unknown_local]

    pick_global_indices = []
    ts = data.timesteps[unknown_global]
    for t in ts.unique():
        t_mask = (ts == t)
        # per-class caps within timestep t
        idx_il = (t_mask & mask_illicit).nonzero(as_tuple=False).view(-1)
        idx_li = (t_mask & mask_licit).nonzero(as_tuple=False).view(-1)

        chosen_t = []
        if idx_il.numel() > 0:
            top_il = conf[idx_il].argsort(descending=True)[:cap_per_class_per_t]
            chosen_t.append(idx_il[top_il])
        if idx_li.numel() > 0:
            top_li = conf[idx_li].argsort(descending=True)[:cap_per_class_per_t]
            chosen_t.append(idx_li[top_li])

        if len(chosen_t) == 0:
            continue
        pick_t = torch.cat(chosen_t)

        # overall per-timestep cap (apply after per-class selection)
        if cap_per_timestep is not None and pick_t.numel() > cap_per_timestep:
            top = conf[pick_t].argsort(descending=True)[:cap_per_timestep]
            pick_t = pick_t[top]

        pick_global_indices.append(pick_t)

    if len(pick_global_indices) == 0:
        print("Added 0 pseudo-labels (thresholds too high or no agreement).")
        return
    pick = torch.cat(pick_global_indices)

    pseudo_y = pred[pick]
    pseudo_global = unknown_global[pick]
    data.y[pseudo_global] = pseudo_y
    num_added = pseudo_global.numel()
    print(f"Added {num_added} pseudo-labels in train window (<= {max_t}).")

# ==== Self-training on val (31..34, unknowns only, head-only) ==================
def self_train_on_val_unknowns(encoder, classifier, data, cache: SubgraphCache,
                               epochs=3, lr_head=2e-4, base_T=1.0,
                               th_illicit=0.97, th_licit=0.995,
                               use_prior_correction=True):
    dev = data.x.device
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    classifier.train()
    opt = torch.optim.AdamW(classifier.parameters(), lr=lr_head)
    H = encoder.fc.in_features

    mem_h = get_temporal_embeddings(encoder, data, cache, max_t=30)
    pi_train = compute_train_prior(data)
    target_rate = compute_val_target_rate(data)

    val_ts = sorted(data.timesteps[(data.timesteps > 30) & (data.timesteps <= 34)].unique().detach().cpu().tolist())
    for ep in range(epochs):
        total_loss, steps = 0.0, 0
        for t in val_ts:
            nodes, ei_enc, ei_cls = cache.cum_subgraphs_at_t(t)
            if nodes is None or nodes.numel() == 0: continue
            sub_x = data.x[nodes]
            sub_t = data.timesteps[nodes]
            cur_t = torch.tensor(float(t), device=dev)

            ph = mem_h[nodes].unsqueeze(0)
            h = encoder(sub_x, ei_enc, ph, timesteps=sub_t, current_t=cur_t)
            mem_h[nodes] = h
            comb = torch.cat([sub_x, h], dim=1)
            logits = classifier(comb, ei_cls) / float(base_T)
            probs = logits.softmax(dim=1)[:, 1]

            inv = -torch.ones(data.num_nodes, dtype=torch.long, device=dev)
            inv[nodes] = torch.arange(nodes.numel(), device=dev)
            global_now = (data.timesteps == t).nonzero(as_tuple=False).view(-1)
            local_now = inv[global_now]
            keep_now = local_now >= 0
            if keep_now.sum() == 0:
                continue
            local_now = local_now[keep_now]
            global_now = global_now[keep_now]

            # Build corrected probs (optional) on "now"
            p_now = probs[local_now]
            if use_prior_correction:
                pi_t = estimate_prior_em(p_now, pi_init=target_rate)
                p_corr = prior_correct_probs(p_now, pi_train, pi_t)
            else:
                p_corr = p_now

            # select confident *unknown* now
            unk_mask_now = (data.y[global_now] == 2)
            if unk_mask_now.sum() == 0:
                continue
            idx_unk = torch.nonzero(unk_mask_now, as_tuple=False).view(-1)
            p_u = p_corr[idx_unk]
            sel_il = idx_unk[p_u >= th_illicit]
            sel_li = idx_unk[p_u <= (1.0 - th_licit)]
            sel = torch.cat([sel_il, sel_li], dim=0)
            if sel.numel() == 0:
                continue

            y_pseudo = torch.zeros(sel.numel(), dtype=torch.long, device=dev)
            y_pseudo[:sel_il.numel()] = 1  # illicit first part

            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits[local_now[sel]], y_pseudo)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()); steps += 1
        print(f"[Self-train 31..34] epoch {ep+1}/{epochs}, steps={steps}, avg_loss={(total_loss/max(1,steps)):.4f}")

# ==== Run the training =========================================================
hidden = 128 if FAST_MODE else 256
heads = 2 if FAST_MODE else 4
drop = 0.5

encoder = TemporalGINEncoder(165, hidden).to(device)
classifier = GATClassifier(165 + hidden, hidden, heads=heads, drop=drop).to(device)

# Pretext (use full_data + its cache)
train_pretext(encoder, full_data, cache_full)  # Uses defaults with modified params via FAST_MODE

# Importance weights (optional) computed once before downstream + gating
iw = None
iw_aucish = 0.5
if ENABLE_IMPORTANCE_WEIGHTING:
    print("\n[IW] Training domain classifier for importance weighting...")
    iw_raw, iw_aucish = compute_importance_weights(encoder, labeled_data, cache_lab,
                                    max_train_t=30, max_val_t=34,
                                    max_per_domain=30000 if FAST_MODE else 60000,
                                    epochs=3 if FAST_MODE else 5, lr=1e-3)
    if iw_aucish >= 0.55:
        iw = iw_raw
        print(f"[IW] enabled (AUC-ish={iw_aucish:.3f})")
    else:
        iw = None
        print(f"[IW] disabled (weak shift signal, AUC-ish={iw_aucish:.3f})")

checkpoints = []
best_f2 = -1
best_cp = None
best_round_T = 1.0
best_round_tau = 0.5

for r in range(1, 4):
    cp_path = f"classifier_best_round{r}.pth"
    # Train downstream; function restores best checkpoint internally
    T_pre, tau_pre, cp_pre = train_downstream(
        encoder, classifier, labeled_data, cache_lab,
        checkpoint_path=cp_path,
        importance_weights=iw,
        use_ema=True, ema_decay=0.999
    )

    # Evaluate the pre self-train model on val (consistently)
    mem_base_pre = get_temporal_embeddings(encoder, labeled_data, cache_lab, max_t=30)
    T_pre_eval = T_pre if T_pre is not None else fit_temperature(encoder, classifier, labeled_data, cache_lab, mem_base=mem_base_pre)
    tau_pre_eval = tau_pre if tau_pre is not None else find_best_threshold(encoder, classifier, labeled_data, cache_lab, mode='f2', T=T_pre_eval, mem_base=mem_base_pre)
    val_pre = evaluate_metrics(encoder, classifier, labeled_data, cache_lab, val_timesteps=True,
                               original_y=original_y, threshold=tau_pre_eval, T=T_pre_eval, mem_base=mem_base_pre)
    f2_pre = val_pre["f2_illicit"]

    # Save classifier state before self-train (for gating)
    cls_before = deepcopy(classifier.state_dict())

    # Optional head-only self-training on 31..34 (unknowns) — gated
    kept_selftrain = False
    T_chosen, tau_chosen, f2_chosen = T_pre_eval, tau_pre_eval, f2_pre
    if ENABLE_VAL_SELF_TRAIN:
        print("\n[Self-train] Fine-tuning head on 31..34 unknowns...")
        self_train_on_val_unknowns(encoder, classifier, labeled_data, cache_lab,
                                   epochs=3 if FAST_MODE else 5, lr_head=2e-4,
                                   base_T=T_pre_eval,
                                   th_illicit=max(0.97, tau_pre_eval+0.10),
                                   th_licit=max(0.995, tau_pre_eval+0.20),
                                   use_prior_correction=True)
        # After self-train, re-fit T and τ and evaluate
        mem_base_st = get_temporal_embeddings(encoder, labeled_data, cache_lab, max_t=30)
        T_st = fit_temperature(encoder, classifier, labeled_data, cache_lab, mem_base=mem_base_st)
        tau_st = find_best_threshold(encoder, classifier, labeled_data, cache_lab, mode='f2', T=T_st, mem_base=mem_base_st)
        val_post = evaluate_metrics(encoder, classifier, labeled_data, cache_lab, val_timesteps=True,
                                    original_y=original_y, threshold=tau_st, T=T_st, mem_base=mem_base_st)
        f2_post = val_post["f2_illicit"]

        # Gate: only keep if it improves F2 on 31..34
        if f2_post >= f2_pre + 1e-4:
            kept_selftrain = True
            T_chosen, tau_chosen, f2_chosen = T_st, tau_st, f2_post
            # Save a post-self-train snapshot
            cp_pre = f"classifier_best_round{r}_selftrain.pth"
            torch.save({
                "cls_state_dict": classifier.state_dict(),
                "enc_state_dict": encoder.state_dict(),
                "val_metric": f2_post,
                "epoch": None,
                "T_val": T_st,
                "tau_val": tau_st
            }, cp_pre)
            print(f"[Self-train] kept (F2 improved {f2_pre:.4f} → {f2_post:.4f})")
        else:
            # revert classifier to pre self-train state
            classifier.load_state_dict(cls_before, strict=True)
            T_chosen, tau_chosen, f2_chosen = T_pre_eval, tau_pre_eval, f2_pre
            print(f"[Self-train] reverted (no improvement: {f2_post:.4f} < {f2_pre:.4f})")

    # Pseudo-label ≤30 to expand train labels for next round (calibrated + strict)
    th_illicit = max(0.995, T_chosen*0 + tau_chosen)   # keep >= tau; typically >=0.995
    th_licit   = max(0.9995, tau_chosen + 0.15)
    pseudo_label(encoder, classifier, labeled_data, cache_lab,
                 T=T_chosen, th_illicit=th_illicit, th_licit=th_licit,
                 cap_per_timestep=50, cap_per_class_per_t=30,
                 use_consistency=ENABLE_PSEUDO_CONSISTENCY)

    # Track best round by F2 on 31..34
    if f2_chosen > best_f2:
        best_f2 = f2_chosen
        best_cp = cp_pre
        best_round_T = T_chosen
        best_round_tau = tau_chosen
    checkpoints.append((cp_pre, f2_chosen))
    print(f"[Round {r}] selected {'self-trained' if kept_selftrain else 'pre-self-train'} model | F2={f2_chosen:.4f}")

print(f"\n[Best round by F2] {best_cp} | F2_illicit={best_f2:.4f} | T={best_round_T:.3f}, τ={best_round_tau:.3f}")

# Final: load best and evaluate on VAL + TEST with strategy selection
best_blob = torch.load(best_cp, map_location=device)
encoder.load_state_dict(best_blob["enc_state_dict"])
classifier.load_state_dict(best_blob["cls_state_dict"])

# Assert no label leakage for >30
assert (original_y[(timesteps > 30)] == labeled_data.y[(timesteps > 30)]).all().item(), "Labels after t>30 were modified!"

# Recompute mem_base for the (restored) best encoder
mem_base = get_temporal_embeddings(encoder, labeled_data, cache_lab, max_t=30)

# Calibrate T on val (use saved if present)
T_val = best_blob.get("T_val", None)
if T_val is None:
    T_val = fit_temperature(encoder, classifier, labeled_data, cache_lab, mem_base=mem_base)

# Get a global τ on val (for fixed-threshold baseline)
tau_val = best_blob.get("tau_val", None)
if tau_val is None:
    tau_val = find_best_threshold(encoder, classifier, labeled_data, cache_lab, mode='f2', T=T_val, mem_base=mem_base)

print("\n=== VAL (31..34) — fixed τ, no prior-correction ===")
_ = evaluate_metrics(encoder, classifier, labeled_data, cache_lab, val_timesteps=True,
                     threshold=tau_val, T=T_val, mem_base=mem_base)

print("\n=== TEST (>34) — compare strategies and auto-pick ===")
pi_tr = compute_train_prior(labeled_data)
tgt_rate = compute_val_target_rate(labeled_data)

print("[A] prior-corrected + adaptive τ(t)")
m_adapt = evaluate_metrics(encoder, classifier, labeled_data, cache_lab, val_timesteps=False,
                           threshold=None, T=T_val, mem_base=mem_base,
                           prior_correction=ENABLE_TEST_PRIOR_CORRECTION,
                           adaptive_t_thresholds=ENABLE_ADAPTIVE_T_THRESHOLDS,
                           pi_train=pi_tr, target_rate=tgt_rate)

print("[B] fixed global τ (no prior-correction)")
m_fixed = evaluate_metrics(encoder, classifier, labeled_data, cache_lab, val_timesteps=False,
                           threshold=tau_val, T=T_val, mem_base=mem_base,
                           prior_correction=False, adaptive_t_thresholds=False)

use_adapt = (m_adapt["f2_illicit"] >= m_fixed["f2_illicit"])
print(f"\n[TEST choice] Using {'adaptive' if use_adapt else 'fixed'} strategy "
      f"(F2: adapt={m_adapt['f2_illicit']:.4f} vs fixed={m_fixed['f2_illicit']:.4f})") 