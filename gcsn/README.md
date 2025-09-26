How to use (quick)

Save the three files next to your dataset.

Prepare PyG Data graphs with x, edge_index, y, train_mask/val_mask/test_mask.

Train:

python train_gcsn.py \
  --data_dir ./data_cfg_pt \
  --train_glob "train_*.pt" \
  --val_glob "val_*.pt" \
  --test_glob "test_*.pt" \
  --epochs 120 --batch_size 1 --amp


This writes ckpt_gcsn/best.pt and feature_keep_probs.json.

Predict (and export evidence):

python predict_gcsn.py \
  --ckpt ckpt_gcsn/best.pt \
  --data ./data_cfg_pt/test_000.pt \
  --out_dir preds_test \
  --edge_threshold 0.6


Per-graph .pt files contain node predictions plus the indices of kept edges, i.e., your minimal explanatory subgraph under the learned mask.

A model built for “all-or-nothing” relevance
Intuition

Given a CFG 
𝐺
=
(
𝑉
,
𝐸
)
G=(V,E) with node features 
𝑥
𝑣
x
v
	​

 and target types 
𝑦
𝑣
y
v
	​

, suppose there exists a minimal deterministic evidence set 
𝑆
⋆
⊆
𝐹
S
⋆
⊆F (features/subgraph) such that

Pr
⁡
(
𝑦
𝑣
∣
𝐺
,
𝑋
)
=
1
{
𝑓
(
𝑋
𝑆
⋆
,
𝐺
𝑆
⋆
)
=
type
}
.
Pr(y
v
	​

∣G,X)=1{f(X
S
⋆
	​

,G
S
⋆
	​

)=type}.

Then the right bias is hard sparsity on features and subgraph structure, not smooth shrinkage. We want the network to choose a tiny subgraph and a tiny set of features that suffice to determine the type, and to be penalized for using anything else.

Architecture: Gated Causal Subgraph Network (GCSN)

Components (all learned end-to-end):

Per-feature L0 gates (Concrete/L0)
For each node feature channel 
𝑑
d, learn 
𝑧
𝑑
∈
{
0
,
1
}
z
d
	​

∈{0,1} (relaxed to the Concrete distribution during training) and mask features by 
𝑥
(
𝑚
𝑎
𝑠
𝑘
𝑒
𝑑
)
=
𝑧
⊙
𝑥
x
(masked)
=z⊙x.

Parameterization: logits 
𝛼
𝑑
α
d
	​

; sample 
𝑢
 ⁣
∼
 ⁣
Uniform
(
0
,
1
)
u∼Uniform(0,1), 
𝑠
=
𝜎
(
(
log
⁡
𝑢
−
log
⁡
(
1
−
𝑢
)
+
𝛼
𝑑
)
/
𝜏
)
s=σ((logu−log(1−u)+α
d
	​

)/τ) with temperature 
𝜏
↓
0
τ↓0; straight-through at test time.

Loss term: 
𝜆
𝑓
∑
𝑑
𝐸
[
𝑧
𝑑
]
λ
f
	​

∑
d
	​

E[z
d
	​

] (feature sparsity).

Edge/Node hard subgraph gate (Top-k straight-through)
Compute importance scores 
𝑎
𝑒
a
e
	​

 for edges and 
𝑎
𝑣
a
v
	​

 for nodes (via small MLPs on edge/node features + local context). Keep only the top-k edges/nodes (straight-through one-hot), zero the rest. This produces a selected subgraph 
𝐺
~
G
~
.

Loss term: 
𝜆
𝑔
(
∣
𝐸
(
𝐺
~
)
∣
+
𝛾
∣
𝑉
(
𝐺
~
)
∣
)
λ
g
	​

(∣E(
G
~
)∣+γ∣V(
G
~
)∣).

Relational message passing only on the selected subgraph
Any standard GNN (e.g., GAT/GINE) runs on 
𝐺
~
G
~
 with masked features. This encodes the bias that only selected evidence can influence prediction.

Type head
Per-node classifier to predict the Checker-style type label (or a multi-label vector of qualifiers).

Stability & identifiability auxiliary terms

Counterfactual stability: train on paired program variants 
(
𝐺
,
𝐺
′
)
(G,G
′
) produced by semantics-preserving rewrites; enforce 
𝑝
(
𝑦
∣
𝐺
~
,
𝑋
~
)
≈
𝑝
(
𝑦
∣
𝐺
~
′
,
𝑋
~
′
)
p(y∣
G
~
,
X
~
)≈p(y∣
G
~
′
,
X
~
′
) with KL penalty.

Minimality via MDL: add 
𝛽
 ⁣
⋅
 ⁣
codelen
(
𝐺
~
,
𝑧
)
β⋅codelen(
G
~
,z) (implemented as sparsity + entropy penalties) to prefer smaller explanations.

De-correlation (DeCov): penalize covariance of hidden features to avoid “using many weakly correlated features” when a single decisive one exists.

Training objective

𝐿
=
CE
(
𝑦
,
𝑦
^
)
⏟
accuracy
+
𝜆
𝑓
∑
𝑑
𝐸
[
𝑧
𝑑
]
+
𝜆
𝑔
(
∣
𝐸
(
𝐺
~
)
∣
+
𝛾
∣
𝑉
(
𝐺
~
)
∣
)
+
𝜆
𝑠
KL
⁡
(
𝑝
𝜃
(
𝑦
∣
𝐺
)
,
𝑝
𝜃
(
𝑦
∣
𝐺
′
)
)
+
𝜆
𝑑
𝑐
 
DeCov
.
L=
accuracy
CE(y,
y
^
	​

)
	​

	​

+λ
f
	​

d
∑
	​

E[z
d
	​

]+λ
g
	​

(∣E(
G
~
)∣+γ∣V(
G
~
)∣)+λ
s
	​

KL(p
θ
	​

(y∣G),p
θ
	​

(y∣G
′
))+λ
dc
	​

DeCov.

Why this is “mathematically right” here
If 
𝑆
⋆
S
⋆
 is a Markov blanket of the label (deterministic or near-deterministic), then any superset predicts but is non-minimal; any subset fails. Under standard L0 recovery conditions (restricted strong convexity / identifiability) and sufficient augmentation coverage, the global minimizer of 
𝐿
L selects 
𝑆
⋆
S
⋆
 (or an equivalent minimal set) with probability 
→
1
→1 as data grows. In practice, the Concrete/L0 + top-k gates are the computationally tractable surrogate for exact subset search.

1) Where this model can stumble — and how to harden it

A. Gate collapse (too sparse or too dense).

Symptom: training loss stalls; masks go all 0s or nearly all 1s.

Fixes:

Warm-start gates: small positive logit init for features and edges (e.g., init_logit=+0.5).

Anneal L0 temperature τ from 2/3 → 0.1 over epochs; linearly/cosine.

Budget scheduling: start with larger edge k_ratio (e.g., 0.4) then decay to target (e.g., 0.1).

Use loss-adaptive gate penalties (increase λ if mask density exceeds target, decrease if too small).

B. Non-differentiable edge pruning kills learning signal.

Symptom: if edges are hard-dropped by reconstructing edge_index, gradients won’t reach the scorer.

Fix: Keep full edge_index; pass the (straight-through) mask as a continuous edge feature into an attention GNN (e.g., GATv2Conv(edge_dim=1)), so attention multiplies by the mask → gradients flow to the gate scorer.

C. Shortcut leakage via label-adjacent features.

Symptom: the model “reads” target-only features (e.g., oracle types in scope) instead of CFG evidence.

Fix: audit features; if needed, block gradients through “label-adjacent” channels (stop-grad), or place them behind higher λ L0 gates, or drop them entirely. Add counterfactuals: shuffle/perturb those channels while holding CFG constant; penalize prediction change.

D. Class imbalance.

Symptom: great loss, poor minority-type recall.

Fix: class-weighted CE (weight=1/log(freq+δ)), macro-F1 early-stopping, and minority-oversampling of graphs/nodes.

E. Oversmoothing on long CFG paths / large graphs.

Symptom: depth hurts performance; signals blur.

Fix: shallow (2–3) GATv2 layers, residual MLP head, and the edge gate itself (selective propagation) reduce smoothing. Optionally, add PairNorm or Jumping-Knowledge.

F. Instability across semantics-preserving rewrites.

Symptom: predictions flip after variable-renames or commutations.

Fix: train-time equivariance set: batch pairs of augmented graphs; add KL stability regularizer between outputs. Also include DeCov on hidden activations to avoid reliance on incidental correlations.

G. OOD projects / symbol sets.

Symptom: poor generalization to unseen APIs/coding styles.

Fix: mixed-project training, token bucketing & hashing for rare/OOV identifiers, and input dropout on identifier channels.

H. Runtime & memory (big E, big N).

Fix: neighbor sampling (mini-batch by graph), float16 AMP, gradient clipping, and mask-aware pruning at inference (threshold the learned edge mask to actually shrink the graph before the forward).

2) Complete implementation (PyTorch Geometric)

The scripts assume you already have a dataset that yields torch_geometric.data.Data objects with:

x [N,D] node features,

edge_index [2,E] CFG edges,

y [N] integer node labels,

boolean masks: train_mask, val_mask, test_mask.
(If you only have per-graph labels, you can adapt the reduction in the loss/pred blocks.)