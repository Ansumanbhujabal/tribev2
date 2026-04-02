# NeuroLens — Learnings, Interview Prep & Technical Deep Dive

> Everything you should know from building this project, organized for interviews, portfolio presentations, and continued learning.

---

## Table of Contents

1. [Project Summary (Elevator Pitch)](#1-project-summary)
2. [Technical Learnings](#2-technical-learnings)
3. [How This Fits Your Profile](#3-how-this-fits-your-profile)
4. [Interview Questions & Answers](#4-interview-questions--answers)
5. [RSA Deep Dive](#5-rsa-deep-dive)
6. [System Design Patterns](#6-system-design-patterns)
7. [How TRIBE v2 Works (Technical Architecture)](#7-how-tribe-v2-works-technical-architecture)
8. [Neuroscience Concepts](#8-neuroscience-concepts-what-you-need-to-know)
9. [Common Follow-Up Questions](#9-common-follow-up-questions)
10. [Key Research Findings](#10-key-research-findings)
11. [Advanced Interview Questions (from Findings)](#11-advanced-interview-questions)
12. [What to Study Next](#12-what-to-study-next)

---

## 1. Project Summary

### 30-Second Pitch

"I built NeuroLens — an interactive neuroscience playground on top of Meta's TRIBE v2 brain encoding model. It has three modules: Predict (visualize predicted brain activation from any video/audio/text), Match (find neurally similar content using cosine similarity search), and Eval (benchmark AI models against brain predictions using RSA). I ran it on 6 video stimuli and found that CLIP achieves 38.6% brain alignment — 13x better than Whisper and anti-correlated with GPT-2 — proving visual representations dominate brain responses to video. The system uses a compute/serving separation — heavy inference pre-computed on GPU, interactive exploration on CPU with sub-second responses."

### 60-Second Version (add these details)

"The Eval module uses RSA — Representational Similarity Analysis — which compares the *geometry* of how stimuli relate to each other in AI embedding space versus brain activation space. Instead of comparing vectors directly (which is impossible when brain space is 20,484 dimensions and CLIP is 512), RSA compares pairwise similarity matrices using Spearman rank correlation. This gives a single alignment score per model.

The findings were striking: speech stimuli cluster tightly (0.801 similarity between Jack Ma and Ronaldo speeches), but two stimuli in the same 'silent video' category can be neurally orthogonal (0.001 similarity). Music produces the most extreme brain response — Language Areas at -0.414, suggesting active neural competition. Motor Cortex is the only region positive across all stimuli, pointing to embodied simulation as the universal response to video.

I implemented the full pipeline: 9 Python modules with 31 tests, automated results generation (90 brain plots, 78 radar charts, 85 JSON files), and a cache architecture using .npz for brain predictions, .pt for model embeddings, and .json for ROI summaries — mirroring the three-tier feature store pattern used in production ML systems."

---

## 2. Technical Learnings

### What You Built (and can talk about)

| Area | What You Did | Interview Signal |
|------|-------------|-----------------|
| **System Design** | Compute/serving separation — GPU precompute, CPU serving | Production ML architecture |
| **AI Evaluation** | RSA implementation + statistical power analysis (n=6 → need n≥20) | Eval methodology with statistical rigor |
| **Multimodal AI** | Worked with 7 models (LLaMA 3.2, V-JEPA2, Wav2Vec-BERT, DINOv2, CLIP, Whisper, GPT-2) | Breadth across modalities |
| **Cache Architecture** | Three-format cache (.npz/.pt/.json) mirroring feature store patterns | Data engineering, storage design |
| **TDD** | 31 tests across 8 files, test-first development | Engineering discipline |
| **Batch Analysis** | Automated results generator — 90 brain plots, 78 radar charts, 85 JSONs from single command | Pipeline engineering |
| **Research Analysis** | Multi-agent parallel analysis of 250+ artifacts → cross-module synthesis | Data analysis at scale |
| **Environment Management** | Solved numpy conflicts, moviepy v1/v2 incompatibility, Colab deployment | Real-world DevOps/MLOps |
| **Brain Visualization** | nilearn cortical surface plots on fsaverage5 mesh (5 views × 3 timepoints × 6 stimuli) | Domain-specific visualization |
| **Similarity Search** | Cosine similarity matching, contrast-based ranking, neural clustering | Recommendation/retrieval systems |

### Skills You Practiced

1. **Designing before coding** — Wrote a full spec and 12-task plan before touching code
2. **TDD across a package** — Every module had failing tests written first
3. **Dependency conflict resolution** — numpy pinning, moviepy v1/v2, Colab environment management
4. **Runtime monkey-patching** — Patched moviepy v2's import path at runtime for neuralset compatibility
5. **GPU memory management** — Diagnosed OOM from long video (181s → 398 timesteps), trimmed clips to 15s max
6. **Cache design** — Chose appropriate formats for different data types and access patterns
7. **Cross-environment debugging** — Made the same code work locally and on Colab with different Python versions and pre-installed packages

---

## 3. How This Fits Your Profile

### For AI Architect Roles

| What They Want | What NeuroLens Shows |
|---------------|---------------------|
| "Design a system that scales" | Compute/serving separation, cache architecture, resource-aware design (Colab T4 constraints) |
| "How do you evaluate AI models?" | RSA methodology — not just "run a benchmark" but understanding representational alignment |
| "Can you work across the stack?" | Model inference, data pipeline, visualization, testing, deployment, documentation |
| "How do you handle production constraints?" | GPU memory budgeting, dependency management, environment compatibility |

### For AIOps Roles

| What They Want | What NeuroLens Shows |
|---------------|---------------------|
| "How do you monitor ML systems?" | RSA as a monitoring metric (track alignment score over model versions) |
| "How do you handle data pipelines?" | Precompute pipeline with multiple models, cache management, format choices |
| "How do you debug production issues?" | Systematic debugging of Colab deployment (4 issues in one session) |
| "Do you understand infrastructure?" | Colab GPU constraints, memory management, dependency conflicts |

### For Master's Applications (SOP angle)

"Through building NeuroLens, I demonstrated the ability to bridge AI engineering with neuroscience research methodology. I implemented Representational Similarity Analysis — a technique from computational neuroscience (Kriegeskorte et al., 2008) — to evaluate how closely AI model representations align with biological neural responses. This interdisciplinary approach reflects my research interest in understanding how artificial and biological intelligence process multimodal information."

---

## 4. Interview Questions & Answers

### Q1: "Tell me about a project where you designed the system architecture."

**Answer:**

"I built NeuroLens, an interactive tool for comparing AI models against human brain responses. The key architectural decision was **compute/serving separation**.

The heavy part — running TRIBE v2 (a 1GB multimodal brain encoding model), CLIP, Whisper, and GPT-2 on video stimuli — takes 5-10 minutes per clip and requires a GPU. The interactive part — brain visualization, similarity search, RSA scoring — needs to be sub-second.

So I split it into two notebooks: a precompute notebook that runs once on GPU and generates a cache, and a main notebook that loads the cache and runs entirely on CPU. The cache uses three formats matched to access patterns: `.npz` for large numerical arrays (brain predictions), `.pt` for PyTorch model embeddings, and `.json` for small pre-aggregated ROI summaries. This mirrors the offline/online/metadata tiers you'd see in a production feature store."

**Follow-up they might ask:** "What happens when you add a new model?"

"Adding a new model is just creating a new directory under `embeddings/` and saving `.pt` files. The `CacheManager.available_models()` method auto-discovers subdirectories. No schema changes, no code changes to the eval module — it picks up new models automatically. This is the same extensibility pattern you'd use in a model registry."

---

### Q2: "How do you evaluate AI models beyond standard benchmarks?"

**Answer:**

"In NeuroLens, I used Representational Similarity Analysis — RSA. Instead of measuring task performance (like accuracy on MMLU), RSA measures whether a model's internal representational structure matches a reference — in this case, predicted human brain responses.

The key insight is that you can't directly compare a CLIP embedding (512 dimensions) to a brain activation pattern (20,484 vertices). But you *can* compare how a set of stimuli relate to each other in each space. If stimuli A and B are similar in CLIP space and also similar in brain space, the representational geometry is aligned.

Concretely: I compute pairwise cosine similarity matrices for both the AI model embeddings and the brain predictions across all stimuli, then take the Spearman rank correlation of their upper triangles. A score of 1.0 means perfect alignment; 0.0 means no relationship.

I chose Spearman over Pearson because we only care about rank ordering — whether the model agrees with the brain on *which pairs of stimuli are most similar* — not the absolute magnitude of similarity values."

**Follow-up:** "When would RSA fail or be misleading?"

"Three main limitations:
1. **Sample size** — With fewer than 3 stimuli, there aren't enough pairs for meaningful correlation. I coded a guard for this: the function returns 0.0 if embeddings < 3.
2. **Stimulus selection bias** — If all stimuli are from the same category (all speech clips), the similarity matrix has low variance and RSA scores are noisy. You need diverse stimuli.
3. **Aggregation loss** — I average brain predictions across timesteps before computing RSA, which loses temporal dynamics. A speech clip might activate auditory cortex early and language areas later — averaging collapses this."

---

### Q3: "Walk me through how you'd handle a dependency conflict in production."

**Answer:**

"I hit this exact problem deploying to Google Colab. TRIBE v2's `pyproject.toml` pins `numpy==2.2.6`, but Colab pre-installs its own numpy version. Force-installing 2.2.6 broke Colab's pre-installed TensorFlow and numba.

My approach was systematic:
1. **Identify the real constraint** — Does TRIBE v2 actually *need* numpy 2.2.6, or does it just work with any 2.x? I tested: it works fine with Colab's numpy.
2. **Install with `--no-deps`** — This registers the package without pulling its pinned dependencies.
3. **Install actual dependencies individually** — Everything except the conflicting numpy pin.
4. **Verify** — Import the package and run a smoke test in the same cell.

The general pattern: when you have a constrained environment (Colab, Docker base image, corporate Python), install the constrained package with `--no-deps` and manually manage its dependency tree. This is the same approach used in production Docker images where you control the base layer and don't want pip to blow it up.

Another issue: `neuralset` does `from moviepy import VideoFileClip` which works in moviepy v1 but broke in v2 (they moved the import path). But neuralset also uses `.subclipped()` which is a v2 API. So it needs v2 with v1 import paths. I monkey-patched it at runtime:

```python
from moviepy.video.io.VideoFileClip import VideoFileClip
moviepy.VideoFileClip = VideoFileClip
sys.modules['moviepy'].VideoFileClip = VideoFileClip
```

This is a last-resort pattern — I'd prefer to fix it upstream. But for a demo that needs to work *now*, runtime patching with clear documentation is acceptable."

---

### Q4: "How would you scale this system to handle 10,000 stimuli?"

**Answer:**

"Current design handles 6 stimuli with files on disk. For 10,000, I'd make these changes:

**Storage tier:**
- Replace individual `.npz` files with **HDF5 or Zarr** — these support chunked, compressed arrays with random row access. Loading one stimulus's predictions doesn't require reading all 10,000.
- Move embeddings from `.pt` files to a **vector database** (FAISS index, Pinecone, or Weaviate) for approximate nearest neighbor search instead of brute-force cosine similarity.
- ROI summaries move to a **key-value store** (Redis or DynamoDB) for O(1) lookup.

**Precompute tier:**
- Parallelize across stimuli using a **job queue** (Celery + Redis, or Kubernetes Jobs). Each stimulus is an independent task.
- Add **model versioning** — tag each cached artifact with the model version that produced it. When TRIBE v2 is updated, only invalidate and recompute the TRIBE outputs, not the CLIP/Whisper embeddings.
- Add **incremental precompute** — only process new stimuli, don't re-run the entire library.

**Serve tier:**
- RSA computation with 10,000 stimuli means a 10,000 x 10,000 similarity matrix = 100M entries. Upper triangle has ~50M pairs. Spearman on 50M values takes ~seconds with scipy, but you'd want to **subsample or approximate** — random sampling of pairs gives a good RSA estimate with O(k) pairs instead of O(n^2).
- Pre-compute RSA scores nightly and cache them. The interactive layer just reads the leaderboard.

**Cost estimate:**
- 10,000 clips × ~15 seconds × ~5 min/clip on T4 = ~833 GPU-hours. At $0.35/hr (Colab Pro) = ~$290 one-time.
- Storage: 10,000 × ~2MB (brain_preds) = ~20GB. Fits in a single S3 bucket.
- Serving: CPU-only, auto-scaling based on request load."

---

### Q5: "What's the difference between cosine similarity and RSA?"

**Answer:**

"They operate at different levels.

**Cosine similarity** is a first-order comparison — it directly measures the angle between two vectors. In the Match module, I use it to find stimuli whose brain predictions are most similar to a target pattern: `cos(target, prediction)`.

**RSA** is a second-order comparison — it compares *patterns of similarities*. You first compute pairwise cosine similarity within each space (creating NxN matrices), then correlate those matrices. RSA answers: 'Do the relationships between stimuli look the same in both spaces?'

The reason we need RSA for the Eval module is dimensionality mismatch. CLIP embeddings are 512-dimensional. Brain predictions are 20,484-dimensional. You can't compute cosine similarity between them directly. But you *can* ask: 'Does CLIP think stimulus A and B are similar to the same degree that the brain does?' — that's what RSA measures."

---

### Q6: "How did you handle GPU memory constraints?"

**Answer:**

"I hit an OOM on Colab's T4 (15GB VRAM). The romantic couple clip was 181 seconds long — TRIBE v2 processes video in 0.5-second frames, so that's 362 frames. The V-JEPA2 video encoder produces a (120, 20, 1408) tensor per 60-second chunk, and with two chunks plus all three modality encoders loaded simultaneously, it exceeded 15GB.

The fix was simple: trim all clips to 15 seconds maximum. But the deeper lesson is about **GPU memory budgeting**: video duration maps linearly to frame count, which maps to tensor size in the encoder. For a production system, I'd add a pre-check:

```python
estimated_vram = (duration_sec / 0.5) * frames_per_chunk * hidden_dim * 4  # float32
if estimated_vram > available_vram * 0.8:  # 80% safety margin
    chunk_and_process_sequentially()
```

I also added `gc.collect()` and `torch.cuda.empty_cache()` between stimuli to reclaim memory from previous inference runs."

---

### Q7: "Explain the cache architecture. Why three different formats?"

**Answer:**

"Each format matches the data's nature and access pattern:

| Data | Format | Why |
|------|--------|-----|
| Brain predictions (T, 20484) float32 arrays | `.npz` | Large numerical data, needs compression, NumPy-native, cross-framework portable |
| Model embeddings (512-768 dim) tensors | `.pt` | PyTorch-native, preserves dtype/device metadata, loaded with `weights_only=True` for security |
| ROI summaries (9 float values) | `.json` | Small metadata, human-readable, can be inspected without code |

This maps to the three-tier feature store pattern:
- `.npz` = **offline store** (full-fidelity batch data, like BigQuery/Parquet)
- `.json` = **online store** (pre-aggregated for fast serving, like Redis/DynamoDB)
- `.pt` = **model artifact store** (versioned per-model, like MLflow registry)

The directory structure `embeddings/{model_name}/{stimulus_id}.pt` is also a design choice — adding a new model is just creating a new directory. No schema migration, no code changes. The `CacheManager.available_models()` auto-discovers subdirectories."

---

### Q8: "What would you do differently if you built this again?"

**Answer:**

"Three things:

1. **Add version tracking to the cache.** Right now there's no metadata about which model version produced each file. If TRIBE v2 updates, I can't tell which cache files are stale. I'd add a `cache_manifest.json` with model version hashes.

2. **Use HDF5 instead of individual .npz files.** With 6 stimuli, individual files work. At scale, opening 10,000 files creates filesystem overhead. HDF5 gives you a single file with random access to any stimulus's data.

3. **Stream brain predictions instead of loading all at once.** The time slider in the Predict module currently loads the entire (T, 20484) array to display one timestep. For long stimuli, I'd memory-map the array or use chunked loading."

---

## 5. RSA Deep Dive

### The Core Problem: How Do You Compare Brain Predictions to AI Models?

TRIBE v2 outputs a brain activation map: `(n_timesteps, 20,484 vertices)` — a prediction of what every point on the cortical surface does for a given video.

CLIP outputs a 512-dim embedding. GPT-2 outputs a 768-dim embedding. Whisper outputs a 512-dim embedding.

**You can't directly compare a 20,484-dim brain vector to a 512-dim CLIP vector.** They live in completely different spaces. So how do you compare them?

### The Answer: Compare Relationships, Not Vectors

Instead of comparing vectors directly, RSA compares **how stimuli relate to each other** in each space. The key insight: if both spaces agree that "Jack Ma and Ronaldo are similar, but both are different from Food Reel," then they have aligned representational geometry — regardless of dimensionality.

### The Full Pipeline (How It Actually Works)

```
Step 1: Run TRIBE v2 on all 6 videos → 6 brain prediction vectors
        Then aggregate 20,484 vertices → 9 ROI scores per stimulus
Step 2: Run CLIP on all 6 videos → 6 CLIP embedding vectors (512-dim)
Step 3: For brain space, compute pairwise cosine similarity
         between all 6 stimuli → 6×6 matrix (15 unique pairs)
Step 4: For CLIP space, compute the same → 6×6 matrix (15 unique pairs)
Step 5: Extract upper triangles of both matrices (15 values each)
Step 6: Spearman rank correlation between the two lists of 15 values
         → single RSA score
```

### Concrete Example (With Real-ish Numbers)

Say you have 3 stimuli. The brain says:
```
Brain similarities:
  Jack Ma ↔ Ronaldo = 0.80  (similar brain patterns — both speech)
  Jack Ma ↔ Food    = 0.16  (very different — speech vs visual)
  Ronaldo ↔ Food    = 0.20  (different)
```

CLIP says:
```
CLIP similarities:
  Jack Ma ↔ Ronaldo = 0.75  (similar visual frames — both people talking)
  Jack Ma ↔ Food    = 0.22  (different — person vs food)
  Ronaldo ↔ Food    = 0.30  (different)
```

Both agree on the ranking: Jack Ma & Ronaldo are most similar, Food is the odd one out. → **High RSA score** (CLIP "thinks like the brain").

Now GPT-2 says:
```
GPT-2 similarities:
  Jack Ma ↔ Ronaldo = 0.95  (both "motivation, speech" text → very similar)
  Jack Ma ↔ Food    = 0.10  (different text descriptions)
  Ronaldo ↔ Food    = 0.12  (different text descriptions)
```

GPT-2 over-groups the speech stimuli and can't see that the brain treats them differently (Jack Ma is auditory-linguistic dominant, Ronaldo is motor-dominant). → **Low or negative RSA**.

### Why Not Compare Embeddings Directly?

| Approach | Problem |
|----------|---------|
| Direct cosine sim (brain vs CLIP) | Dimensions don't match (20,484 vs 512) — literally impossible |
| Project to same space (PCA, linear probe) | Requires paired training data, loses relational structure |
| CKA (Centered Kernel Alignment) | Works but less interpretable than RSA |
| **RSA** | **Works across any dimensions**, only needs multiple stimuli in both spaces |

That's the elegance of RSA — it's **model-agnostic and dimension-agnostic**. You can compare any two representation spaces as long as you have the same set of stimuli in both.

### The Role of ROI Aggregation

Before computing brain-space similarity, we don't use all 20,484 vertices directly (too noisy, too high-dimensional for just 6 stimuli). We aggregate:

```
20,484 vertices → group by HCP atlas regions → 9 ROI scores
  (Visual Cortex, Auditory Cortex, Language Areas, Motor Cortex, etc.)
```

So the "brain vector" for each stimulus is 9 numbers — one per brain region. This makes cosine similarity between stimuli meaningful and stable.

### The Full Pipeline in Code

```python
# 1. Load brain predictions (from TRIBE v2 precompute)
brain_preds = cache.load_brain_preds("clip_001")  # (T, 20484)

# 2. Aggregate to ROI summary (time-averaged)
roi_summary = summarize_by_roi_group(brain_preds.mean(axis=0))  # 9 values

# 3. Load model embeddings
clip_emb = cache.load_embedding("clip", "clip_001")  # (512,)

# 4. For each model, build pairwise similarity matrix across all stimuli
brain_sim_matrix = cosine_similarity(all_brain_rois)     # (6, 6)
clip_sim_matrix = cosine_similarity(all_clip_embeddings)  # (6, 6)

# 5. RSA = Spearman correlation of upper triangles
rsa_score = spearmanr(
    brain_sim_matrix[upper_triangle],
    clip_sim_matrix[upper_triangle]
).correlation
# → 0.386 for CLIP, 0.029 for Whisper, -0.143 for GPT-2
```

### What Each Score Means (Our Actual Results)

| Model | RSA Score | What It Tells Us |
|-------|-----------|-----------------|
| CLIP = +0.386 | Moderate alignment | CLIP's visual geometry partially matches brain geometry. Stimuli CLIP sees as similar are also neurally similar. |
| Whisper = +0.029 | Near-zero | Audio features don't capture brain organization. 3/6 stimuli are silent → Whisper gives near-identical embeddings for very different brain responses. |
| GPT-2 = -0.143 | Anti-correlated | Text-based similarity actively *contradicts* brain similarity. "Jack Ma" and "Ronaldo" look similar in text but differ in the brain. |

---

### The Math, Step by Step

Given N stimuli with representations in two spaces (AI model vs. brain):

**Step 1: Build pairwise similarity matrices**

```python
# For each space, compute NxN cosine similarity matrix
mat = np.stack(vectors)              # (N, D)
norms = np.linalg.norm(mat, axis=1, keepdims=True)
mat_normed = mat / norms
sim_matrix = mat_normed @ mat_normed.T  # (N, N), symmetric, diagonal=1
```

**Step 2: Extract upper triangle**

The matrix is symmetric, diagonal is trivially 1. Only the upper triangle has unique information:

```python
idx = np.triu_indices(n, k=1)  # k=1 skips diagonal
vec = sim_matrix[idx]          # N*(N-1)/2 values
```

For N=6 stimuli: 6*5/2 = **15 pairwise comparisons**.

**Step 3: Spearman rank correlation**

```python
corr, p_value = spearmanr(vec_a, vec_b)
```

Spearman formula: `ρ = 1 - (6 * Σd_i²) / (n*(n²-1))` where `d_i` is rank difference for pair i.

### Why Spearman, Not Pearson?

- **Pearson** is sensitive to scale. If CLIP outputs similarities in [0.8, 1.0] and GPT-2 in [0.1, 0.9], Pearson penalizes the scale difference even if the ranking is identical.
- **Spearman** only compares rank ordering. "Do both spaces agree that clip_1 and clip_2 are the *most similar* pair?" This is more appropriate because absolute similarity magnitudes across heterogeneous spaces are meaningless.
- Spearman is robust to outliers and monotonic nonlinearities.

### Interpretation

| Score | Meaning |
|-------|---------|
| ρ = 1.0 | Perfect alignment — the model ranks stimulus pairs identically to the brain |
| ρ = 0.0 | No relationship — model geometry is unrelated to brain geometry |
| ρ = -1.0 | Perfectly inverted — what the model finds similar, the brain finds dissimilar |
| ρ < 0.3 | Weak alignment |
| 0.3 < ρ < 0.7 | Moderate alignment |
| ρ > 0.7 | Strong alignment |

### RSA Origins

Formalized by **Kriegeskorte et al. (2008)**, *Frontiers in Systems Neuroscience*. Original purpose: compare brain regions to computational models without requiring shared representational space. Classic finding: V1 aligns with pixel-level features; IT cortex aligns with semantic categories — which is what CNNs accidentally learned to replicate (Yamins et al., 2014).

### RSA in Modern AI Evaluation (2025-2026)

- **Model-to-model alignment**: Does CLIP's geometry match GPT-4V's?
- **Multimodal consistency**: Does a model's audio representation of "glass breaking" relate to stimuli the same way as its visual representation?
- **Training dynamics**: Track RSA across checkpoints — models converge to similar geometry even from different seeds.
- **Bias detection**: If demographic groups rank differently in embedding space vs. human judgments, RSA reveals the misalignment.

### Other Representational Comparison Methods

| Method | What It Does | Pros | Cons |
|--------|-------------|------|------|
| **RSA** | Spearman on pairwise similarity matrices | Model-agnostic, handles dim mismatch | Requires diverse stimulus set |
| **CKA** (Centered Kernel Alignment) | Similarity of feature matrices using kernel trick | More sensitive to structure than RSA | Less interpretable |
| **Procrustes** | Optimal rotation/scaling to align spaces | Direct alignment | Requires same dimensions |
| **SVCCA** | Singular vector canonical correlation | Handles high-dimensional data | Computationally expensive |

---

## 6. System Design Patterns

### Compute/Serving Separation

```
[Precompute — GPU, runs once]          [Serve — CPU, per query]
  TRIBE v2 inference                     Load .npz from disk
  CLIP/Whisper/GPT-2 embeddings          Cosine similarity
  ROI aggregation                        RSA computation
  Save to cache                          Brain visualization
  ~10 min/stimulus                       ~100ms/query
```

**Production examples of this pattern:**
- **YouTube recommendations**: Item embeddings computed offline → ANN lookup online
- **Search engines**: Document indexing offline → query matching online
- **Feature stores**: Batch feature computation → online serving via Redis/DynamoDB

### Cache Architecture as Feature Store Analog

```
Production Feature Store    NeuroLens Cache         Format
────────────────────────────────────────────────────────
Offline store (Parquet)     brain_preds/            .npz
Online store (Redis)        roi_summaries/          .json
Model registry (MLflow)     embeddings/             .pt
```

### Key Design Decisions

1. **Staleness tolerance**: Brain predictions don't change → safe to precompute once. User preferences change hourly → need streaming.
2. **Cache invalidation**: What triggers recompute? New model version? New stimuli?
3. **Storage vs. compute tradeoff**: 6 clips × 2MB = 12MB storage vs. 10 min GPU time per clip.
4. **Extensibility**: Adding a model = creating a directory. No schema migration.

---

## 7. How TRIBE v2 Works (Technical Architecture)

> This section explains the complete pipeline from "you give it a video" to "it predicts brain activation." Understand this well — interviewers will ask about it.

### What Is TRIBE v2?

TRIBE v2 (by Meta AI / FAIR) is a **multimodal brain encoding model** — it takes video, audio, and text as input and predicts how the human brain would respond, as measured by fMRI. Think of it as: "Given this video clip, what would the fMRI scanner show?"

It's not a brain *simulator*. It's a *mapping function* from stimulus features to cortical activation patterns, trained on real fMRI data from humans watching videos.

### End-to-End Pipeline

```
Video/Audio/Text
       ↓
┌──────────────────────────────────┐
│  Feature Extractors (frozen)     │
│  ├─ Video: DINOv2-large (image)  │  ← visual keyframes
│  │         + V-JEPA2-ViTg (motion)│  ← temporal dynamics
│  ├─ Audio: Wav2Vec-BERT          │  ← speech + sound features
│  └─ Text:  LLaMA 3.2 (3B)       │  ← word-level semantics
└──────────────────────────────────┘
       ↓ (multi-layer features per modality)
┌──────────────────────────────────┐
│  Per-Modality Projection MLPs    │
│  Each modality → 384-dim vector  │
│  (hidden_size 1152 / 3 modalities)│
└──────────────────────────────────┘
       ↓ (concatenate across modalities)
┌──────────────────────────────────┐
│  Combiner MLP                    │
│  Concatenated → 1152-dim         │
│  LayerNorm + GELU                │
└──────────────────────────────────┘
       ↓ (+ positional embeddings)
┌──────────────────────────────────┐
│  Transformer Encoder (8 layers)  │
│  Self-attention fuses temporal + │
│  cross-modal information         │
│  Output: (batch, time, 1152)     │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│  Subject-Specific Linear Head    │
│  → (batch, 20484, 40 TRs)       │
│  20,484 = fsaverage5 vertices    │
│  40 TRs ≈ seconds of prediction  │
└──────────────────────────────────┘
```

### Feature Extractors (What Each Model Does)

| Model | Modality | What It Extracts | Layers Used | Frequency |
|-------|---------|-----------------|-------------|-----------|
| **DINOv2-large** | Video (spatial) | Object identity, scene layout, textures from keyframes | Layers [0.67, 1.0] | 2 Hz |
| **V-JEPA2-ViTg** | Video (temporal) | Motion, optical flow, temporal dynamics | Layers [0.75, 1.0] | 4s clips |
| **Wav2Vec-BERT** | Audio | Speech features, phonemes, sound textures | Layers [0.75, 1.0] | 2 Hz |
| **LLaMA 3.2 (3B)** | Text | Word semantics, context, narrative structure | 6 layers spread across depth | 2 Hz |

**Key insight**: These are *frozen* pre-trained models. TRIBE v2 doesn't fine-tune them — it learns a Transformer that maps their combined features to brain activation. This is why it's called an "encoding model": it encodes stimulus features into brain space.

### The Transformer Fusion

The 8-layer Transformer is where the magic happens:
- **Input**: Concatenated features from all modalities at each timestep → (batch, time, 1152)
- **Self-attention**: Each timestep attends to all other timesteps, allowing the model to learn cross-modal and temporal dependencies. "This sound at t=3 relates to this visual at t=1."
- **Positional embeddings**: Learned (not sinusoidal), max 1024 timesteps
- **Modality dropout** (0.3): Randomly drops entire modalities during training → forces the model to work with any subset of inputs
- **Output**: Per-timestep, 1152-dim fused representation

### Brain Output Format

- **Template**: fsaverage5 — a standard brain surface mesh
- **Vertices**: 10,242 per hemisphere × 2 = **20,484 total** (~1mm² each)
- **Output shape**: `(n_timesteps, 20484)` — a full cortical activation map per time point
- **Hemodynamic lag**: Predictions are offset by 5 seconds to compensate for the delay between neural activity and fMRI signal
- **Projection**: Uses `nilearn.surface.vol_to_surf` with ball-radius interpolation (3.0mm)

### Training

- **Data**: Real fMRI from humans watching video (Algonauts 2025, Lahner 2024 datasets)
- **Loss**: Mean Squared Error between predicted and actual fMRI activation per vertex
- **Metrics**: Pearson correlation per vertex, per subject, and top-1 retrieval accuracy
- **Training**: 15 epochs, Adam optimizer (lr=1e-4), OneCycleLR scheduler

### What NeuroLens Does With TRIBE v2

NeuroLens is the **analysis layer** on top of TRIBE v2's predictions:

```
TRIBE v2 predicts:  (n_timesteps, 20484)  ← raw vertex activations
NeuroLens adds:
  ├─ ROI aggregation  → 9 brain region scores (by averaging vertices per HCP region)
  ├─ Brain surface viz → 3D cortical plots colored by activation
  ├─ Similarity search → which stimuli produce similar brain patterns?
  ├─ Contrast analysis → which stimuli maximize/minimize specific regions?
  └─ RSA evaluation   → how well do AI models match brain geometry?
```

---

## 8. Neuroscience Concepts (What You Need to Know)

### fMRI and BOLD Signal

fMRI measures Blood Oxygen Level Dependent (BOLD) signal — a proxy for neural activity. When neurons fire, local blood flow increases, changing the MRI signal. Temporal resolution: ~1-2 seconds (TR = repetition time). Spatial resolution: ~2-3mm.

**Limitation**: fMRI is *indirect* — it measures blood flow, not neurons directly. There's a 4-6 second hemodynamic delay between neural activity and the BOLD response. TRIBE v2 compensates for this with a 5-second offset.

### fsaverage5 (20,484 vertices)

The human cortex is a folded sheet (~2,500 cm², roughly a large pizza). To compare across subjects, data is projected onto a standard template surface — **fsaverage**. fsaverage5 is the 5th-level icosphere subdivision:

```
Level 0:    12 vertices (icosahedron)
Level 5: 10,242 vertices per hemisphere × 2 = 20,484 total
```

Each vertex ≈ 1mm² of cortical surface. ~4mm effective spatial resolution.

### HCP MMP1.0 Atlas (180 regions per hemisphere)

The Human Connectome Project Multi-Modal Parcellation (Glasser et al., 2016, Nature) divides the cortex into 360 regions using myelin content, cortical thickness, functional connectivity, and topographic organization.

In NeuroLens, we group these 360 regions into 9 friendly ROI groups:
```python
ROI_GROUPS = {
    "Visual Cortex": ["V1", "V2", "V3", "V4"],          # seeing
    "Auditory Cortex": ["A1", "A4", "A5", ...],          # hearing
    "Language Areas": ["44", "45", "47l", ...],           # Broca's + Wernicke's
    "Motor Cortex": ["4", "3a", "3b", "6mp", ...],       # movement planning
    "Somatosensory Cortex": ["1", "2", "3a", "3b", ...], # touch/body sensation
    "Parietal Cortex": ["7Am", "7Pm", "IP0", ...],       # spatial awareness
    "Temporal Cortex": ["TE1a", "TE2a", ...],             # object/scene recognition
    "Prefrontal Cortex": ["8Av", "8C", "9m", ...],       # executive/planning
    "Face-Selective Areas": ["FFC", "OFA", "pSTS", ...],  # face processing
}
```

### What the ROI Scores Mean (Real Results)

From the actual Muay Thai Kick clip:
```
Parietal Cortex........... +0.152   ← spatial awareness, tracking the kick trajectory
Face-Selective Areas...... +0.131   ← detecting the fighter's body/face
Language Areas............ +0.105   ← internal narration ("that's a powerful kick")
Motor Cortex.............. +0.101   ← mirror neurons simulating the kick action
Visual Cortex............. +0.099   ← processing the visual scene
Somatosensory Cortex...... +0.083   ← "feeling" the impact vicariously
Prefrontal Cortex......... +0.081   ← attention, tracking what's happening
Auditory Cortex........... +0.042   ← near-zero (silent clip)
Temporal Cortex........... -0.031   ← slightly suppressed (not a memory/scene task)
```

Parietal cortex highest for a martial arts kick makes neuroscientific sense — it handles spatial awareness and body movement perception. 8 of 9 ROIs are positive — this is the most broadly engaging stimulus in the dataset.

### Key Neuroscience Principles from Our Findings

| Principle | Our Evidence | Literature |
|-----------|-------------|-----------|
| **Visual dominance** | CLIP (visual) RSA = 0.386 >> Whisper (audio) = 0.029 | Colavita effect (1974); ~30% of cortex is visual |
| **Motor theory of speech perception** | Motor Cortex is #1 ROI for Ronaldo speech (+0.207) | Liberman & Mattingly (1985) |
| **Embodied simulation / mirror neurons** | Motor Cortex positive for ALL stimuli (grand mean +0.084) | Rizzolatti & Craighero (2004) |
| **Music-language competition** | Music suppresses Language Areas to -0.414 | Fedorenko et al. (2011) — music and language share circuits |
| **Left lateralization for speech** | Jack Ma brain plots show left-hemisphere dominant hotspots | Broca (1861), Wernicke (1874) |

---

## 9. Common Follow-Up Questions

### "Why not use Pearson correlation for RSA?"
Pearson is sensitive to scale. If CLIP similarities are in [0.8, 1.0] and brain similarities in [-0.2, 0.3], Pearson would underestimate alignment. Spearman only compares rank ordering.

### "Why do you need at least 3 stimuli?"
With 2 stimuli, there's only 1 pair in the upper triangle. You can't compute correlation on 1 data point. With 3 stimuli, you get 3 pairs — technically computable but noisy. 6+ stimuli gives 15+ pairs — more reliable.

### "What if the brain predictions are wrong?"
TRIBE v2 predicts the "average subject." Individual brains differ. The model's noise ceiling (maximum possible accuracy) is limited by inter-subject variability. RSA comparisons are relative — even if absolute predictions are imperfect, the *relational structure* can still be meaningful.

### "Why cosine similarity and not Euclidean distance?"
Cosine similarity is scale-invariant — it measures angle, not magnitude. A brain region might have consistently higher activation (larger magnitudes) without being more "similar" to another region. Cosine captures direction, which is what we care about for representational geometry.

### "How is this different from just running a benchmark like MMLU?"
MMLU measures task performance (can the model answer questions?). RSA measures representational structure (does the model organize information like the brain does?). A model could score perfectly on MMLU while having completely non-brain-like representations. RSA captures something deeper about how information is structured internally.

### "What about temporal dynamics?"
Current implementation averages brain predictions across time before computing RSA. This loses temporal information. A stronger approach: compute RSA at each timestep and track how alignment changes over the course of a stimulus. Speech might align with language areas early and semantic areas later.

### "How would you deploy this as a web app?"
Replace the notebook with: FastAPI backend (serves RSA scores, brain predictions via REST API), React frontend (renders brain visualizations with Three.js/Niivue), Redis for caching pre-computed results, S3 for storing brain_preds/embeddings. The architecture is already separated — just swap the notebook interface for an API.

---

## 10. Key Research Findings

> From running the full pipeline on 6 video stimuli (2 Speech, 3 Silent+Visuals, 1 Music), evaluated against CLIP, Whisper, and GPT-2. These are real results you can discuss in interviews.

### The Five Headline Numbers

| Finding | Value | Why It Matters |
|---------|-------|---------------|
| **CLIP brain alignment** | RSA = 0.386 (38.6%) | Falls within published neuroscience benchmarks (Kriegeskorte 0.3-0.6). 13x better than Whisper. |
| **Speech neural clustering** | Similarity = 0.801 | Strongest pair in dataset — speech creates tight, reproducible brain signatures |
| **Music language suppression** | Language Areas = -0.414 | Most extreme ROI value — active neural competition, not just absence of activation |
| **Category label failure** | Muay Thai ↔ Romantic = 0.001 | Same category ("Silence+Visuals"), neurally orthogonal. Content semantics > category labels. |
| **Motor universality** | Grand mean = +0.084 | Only ROI positive across all 6 stimuli. Embodied simulation is the universal response to video. |

### Three Major Principles

**1. Visual Dominance Hypothesis**
- Predict module: Visual Cortex shows strongest category modulation (spread = 0.330)
- Match module: Stimuli cluster by visual similarity, not auditory content
- Eval module: CLIP (visual, single frame) = 0.386 >> Whisper (audio, full track) = 0.029
- Consistent with Colavita visual dominance effect and ~30% of cortex being visual

**2. Language Suppression by Music**
- Music drives Language Areas to -0.414 — the most extreme value in the entire dataset
- Music ↔ Jack Ma (speech) similarity = 0.090 — near-zero, opposite ends of neural space
- Motor > Language contrast asymmetry: 12x — the most extreme directional asymmetry
- Suggests non-linguistic auditory input actively inhibits language networks

**3. Embodied Simulation**
- Motor Cortex is the only ROI with positive grand mean (+0.084) across all 6 stimuli
- Appears in top 3 ROIs for 4/6 stimuli (speech, action, music)
- Ronaldo speech: Motor Cortex = #1 ROI (+0.207), not Auditory — sports content overrides speech
- Consistent with mirror neuron literature and motor theory of speech perception

### Temporal Dynamics (from 90 brain surface plots)

| Stimulus Type | Temporal Pattern | Key Feature |
|--------------|-----------------|-------------|
| **Speech** | Inverted-U (escalate → peak → partial return) | Peak at mid-clip, left-lateralized |
| **Action visual** (Muay Thai) | Sustained escalation | Most intense cortical activation in dataset |
| **Passive visual** (Romantic) | Rapid habituation | Near-total flattening by t=8 |
| **Music** | Progressive deepening suppression | Colorbar minimum doubles (−1.0 → −2.0) |
| **Food visual** | Sustained with V1 hotspot | Unique calcarine fissure activation |

### Neural Clustering

Three clusters emerged from the similarity matrix:
- **Cluster A (Speech)**: Jack Ma ↔ Ronaldo = 0.801. Driven by shared Auditory Cortex + Visual Cortex suppression.
- **Cluster B (Active Visual)**: Muay Thai ↔ Food Reel = 0.554. Shared Visual/Parietal/Face-Selective profile.
- **Bridge node**: Ronaldo connects to everyone (>0.2 similarity to all except Food Reel) — uniquely "central" stimulus.

### Statistical Limitations You Must Know

| Parameter | Value | Implication |
|-----------|-------|-------------|
| Stimuli | 6 | Critically underpowered |
| RDM pairs | C(6,2) = 15 | Below threshold for robust RSA |
| Permutation space | 6! = 720 | Exact Mantel test feasible |
| Min RSA for p<0.05 | ~0.45-0.50 | CLIP (0.386) is marginal significance |
| Recommended n | 20+ | For detecting effects of r = 0.3 |

**Key talking point**: "Our results are hypothesis-generating, not confirmatory. CLIP's 0.386 is at the boundary of significance with 6 stimuli. The next step is expanding to 20+ stimuli and running permutation testing for exact p-values."

---

## 11. Advanced Interview Questions (from Findings)

### Q9: "You found CLIP beats Whisper by 13x. But CLIP only sees one frame while Whisper hears the full audio. Isn't that an unfair comparison?"

**Answer:**

"Good catch — and this is actually one of the most interesting findings. There are three explanations:

1. **Stimulus design bias**: 3 of 6 stimuli are silent, so Whisper gets essentially zero signal from half the dataset. A balanced set with equal speech/music/silence would be fairer.

2. **Cortical real estate**: ~30% of human cortex is visual, ~8% auditory. Our RSA uses 9 whole-brain ROIs, so it naturally favors the modality with the most cortical representation. A region-specific RSA (CLIP vs Visual Cortex only, Whisper vs Auditory Cortex only) would be more informative.

3. **CLIP's semantic richness**: CLIP isn't just a pixel encoder — it's trained on 400M image-text pairs, so its visual features capture high-level semantics (action, scene, identity) that map onto brain categorical representations. A single semantically-rich frame may capture the 'gist' as well as 15 seconds of audio.

The fix is to (a) balance the stimulus set, (b) compute region-specific RSA, and (c) add multimodal models like ImageBind that can use both modalities. I'd also try multiple CLIP frames (average 5-10 embeddings) to capture more temporal dynamics."

---

### Q10: "GPT-2 anti-correlates with the brain. What does that actually mean, and is it just noise?"

**Answer:**

"GPT-2's RSA of -0.143 means its similarity structure is *inverted* relative to the brain's. Stimuli GPT-2 considers similar, the brain finds dissimilar.

Concretely: GPT-2 receives 'Jack Ma Motivation, Speech' and 'Ronaldo Motivation, Speech' — it embeds these as very similar (both motivational speech). But the brain data shows Jack Ma is auditory-linguistic dominant while Ronaldo is motor-somatosensory dominant. They actually have quite different brain profiles despite similar text descriptions.

Is it noise? With only 6 stimuli and 15 pairs, -0.143 is non-significant (p >> 0.05). The magnitude isn't reliable. But the *direction* (negative) is consistent with published literature — Schrimpf et al. (2021) showed language models predict language cortex specifically but fail at sensory cortices. Our 9-ROI aggregate is dominated by sensory regions, so a language model anti-correlating is theoretically expected.

The interesting experiment would be: compute RSA for GPT-2 vs Language Areas *only*. I predict it would flip positive — GPT-2's text embeddings should align with how the brain's language network organizes stimuli, even if they fail at the whole-brain level."

---

### Q11: "You found Muay Thai and Romantic Couple have 0.001 similarity despite being the same category. What's the lesson for production ML systems?"

**Answer:**

"The lesson is that **human-assigned labels are lossy abstractions of high-dimensional data**. Both are tagged 'Silence + Visuals' but engage completely different brain networks — Muay Thai activates parietal/motor (spatial awareness, action observation) while Romantic Couple activates almost nothing (weak, diffuse response).

This maps directly to production problems:
1. **Content recommendation**: If you cluster by metadata tags, you'll group fundamentally different content. Users who like action martial arts clips won't enjoy slow romantic scenes just because both are 'visual content.'
2. **Embedding-based retrieval beats tag-based filtering**: The 0.001 similarity means neural embeddings correctly separate these, while category labels fail.
3. **Eval dataset design**: If you're building benchmarks, don't assume samples from the same category are interchangeable. Two 'customer complaint' emails might have totally different semantic profiles.

The broader principle: **always validate category assumptions with representational similarity**, whether you're building recommendation systems, eval datasets, or content moderation pipelines."

---

### Q12: "Motor Cortex is positive for all stimuli — even speech and music. How would you explain this to a non-technical stakeholder?"

**Answer:**

"Imagine watching any video. Even if you're sitting still, your brain's motor system is subtly activated — it's simulating the actions you're seeing. When you watch a martial arts kick, your motor cortex fires as if you might kick. When you hear speech, your motor cortex activates the circuits for producing speech — as if you're about to speak. When you hear music, your motor cortex responds to the beat — you feel the urge to tap or move.

This is called embodied simulation, and our data shows it's the most universal brain response to video content. Out of 9 brain regions measured, Motor Cortex is the only one that's positive across all 6 stimuli — everything from motivational speeches to food reels to instrumental music.

For a product team, this means: **all video content is inherently 'active' from the brain's perspective**. There's no such thing as purely passive video consumption. This has implications for engagement metrics, ad effectiveness, and content accessibility design."

---

### Q13: "You generated 90 brain plots, 78 radar charts, and 85 JSONs. How did you analyze all of that systematically?"

**Answer:**

"I built an automated results generator (`generate_all_results.py`) that runs all three modules across all parameter combinations — 6 stimuli × 3 timepoints × 5 views for brain plots, all pairwise stimulus comparisons for similarity, all 72 directed ROI contrast pairs, and all model pairwise comparisons for eval.

For analysis, I used a multi-agent parallel approach: three specialized analysis agents ran simultaneously:
1. A **neuroimaging specialist** examined all 90 brain surface plots for spatial patterns, temporal dynamics, and hemispheric asymmetry
2. A **similarity specialist** analyzed all 78 radar charts and match data for clustering patterns and contrast extremes
3. An **eval specialist** assessed the leaderboard against published RSA benchmarks and computed statistical power requirements

Each agent produced a structured report, and I synthesized the cross-module findings (Visual Dominance, Language Suppression, Embodied Simulation) from their overlapping evidence.

The production parallel: this is like running multiple monitoring dashboards in an observability system — each agent watches a different signal dimension, and the synthesis layer correlates alerts across agents."

---

### Q14: "Your CLIP score of 0.386 is at the boundary of statistical significance. How do you communicate uncertain results to stakeholders?"

**Answer:**

"I frame it as 'hypothesis-generating, not confirmatory.' Specifically:

1. **Lead with what we know for sure**: The ranking CLIP > Whisper > GPT-2 is consistent with published neuroscience (Yamins et al., Kriegeskorte et al., Schrimpf et al.). The direction is reliable.

2. **Be transparent about what we don't know**: With 6 stimuli, we can't confirm the magnitude. CLIP's 0.386 needs 20+ stimuli to reach robust significance. I'd show the power analysis table: n=6 gives 15 pairs (underpowered), n=20 gives 190 pairs (sufficient).

3. **Propose the concrete next step**: 'We need 14 more stimuli to confirm this. Here's the balanced design: 5 speech, 5 music, 5 silent-action, 5 silent-social. Estimated cost: 2 hours on Colab Pro.'

The meta-skill here is knowing when your data supports a conclusion vs. when it only supports a hypothesis. In production, this translates to: don't ship a model decision based on an A/B test that hasn't reached statistical power — extend the test, don't over-interpret early signal."

---

## 12. What to Study Next

### Based on Gaps This Project Exposed

| Gap | What to Study | Resource |
|-----|--------------|----------|
| RSA is marginal with n=6 — need statistical rigor | Permutation testing, Mantel test, bootstrap CIs | Kriegeskorte et al. 2008, `scipy.stats.permutation_test` |
| CLIP wins but is it the visual ROIs specifically? | Region-specific RSA, ROI-level alignment maps | Khaligh-Razavi & Kriegeskorte 2014 |
| Implemented RSA but not CKA or Procrustes | Representational comparison methods | Kornblith et al. 2019 (CKA paper) |
| No temporal RSA — averaged across time | Sliding window RSA, temporal generalization | Cichy et al. 2014 |
| No video-native models tested | VideoMAE, InternVideo, TimeSformer | VideoMAE paper (He et al. 2022) |
| No multimodal models tested | ImageBind, AudioCLIP — should beat single-modality | Girdhar et al. 2023 (ImageBind) |
| GPT-2 used stimulus titles, not actual transcriptions | Fair text model comparison | Use Whisper transcription as GPT input |
| Cache is file-based, not a real feature store | Feature store architecture | Feast docs, Tecton blog |
| No observability in the pipeline | OpenTelemetry for ML | AgentScope's OTel integration |
| No CI/CD or automated testing on push | MLOps pipeline | GitHub Actions + pytest |
| No model versioning in cache | ML artifact management | MLflow, DVC |

### Priority Study Path (aligned with job switch goals)

1. **Permutation testing** — Implement Mantel test for exact p-values on current data (can do immediately)
2. **Region-specific RSA** — Compute per-ROI alignment maps (can do immediately, high interview value)
3. **AgentScope** — Learn agent evals (ACEBench), OpenTelemetry observability, K8s deployment
4. **System design practice** — Design a "brain encoding API at scale" using the patterns from this project
5. **Interview prep** — Practice explaining the 5 headline findings, statistical limitations, and next steps verbally
6. **Expand stimulus set** — 20+ stimuli with balanced categories for robust RSA
