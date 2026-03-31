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
7. [Neuroscience Concepts](#7-neuroscience-concepts)
8. [Common Follow-Up Questions](#8-common-follow-up-questions)
9. [What to Study Next](#9-what-to-study-next)

---

## 1. Project Summary

### 30-Second Pitch

"I built NeuroLens — an interactive neuroscience playground on top of Meta's TRIBE v2 brain encoding model. It has three modules: Predict (visualize predicted brain activation from any video/audio/text), Match (find content that triggers specific brain states using cosine similarity search), and Eval (benchmark how closely AI models like CLIP, Whisper, and GPT-2 represent information compared to the human brain, using Representational Similarity Analysis). I designed the system with a compute/serving separation — heavy inference is pre-computed on GPU, and the interactive notebook runs on CPU with sub-second responses."

### 60-Second Version (add these details)

"The Eval module uses RSA — Representational Similarity Analysis — which compares the *geometry* of how stimuli relate to each other in AI embedding space versus brain activation space. Instead of comparing vectors directly (which is impossible when brain space is 20,484 dimensions and CLIP is 512), RSA compares pairwise similarity matrices using Spearman rank correlation. This gives a single alignment score per model.

I implemented the full pipeline: 7 Python modules with 31 tests (TDD), two Colab notebooks (precompute + interactive), and a cache architecture using .npz for brain predictions, .pt for model embeddings, and .json for ROI summaries — mirroring the three-tier feature store pattern used in production ML systems."

---

## 2. Technical Learnings

### What You Built (and can talk about)

| Area | What You Did | Interview Signal |
|------|-------------|-----------------|
| **System Design** | Compute/serving separation — GPU precompute, CPU serving | Production ML architecture |
| **AI Evaluation** | RSA implementation (Spearman on similarity matrices) | Eval methodology, not just "run benchmarks" |
| **Multimodal AI** | Worked with 7 models (LLaMA 3.2, V-JEPA2, Wav2Vec-BERT, DINOv2, CLIP, Whisper, GPT-2) | Breadth across modalities |
| **Cache Architecture** | Three-format cache (.npz/.pt/.json) mirroring feature store patterns | Data engineering, storage design |
| **TDD** | 31 tests across 8 files, test-first development | Engineering discipline |
| **Environment Management** | Solved numpy conflicts, moviepy v1/v2 incompatibility, Colab deployment | Real-world DevOps/MLOps |
| **Brain Visualization** | nilearn cortical surface plots on fsaverage5 mesh | Domain-specific visualization |
| **Similarity Search** | Cosine similarity matching, contrast-based ranking | Recommendation/retrieval systems |

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

## 7. Neuroscience Concepts

### fMRI and BOLD Signal

fMRI measures Blood Oxygen Level Dependent (BOLD) signal — a proxy for neural activity. When neurons fire, local blood flow increases, changing the MRI signal. Temporal resolution: ~1-2 seconds (TR = repetition time).

### fsaverage5 (20,484 vertices)

The human cortex is a folded sheet. To compare across subjects, data is projected onto a standard template surface — **fsaverage**. fsaverage5 is the 5th-level icosphere subdivision:

```
Level 0:    12 vertices (icosahedron)
Level 5: 10,242 vertices per hemisphere × 2 = 20,484 total
```

Each vertex ≈ 1mm² of cortical surface. ~4mm effective spatial resolution.

### HCP MMP1.0 Atlas (180 regions per hemisphere)

The Human Connectome Project Multi-Modal Parcellation (Glasser et al., 2016, Nature) divides the cortex into 360 regions using myelin content, cortical thickness, functional connectivity, and topographic organization.

In NeuroLens, friendly names map to HCP regions:
```python
ROI_GROUPS = {
    "Visual Cortex": ["V1", "V2", "V3", "V4"],
    "Auditory Cortex": ["A1", "A4", "A5", "RI", "MBelt", "LBelt", "PBelt"],
    "Language Areas": ["44", "45", "47l", ...],  # Broca's area
    ...
}
```

### What the ROI Scores Mean

From the actual Muay Thai Kick clip:
```
Parietal Cortex........... +0.152   ← spatial awareness, body movement
Face-Selective Areas...... +0.131   ← face/person detection
Language Areas............ +0.105   ← internal narration?
Motor Cortex.............. +0.101   ← mirror neurons for observed action
Visual Cortex............. +0.099   ← visual processing
```

Parietal cortex highest for a martial arts kick makes neuroscientific sense — it handles spatial awareness and body movement perception.

---

## 8. Common Follow-Up Questions

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

## 9. What to Study Next

### Based on Gaps This Project Exposed

| Gap | What to Study | Resource |
|-----|--------------|----------|
| You implemented RSA but didn't deeply understand CKA or Procrustes | Representational comparison methods | Kornblith et al. 2019 (CKA paper) |
| Cache is file-based, not a real feature store | Feature store architecture | Feast docs, Tecton blog |
| No observability in the pipeline | OpenTelemetry for ML | AgentScope's OTel integration |
| No CI/CD or automated testing on push | MLOps pipeline | GitHub Actions + pytest |
| RSA at one time point, no temporal analysis | Temporal RSA, sliding window RSA | Cichy et al. 2014 |
| No model versioning in cache | ML artifact management | MLflow, DVC |
| Single-machine, no distributed compute | Distributed inference | Ray, Dask, K8s Jobs |

### Priority Study Path (aligned with job switch goals)

1. **AgentScope** — Learn agent evals (ACEBench), OpenTelemetry observability, K8s deployment
2. **Feature store patterns** — Feast tutorial, understand offline/online/streaming tiers
3. **System design practice** — Design a "brain encoding API at scale" using the patterns from this project
4. **Interview prep** — Practice explaining RSA, compute/serving separation, and cache architecture verbally
