# NeuroLens — Design Spec

> Interactive Colab notebook that flips TRIBE v2 into a neuroscience playground: predict brain responses, discover brain-matched content, and benchmark AI models against biological neural responses.

**Date:** 2026-03-29
**Status:** Approved
**Author:** Ansuman SS Bhujabala

---

## 1. Overview

NeuroLens is built on top of Meta's TRIBE v2 — a multimodal brain encoding model that predicts fMRI responses from video, audio, and text. NeuroLens wraps TRIBE v2 into three interactive modules inside a Colab notebook:

1. **PREDICT** — Pick a stimulus, see predicted brain activation on a cortical map
2. **MATCH** — Define a desired brain state, find content that triggers it
3. **EVAL** — Compare how closely different AI models' representations align with biological brain responses

All heavy computation is pre-computed and cached. The main notebook runs on CPU.

---

## 2. Architecture

```
+---------------------------------------------+
|              NeuroLens Notebook              |
+-------------+--------------+----------------+
|  Module 1   |  Module 2    |  Module 3      |
|  PREDICT    |  MATCH       |  EVAL          |
|  stimulus > |  brain state |  AI model vs   |
|  brain map  |  > content   |  brain align.  |
+-------------+--------------+----------------+
|            Shared Data Layer                 |
|  - Pre-computed brain predictions cache      |
|  - Stimulus library (clips metadata)         |
|  - AI model embeddings cache                 |
|  - fsaverage5 cortical mesh + ROI labels     |
+---------------------------------------------+
```

### Compute strategy

- **Precompute notebook** (`neurolens_precompute.ipynb`): Runs once on Colab GPU. Downloads TRIBE v2 and comparison models, processes the stimulus library, generates all caches.
- **Main notebook** (`neurolens.ipynb`): Loads caches, runs on CPU. All interactivity is lightweight lookups, cosine similarities, and visualization.

### Deliverables

| File | Purpose |
|------|---------|
| `neurolens.ipynb` | Main interactive experience (CPU) |
| `neurolens_precompute.ipynb` | One-time cache generation (GPU) |

---

## 3. Stimulus Library

~50-80 short clips (5-15 seconds each) from public domain / Creative Commons sources (Pexels, Pixabay, LibriVox). Any video/audio/text content is valid.

### Categories

| Category | Examples | Neuroscience rationale |
|----------|----------|----------------------|
| Speech | TED talk excerpt, podcast clip, audiobook | Language processing regions |
| Music | Classical, hip-hop, ambient, vocals-only | Auditory cortex variations |
| Silence + Visuals | Nature timelapse, abstract art, faces | Pure visual processing |
| Emotional | Horror scene, comedy bit, heartwarming moment | Limbic system activation |
| Multimodal-rich | Movie scene with dialogue + action + music | Full brain engagement |
| Text-only | Narrated story, poetry reading | Language network isolation |

### Per-stimulus cache

For each stimulus, the precompute notebook generates:
- TRIBE v2 brain predictions (~20k vertices per timestep) as `.npz`
- Per-ROI activation summaries (average activation per brain region) as `.json`
- Feature embeddings from each model as `.pt` files

---

## 4. Cache Structure

Hosted on HuggingFace Hub or Google Drive. Downloaded once on first run.

```
neurolens_cache/
+-- stimuli/                   -- metadata.json (clip info, categories, paths)
+-- brain_preds/               -- {stimulus_id}.npz (TRIBE v2 predictions)
+-- roi_summaries/             -- {stimulus_id}.json (per-ROI averages)
+-- embeddings/
|   +-- vjepa2/                -- {stimulus_id}.pt
|   +-- llama/                 -- {stimulus_id}.pt
|   +-- wav2vec/               -- {stimulus_id}.pt
|   +-- dinov2/                -- {stimulus_id}.pt
|   +-- clip/                  -- {stimulus_id}.pt
|   +-- whisper/               -- {stimulus_id}.pt
+-- mesh/                      -- fsaverage5 surface + ROI labels
```

---

## 5. Module 1 — PREDICT

**Input:** User picks a stimulus from a dropdown (library mode only, no live upload).

**Processing:** Load cached brain predictions for selected stimulus.

**Visualization:**
- 2D cortical flatmap (nilearn) with heatmap overlay showing activation intensity
- Sidebar: top-5 most activated ROIs with anatomical labels (e.g., "V1 — Primary Visual Cortex", "STG — Superior Temporal Gyrus")
- Time slider: scrub through clip timeline, watch activation change frame by frame

**Interactive controls:**
- Modality toggle: switch between video-only, audio-only, text-only, and all-combined predictions to see each modality's contribution
- Compare mode: side-by-side brain maps for two different stimuli

---

## 6. Module 2 — MATCH

**Purpose:** Given a desired brain activation pattern, find stimuli that best trigger it.

### Three input modes

**A) Region picker:**
- Dropdown to select brain regions (e.g., "Broca's Area", "Auditory Cortex", "Fusiform Face Area")
- Intensity selector: low / medium / max per region
- System builds a target activation vector from constraints

**B) "More like this":**
- User picks a stimulus from Module 1
- Its brain prediction becomes the target vector
- System ranks all other stimuli by neural similarity (cosine similarity on activation vectors)
- Returns top-N matches with similarity scores

**C) Contrast mode:**
- User picks a "maximize" region and a "minimize" region
- System ranks stimuli by activation difference (max_region - min_region)
- Answers questions like: "What activates visual cortex BUT NOT auditory cortex?"

### Output
- Ranked list of matching stimuli with similarity scores
- Side-by-side brain maps: target pattern vs. best match
- Radar chart: activation profile across major ROIs for target and matches

---

## 7. Module 3 — EVAL

**Purpose:** Benchmark AI models against biological brain responses using Representational Similarity Analysis (RSA).

### Models compared

**TRIBE v2 internal encoders (already cached):**
- V-JEPA2 (video)
- LLaMA 3.2 (text)
- Wav2Vec-BERT (audio)
- DINOv2 (image)

**External comparison models (pre-computed in precompute notebook):**
- CLIP (vision-language)
- Whisper (audio)
- GPT-2 (text)

### Methodology

For each model:
1. Compute pairwise similarity matrix across all stimuli in embedding space
2. Compute pairwise similarity matrix across all stimuli in brain prediction space
3. RSA score = correlation between the two matrices
4. Per-ROI alignment: repeat using brain predictions restricted to specific regions

### Interactive elements

- **Leaderboard table:** Models ranked by overall brain alignment score, sortable by modality
- **Radar chart:** Pick any two models, compare alignment across brain regions (visual cortex, auditory cortex, language areas, motor cortex, etc.)
- **Stimulus drill-down:** Pick a clip, see which model's embedding was closest to the brain response
- **"Brain report card":** Summary card per model (e.g., "CLIP: 73% brain-aligned in visual areas, 12% in auditory areas")

---

## 8. Notebook Structure

```
neurolens.ipynb
|
+-- 0. Setup & Install        -- pip installs, download caches, imports
+-- 1. About NeuroLens        -- Markdown intro, what this is, how to use
+-- 2. Module: PREDICT         -- Library picker, brain visualization
+-- 3. Module: MATCH           -- Region picker / more like this / contrast
+-- 4. Module: EVAL            -- Model leaderboard, radar charts, drill-down
+-- 5. Explore Further         -- Links to paper, ideas for extending
```

### Dependencies

| Library | Purpose |
|---------|---------|
| `ipywidgets` | Dropdowns, sliders, toggles |
| `matplotlib` | Brain plots base |
| `nilearn` | 2D cortical flatmaps |
| `plotly` | Radar charts, interactive comparisons |
| `numpy`, `scipy` | Similarity computations, RSA |
| `torch` | Loading cached embeddings |
| `tribev2` | Brain predictions (precompute only) |

---

## 9. Scope Boundaries

**In scope:**
- Pre-computed stimulus library (~50-80 clips)
- Three interactive modules as described
- Precompute notebook for cache generation
- All visualization in-notebook (no external server)

**Out of scope:**
- Live video upload/inference in main notebook
- Fine-tuning any models
- Subject-specific predictions (uses "average" subject only)
- Subcortical regions (cortical surface only for v1)
- External web app or API

---

## 10. Learning Outcomes

| Growth area | What NeuroLens teaches |
|-------------|----------------------|
| System Design | Cache architecture, compute/serving separation, data pipeline design |
| Evals | RSA methodology, building benchmarks, alignment scoring (transferable to LLM evals) |
| AIOps thinking | Pre-compute vs. real-time tradeoffs, resource-aware architecture |
| Multimodal AI | Hands-on with LLaMA, V-JEPA2, Wav2Vec-BERT, CLIP, DINOv2, Whisper |
| Visualization | Interactive brain maps, radar charts, data storytelling |
