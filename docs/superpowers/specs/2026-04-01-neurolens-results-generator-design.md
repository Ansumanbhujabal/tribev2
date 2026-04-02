# NeuroLens Automated Results Generator — Design Spec

**Date**: 2026-04-01
**Goal**: Generate all meaningful results from the 3 NeuroLens modules for offline analysis, reporting, and insight extraction.

---

## Scope

A single Python script (`neurolens/generate_all_results.py`) that reads the pre-computed `neurolens_cache/` and produces a structured output directory with images and data files.

**Runs on CPU only** — no GPU needed. Just reads cache and generates plots/data.

---

## What Gets Generated

### Module 1: Predict

**Brain surface plots**: 6 stimuli x 3 key frames x 5 views = **90 PNGs**

- **Key frames per stimulus**: first (t=0), middle (t=n//2), last (t=n-1)
- **Views**: left, right, medial_left, medial_right, dorsal
- **Colormap**: "hot" (matching the interactive notebook)

**ROI summaries**: 6 JSON files (one per stimulus)
- Time-averaged top ROIs with activation scores
- All 9 ROI groups with values

### Module 2: Match

**More Like This**: 6 runs (one source stimulus each)
- Ranked matches JSON (top 5 similar stimuli + cosine similarity scores)
- Radar chart PNG comparing top 3 activation profiles

**Contrast**: 72 directed ROI pairs (9 x 8, maximize vs minimize)
- Ranked matches JSON (top 5 stimuli + contrast scores)
- Radar chart PNG for top 3 matches
- Total: 72 JSONs + 72 radar PNGs

### Module 3: Eval

**Leaderboard**: 1 run
- JSON with all 3 model RSA scores ranked
- Bar chart PNG

**Model Comparison**: 3 pairwise runs (CLIP/Whisper, CLIP/GPT-2, Whisper/GPT-2)
- JSON per pair with individual RSA scores + brain alignment percentages

### Master Index

`summary.json` at output root:
- Generation timestamp
- Stimulus metadata
- File counts per module
- Paths to all generated files

---

## Output Structure

```
neurolens_results/
├── predict/
│   ├── clip_001/
│   │   ├── brain_t00_left.png
│   │   ├── brain_t00_right.png
│   │   ├── brain_t00_medial_left.png
│   │   ├── brain_t00_medial_right.png
│   │   ├── brain_t00_dorsal.png
│   │   ├── brain_t05_left.png
│   │   ├── ... (middle + last frames)
│   │   └── roi_summary.json
│   ├── clip_002/
│   │   └── ...
│   └── ... (clip_003 through clip_006)
├── match/
│   ├── more_like_this/
│   │   ├── clip_001_matches.json
│   │   ├── clip_001_radar.png
│   │   └── ... (all 6 stimuli)
│   └── contrast/
│       ├── max_Visual_Cortex_min_Auditory_Cortex.json
│       ├── max_Visual_Cortex_min_Auditory_Cortex_radar.png
│       └── ... (all 72 directed pairs)
├── eval/
│   ├── leaderboard.json
│   ├── leaderboard.png
│   ├── compare_CLIP_vs_Whisper.json
│   ├── compare_CLIP_vs_GPT2.json
│   └── compare_Whisper_vs_GPT2.json
└── summary.json
```

---

## Technical Approach

- Uses existing `neurolens/` modules: `predict.py`, `match.py`, `eval.py`, `viz.py`, `roi.py`, `cache.py`, `stimulus.py`
- Matplotlib `Agg` backend for headless rendering
- Progress bars via tqdm for each module
- Script is importable (functions) and runnable (`python -m neurolens.generate_all_results`)
- Output directory configurable via CLI arg (default: `neurolens_results/`)

---

## Estimated Output

| Type | Count |
|------|-------|
| Brain surface PNGs | 90 |
| Radar chart PNGs | 78 |
| Bar chart PNGs | 1 |
| JSON data files | 85 |
| Master index | 1 |
| **Total files** | **~255** |
