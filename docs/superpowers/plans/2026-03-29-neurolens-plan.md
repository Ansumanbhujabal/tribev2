# NeuroLens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two Colab notebooks — a precompute notebook that caches TRIBE v2 predictions and model embeddings for a stimulus library, and a main interactive notebook with three modules (Predict, Match, Eval).

**Architecture:** Pre-compute all heavy inference (TRIBE v2 + comparison models) into `.npz`/`.pt` cache files. The main notebook loads caches and runs lightweight CPU operations (cosine similarity, RSA, visualization). All interactivity uses `ipywidgets`. Brain visualization uses `nilearn`'s `plot_surf_stat_map` via TRIBE v2's existing `PlotBrainNilearn` class.

**Tech Stack:** Python 3.10+, PyTorch, TRIBE v2, nilearn, matplotlib, plotly, ipywidgets, numpy, scipy, transformers (CLIP, Whisper, GPT-2)

---

## File Structure

```
neurolens/                         # New top-level package (sibling to tribev2/)
├── __init__.py                    # Package marker
├── cache.py                       # CacheManager: load/save cached data
├── stimulus.py                    # StimulusLibrary: metadata, lookup
├── predict.py                     # PredictModule: brain visualization widgets
├── match.py                       # MatchModule: brain-state matching widgets
├── eval.py                        # EvalModule: RSA computation + visualization
├── roi.py                         # ROI utilities: named regions, groupings
└── viz.py                         # Shared visualization helpers (brain plots, radar)

neurolens_precompute.ipynb         # One-time GPU notebook
neurolens.ipynb                    # Main interactive CPU notebook

tests/
├── test_cache.py
├── test_stimulus.py
├── test_predict.py
├── test_match.py
├── test_eval.py
├── test_roi.py
└── test_viz.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `neurolens/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create neurolens package**

```python
# neurolens/__init__.py
"""NeuroLens: Interactive neuroscience playground built on TRIBE v2."""
```

```python
# tests/__init__.py
```

- [ ] **Step 2: Verify import works**

Run: `python -c "import neurolens; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add neurolens/__init__.py tests/__init__.py
git commit -m "feat: scaffold neurolens package and tests directory"
```

---

## Task 2: ROI Utilities

**Files:**
- Create: `neurolens/roi.py`
- Test: `tests/test_roi.py`

The TRIBE v2 codebase has `get_hcp_labels()` and `summarize_by_roi()` in `tribev2/utils.py`. We need a mapping from human-friendly ROI group names (like "Visual Cortex", "Auditory Cortex") to HCP atlas region names. This avoids users needing to know HCP nomenclature.

- [ ] **Step 1: Write failing test**

```python
# tests/test_roi.py
import numpy as np
from neurolens.roi import ROI_GROUPS, get_roi_group_names, summarize_by_roi_group


def test_roi_groups_are_nonempty():
    assert len(ROI_GROUPS) > 0
    for name, regions in ROI_GROUPS.items():
        assert isinstance(regions, list)
        assert len(regions) > 0


def test_get_roi_group_names():
    names = get_roi_group_names()
    assert isinstance(names, list)
    assert "Visual Cortex" in names
    assert "Auditory Cortex" in names
    assert "Language Areas" in names


def test_summarize_by_roi_group():
    # 20484 vertices = fsaverage5 (10242 per hemisphere * 2)
    fake_data = np.random.randn(20484)
    result = summarize_by_roi_group(fake_data)
    assert isinstance(result, dict)
    assert "Visual Cortex" in result
    for name, value in result.items():
        assert isinstance(value, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_roi.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write implementation**

```python
# neurolens/roi.py
"""Human-friendly ROI groups mapped to HCP atlas regions."""

import numpy as np

# Maps friendly names to lists of HCP MMP1.0 region name prefixes.
# We use wildcard suffixes so get_hcp_roi_indices(name + "*") catches
# all sub-regions (e.g. "V1" matches "V1-lh", "V1-rh").
ROI_GROUPS: dict[str, list[str]] = {
    "Visual Cortex": ["V1", "V2", "V3", "V4"],
    "Auditory Cortex": ["A1", "A4", "A5", "RI", "MBelt", "LBelt", "PBelt"],
    "Language Areas": ["44", "45", "47l", "IFSa", "IFSp", "IFJa", "IFJp",
                       "STSda", "STSdp", "STSva", "STSvp", "STV",
                       "TPOJ1", "TPOJ2", "TPOJ3"],
    "Motor Cortex": ["4", "3a", "3b", "1", "2"],
    "Prefrontal Cortex": ["8Av", "8Ad", "8BL", "8C", "9a", "9p", "9m",
                          "10d", "10r", "10v", "46", "p9-46v", "a9-46v"],
    "Temporal Cortex": ["TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
                        "TGd", "TGv", "TF"],
    "Parietal Cortex": ["7AL", "7Am", "7PC", "7PL", "7Pm",
                        "AIP", "IP0", "IP1", "IP2", "LIPd", "LIPv",
                        "MIP", "VIP"],
    "Somatosensory Cortex": ["3a", "3b", "1", "2"],
    "Face-Selective Areas": ["FFC", "OFA", "PeEc"],
}


def get_roi_group_names() -> list[str]:
    """Return sorted list of all ROI group names."""
    return sorted(ROI_GROUPS.keys())


def summarize_by_roi_group(
    data: np.ndarray, mesh: str = "fsaverage5"
) -> dict[str, float]:
    """Compute mean activation per ROI group.

    Parameters
    ----------
    data : np.ndarray
        1D array of shape (n_vertices,) on fsaverage5 (20484 vertices).
    mesh : str
        Mesh resolution name.

    Returns
    -------
    dict mapping ROI group name to mean activation (float).
    """
    from tribev2.utils import get_hcp_roi_indices

    result = {}
    for group_name, regions in ROI_GROUPS.items():
        all_indices = []
        for region in regions:
            try:
                indices = get_hcp_roi_indices(region, hemi="both", mesh=mesh)
                all_indices.append(indices)
            except ValueError:
                # Region not found in atlas — skip it
                continue
        if all_indices:
            combined = np.concatenate(all_indices)
            # Deduplicate indices
            combined = np.unique(combined)
            result[group_name] = float(data[combined].mean())
        else:
            result[group_name] = 0.0
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_roi.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/roi.py tests/test_roi.py
git commit -m "feat: add ROI group utilities mapping HCP regions to friendly names"
```

---

## Task 3: Stimulus Library

**Files:**
- Create: `neurolens/stimulus.py`
- Test: `tests/test_stimulus.py`

Manages stimulus metadata: each stimulus has an id, name, category, media type, and file path. The library is loaded from a `metadata.json` file in the cache directory.

- [ ] **Step 1: Write failing test**

```python
# tests/test_stimulus.py
import json
import tempfile
from pathlib import Path

from neurolens.stimulus import Stimulus, StimulusLibrary


def _make_metadata(tmp: Path) -> Path:
    stimuli = [
        {
            "id": "clip_001",
            "name": "Nature timelapse",
            "category": "Silence + Visuals",
            "media_type": "video",
            "duration_sec": 10.0,
        },
        {
            "id": "clip_002",
            "name": "TED talk excerpt",
            "category": "Speech",
            "media_type": "video",
            "duration_sec": 12.0,
        },
        {
            "id": "clip_003",
            "name": "Classical music",
            "category": "Music",
            "media_type": "audio",
            "duration_sec": 15.0,
        },
    ]
    meta_path = tmp / "stimuli" / "metadata.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(stimuli))
    return tmp


def test_stimulus_dataclass():
    s = Stimulus(
        id="clip_001",
        name="Nature timelapse",
        category="Silence + Visuals",
        media_type="video",
        duration_sec=10.0,
    )
    assert s.id == "clip_001"
    assert s.category == "Silence + Visuals"


def test_library_load():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        assert len(lib) == 3
        assert lib.get("clip_001").name == "Nature timelapse"


def test_library_filter_by_category():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        music = lib.filter_by_category("Music")
        assert len(music) == 1
        assert music[0].id == "clip_003"


def test_library_categories():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        cats = lib.categories()
        assert set(cats) == {"Silence + Visuals", "Speech", "Music"}


def test_library_get_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        assert lib.get("nonexistent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stimulus.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/stimulus.py
"""Stimulus library: metadata loading and lookup."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stimulus:
    id: str
    name: str
    category: str
    media_type: str  # "video", "audio", or "text"
    duration_sec: float


class StimulusLibrary:
    """Loads and queries stimulus metadata from a cache directory.

    Expects ``<cache_dir>/stimuli/metadata.json`` — a JSON array of objects
    with keys: id, name, category, media_type, duration_sec.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "stimuli" / "metadata.json"
        raw = json.loads(meta_path.read_text())
        self._stimuli = [Stimulus(**item) for item in raw]
        self._by_id = {s.id: s for s in self._stimuli}

    def __len__(self) -> int:
        return len(self._stimuli)

    def get(self, stimulus_id: str) -> Stimulus | None:
        """Return a Stimulus by id, or None if not found."""
        return self._by_id.get(stimulus_id)

    def all(self) -> list[Stimulus]:
        """Return all stimuli."""
        return list(self._stimuli)

    def filter_by_category(self, category: str) -> list[Stimulus]:
        """Return stimuli matching the given category."""
        return [s for s in self._stimuli if s.category == category]

    def categories(self) -> list[str]:
        """Return sorted list of unique categories."""
        return sorted(set(s.category for s in self._stimuli))

    def ids(self) -> list[str]:
        """Return list of all stimulus ids."""
        return [s.id for s in self._stimuli]

    def dropdown_options(self) -> list[tuple[str, str]]:
        """Return (display_label, id) pairs for ipywidgets Dropdown."""
        return [(f"{s.name} [{s.category}]", s.id) for s in self._stimuli]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_stimulus.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/stimulus.py tests/test_stimulus.py
git commit -m "feat: add StimulusLibrary for metadata loading and lookup"
```

---

## Task 4: Cache Manager

**Files:**
- Create: `neurolens/cache.py`
- Test: `tests/test_cache.py`

Loads pre-computed brain predictions (`.npz`), ROI summaries (`.json`), and model embeddings (`.pt`) from the cache directory.

- [ ] **Step 1: Write failing test**

```python
# tests/test_cache.py
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from neurolens.cache import CacheManager


def _setup_cache(tmp: Path) -> Path:
    """Create a minimal cache structure."""
    # Brain predictions
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    np.savez(preds_dir / "clip_001.npz", preds=np.random.randn(5, 20484))

    # ROI summaries
    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()
    (roi_dir / "clip_001.json").write_text(
        json.dumps({"Visual Cortex": 0.5, "Auditory Cortex": 0.3})
    )

    # Embeddings
    for model_name in ["vjepa2", "clip"]:
        emb_dir = tmp / "embeddings" / model_name
        emb_dir.mkdir(parents=True)
        torch.save(torch.randn(256), emb_dir / "clip_001.pt")

    return tmp


def test_load_brain_preds():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        preds = cm.load_brain_preds("clip_001")
        assert preds.shape == (5, 20484)


def test_load_brain_preds_missing():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        assert cm.load_brain_preds("nonexistent") is None


def test_load_roi_summary():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        roi = cm.load_roi_summary("clip_001")
        assert roi["Visual Cortex"] == 0.5


def test_load_embedding():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        emb = cm.load_embedding("clip_001", "vjepa2")
        assert emb.shape == (256,)


def test_load_embedding_missing_model():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        assert cm.load_embedding("clip_001", "nonexistent") is None


def test_available_models():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        models = cm.available_models()
        assert set(models) == {"vjepa2", "clip"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/cache.py
"""CacheManager: load pre-computed brain predictions, ROI summaries, and embeddings."""

import json
from pathlib import Path

import numpy as np
import torch


class CacheManager:
    """Loads cached data from the NeuroLens cache directory.

    Expected layout::

        cache_dir/
        ├── brain_preds/{stimulus_id}.npz    (key: "preds")
        ├── roi_summaries/{stimulus_id}.json
        └── embeddings/{model_name}/{stimulus_id}.pt
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def load_brain_preds(self, stimulus_id: str) -> np.ndarray | None:
        """Load brain predictions array of shape (n_timesteps, n_vertices).

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "brain_preds" / f"{stimulus_id}.npz"
        if not path.exists():
            return None
        return np.load(path)["preds"]

    def load_roi_summary(self, stimulus_id: str) -> dict[str, float] | None:
        """Load per-ROI-group mean activations.

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "roi_summaries" / f"{stimulus_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_embedding(self, stimulus_id: str, model_name: str) -> torch.Tensor | None:
        """Load a model embedding tensor.

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "embeddings" / model_name / f"{stimulus_id}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def available_models(self) -> list[str]:
        """Return sorted list of model names that have cached embeddings."""
        emb_dir = self.cache_dir / "embeddings"
        if not emb_dir.exists():
            return []
        return sorted(d.name for d in emb_dir.iterdir() if d.is_dir())

    def all_brain_pred_ids(self) -> list[str]:
        """Return stimulus ids that have cached brain predictions."""
        preds_dir = self.cache_dir / "brain_preds"
        if not preds_dir.exists():
            return []
        return sorted(p.stem for p in preds_dir.glob("*.npz"))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cache.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/cache.py tests/test_cache.py
git commit -m "feat: add CacheManager for loading pre-computed data"
```

---

## Task 5: Shared Visualization Helpers

**Files:**
- Create: `neurolens/viz.py`
- Test: `tests/test_viz.py`

Wraps TRIBE v2's `PlotBrainNilearn` for brain surface plots and provides a helper to create plotly radar charts for ROI profiles.

- [ ] **Step 1: Write failing test**

```python
# tests/test_viz.py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

from neurolens.viz import plot_brain_surface, make_radar_chart


def test_plot_brain_surface_returns_figure():
    data = np.random.randn(20484)
    fig = plot_brain_surface(data, views=["left", "right"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_make_radar_chart_single():
    roi_data = {
        "Visual Cortex": 0.8,
        "Auditory Cortex": 0.3,
        "Language Areas": 0.6,
    }
    fig = make_radar_chart({"Stimulus A": roi_data})
    assert fig is not None


def test_make_radar_chart_comparison():
    data_a = {"Visual Cortex": 0.8, "Auditory Cortex": 0.3}
    data_b = {"Visual Cortex": 0.4, "Auditory Cortex": 0.7}
    fig = make_radar_chart({"A": data_a, "B": data_b})
    assert fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/viz.py
"""Shared visualization helpers: brain plots and radar charts."""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_brain_surface(
    data: np.ndarray,
    views: list[str] | None = None,
    cmap: str = "hot",
    title: str | None = None,
    colorbar: bool = True,
) -> matplotlib.figure.Figure:
    """Plot brain activation on a cortical surface using nilearn.

    Parameters
    ----------
    data : np.ndarray
        1D array of shape (n_vertices,) on fsaverage5 (20484).
    views : list of str
        View angles, e.g. ["left", "right"]. Defaults to ["left", "right"].
    cmap : str
        Matplotlib colormap name.
    title : str or None
        Optional figure title.
    colorbar : bool
        Whether to show a colorbar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from tribev2.plotting.cortical import PlotBrainNilearn

    if views is None:
        views = ["left", "right"]

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    fig, axarr = plotter.get_fig_axes(views)
    plotter.plot_surf(
        data,
        views=views,
        axes=axarr,
        cmap=cmap,
        colorbar=colorbar,
    )
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig


def make_radar_chart(
    datasets: dict[str, dict[str, float]],
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Create a radar/spider chart comparing ROI activation profiles.

    Parameters
    ----------
    datasets : dict
        Maps label → {roi_name: value}. All dicts must have the same keys.
    title : str or None
        Optional chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(next(iter(datasets.values())).keys())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    for (name, values), color in zip(datasets.items(), colors):
        vals = [values[label] for label in labels]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    return fig
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_viz.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/viz.py tests/test_viz.py
git commit -m "feat: add brain surface plot and radar chart visualization helpers"
```

---

## Task 6: Predict Module

**Files:**
- Create: `neurolens/predict.py`
- Test: `tests/test_predict.py`

Core logic for Module 1: load brain predictions for a stimulus and prepare data for visualization. Widget wiring is done in the notebook; this module handles data operations.

- [ ] **Step 1: Write failing test**

```python
# tests/test_predict.py
import json
import tempfile
from pathlib import Path

import numpy as np

from neurolens.cache import CacheManager
from neurolens.predict import get_prediction_at_time, get_top_rois, get_modality_contribution


def _setup(tmp: Path) -> CacheManager:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    # 5 timesteps, 20484 vertices
    preds = np.random.randn(5, 20484)
    np.savez(preds_dir / "clip_001.npz", preds=preds)

    # Also save per-modality predictions
    for mod in ["video", "audio", "text", "combined"]:
        np.savez(preds_dir / f"clip_001__{mod}.npz", preds=preds * np.random.rand())

    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()
    (roi_dir / "clip_001.json").write_text(
        json.dumps({"Visual Cortex": 0.9, "Auditory Cortex": 0.3, "Language Areas": 0.1})
    )
    return CacheManager(tmp)


def test_get_prediction_at_time():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_prediction_at_time(cm, "clip_001", time_idx=2)
        assert data.shape == (20484,)


def test_get_prediction_at_time_clamps():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_prediction_at_time(cm, "clip_001", time_idx=999)
        assert data.shape == (20484,)  # Should clamp to last timestep


def test_get_top_rois():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        top = get_top_rois(cm, "clip_001", k=2)
        assert len(top) == 2
        assert top[0][0] == "Visual Cortex"  # Highest activation first


def test_get_modality_contribution():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_modality_contribution(cm, "clip_001", modality="video", time_idx=0)
        assert data.shape == (20484,)


def test_get_modality_contribution_missing():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_modality_contribution(cm, "clip_001", modality="nonexistent", time_idx=0)
        assert data is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/predict.py
"""Predict module: load and slice brain predictions from cache."""

from __future__ import annotations

import numpy as np

from neurolens.cache import CacheManager


def get_prediction_at_time(
    cache: CacheManager,
    stimulus_id: str,
    time_idx: int,
) -> np.ndarray:
    """Return brain prediction at a specific timestep.

    Parameters
    ----------
    cache : CacheManager
    stimulus_id : str
    time_idx : int
        Timestep index. Clamped to valid range.

    Returns
    -------
    np.ndarray of shape (n_vertices,)
    """
    preds = cache.load_brain_preds(stimulus_id)
    time_idx = min(time_idx, preds.shape[0] - 1)
    time_idx = max(time_idx, 0)
    return preds[time_idx]


def get_num_timesteps(cache: CacheManager, stimulus_id: str) -> int:
    """Return total number of timesteps for a stimulus."""
    preds = cache.load_brain_preds(stimulus_id)
    return preds.shape[0]


def get_top_rois(
    cache: CacheManager,
    stimulus_id: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k ROI groups by mean activation, sorted descending.

    Returns list of (roi_name, mean_value) tuples.
    """
    roi_summary = cache.load_roi_summary(stimulus_id)
    sorted_rois = sorted(roi_summary.items(), key=lambda x: x[1], reverse=True)
    return sorted_rois[:k]


def get_modality_contribution(
    cache: CacheManager,
    stimulus_id: str,
    modality: str,
    time_idx: int,
) -> np.ndarray | None:
    """Return brain prediction for a specific modality at a timestep.

    Per-modality predictions are stored as ``{stimulus_id}__{modality}.npz``.
    Returns None if the modality file doesn't exist.
    """
    mod_id = f"{stimulus_id}__{modality}"
    preds = cache.load_brain_preds(mod_id)
    if preds is None:
        return None
    time_idx = min(time_idx, preds.shape[0] - 1)
    time_idx = max(time_idx, 0)
    return preds[time_idx]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_predict.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/predict.py tests/test_predict.py
git commit -m "feat: add predict module for brain prediction loading and slicing"
```

---

## Task 7: Match Module

**Files:**
- Create: `neurolens/match.py`
- Test: `tests/test_match.py`

Core logic for Module 2: find stimuli that best match a target brain activation pattern using cosine similarity.

- [ ] **Step 1: Write failing test**

```python
# tests/test_match.py
import json
import tempfile
from pathlib import Path

import numpy as np

from neurolens.cache import CacheManager
from neurolens.match import (
    find_similar_stimuli,
    build_target_from_regions,
    find_contrast_stimuli,
)


def _setup(tmp: Path) -> tuple[CacheManager, list[str]]:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()

    ids = []
    for i in range(5):
        sid = f"clip_{i:03d}"
        ids.append(sid)
        preds = np.random.randn(3, 20484)
        np.savez(preds_dir / f"{sid}.npz", preds=preds)
        (roi_dir / f"{sid}.json").write_text(
            json.dumps({"Visual Cortex": float(np.random.rand()),
                         "Auditory Cortex": float(np.random.rand())})
        )
    return CacheManager(tmp), ids


def test_find_similar_stimuli():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        # Use clip_000's first-timestep prediction as target
        target = cm.load_brain_preds("clip_000")[0]
        results = find_similar_stimuli(cm, target, ids, top_k=3)
        assert len(results) == 3
        # Each result is (stimulus_id, similarity_score)
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        # Results should be sorted descending by similarity
        assert results[0][1] >= results[1][1] >= results[2][1]


def test_find_similar_includes_self():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        target = cm.load_brain_preds("clip_000")[0]
        results = find_similar_stimuli(cm, target, ids, top_k=5)
        # clip_000 should be most similar to itself
        assert results[0][0] == "clip_000"
        assert results[0][1] > 0.99


def test_build_target_from_regions():
    target = build_target_from_regions(
        {"Visual Cortex": 1.0, "Auditory Cortex": 0.0}
    )
    assert target.shape == (20484,)


def test_find_contrast_stimuli():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        results = find_contrast_stimuli(
            cm, ids,
            maximize_roi="Visual Cortex",
            minimize_roi="Auditory Cortex",
            top_k=3,
        )
        assert len(results) == 3
        # Each result is (stimulus_id, contrast_score)
        assert results[0][1] >= results[1][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_match.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/match.py
"""Match module: find stimuli matching target brain activation patterns."""

from __future__ import annotations

import numpy as np

from neurolens.cache import CacheManager
from neurolens.roi import ROI_GROUPS


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D arrays."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_similar_stimuli(
    cache: CacheManager,
    target: np.ndarray,
    stimulus_ids: list[str],
    top_k: int = 5,
    time_aggregation: str = "mean",
) -> list[tuple[str, float]]:
    """Find stimuli whose brain predictions are most similar to a target pattern.

    Parameters
    ----------
    cache : CacheManager
    target : np.ndarray
        Target activation pattern, shape (n_vertices,).
    stimulus_ids : list of str
        Stimulus ids to search over.
    top_k : int
        Number of results to return.
    time_aggregation : str
        How to aggregate across timesteps: "mean" averages all timesteps.

    Returns
    -------
    List of (stimulus_id, similarity_score) sorted descending.
    """
    scores = []
    for sid in stimulus_ids:
        preds = cache.load_brain_preds(sid)
        if preds is None:
            continue
        if time_aggregation == "mean":
            avg_pred = preds.mean(axis=0)
        else:
            avg_pred = preds[0]
        sim = _cosine_similarity(target, avg_pred)
        scores.append((sid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_target_from_regions(
    region_intensities: dict[str, float],
    mesh: str = "fsaverage5",
    n_vertices: int = 20484,
) -> np.ndarray:
    """Build a synthetic target activation vector from ROI group intensities.

    Parameters
    ----------
    region_intensities : dict
        Maps ROI group name (from ROI_GROUPS) to desired intensity (0-1).
    mesh : str
        Mesh resolution.
    n_vertices : int
        Total number of vertices.

    Returns
    -------
    np.ndarray of shape (n_vertices,)
    """
    from tribev2.utils import get_hcp_roi_indices

    target = np.zeros(n_vertices)
    for group_name, intensity in region_intensities.items():
        if group_name not in ROI_GROUPS:
            continue
        for region in ROI_GROUPS[group_name]:
            try:
                indices = get_hcp_roi_indices(region, hemi="both", mesh=mesh)
                target[indices] = intensity
            except ValueError:
                continue
    return target


def find_contrast_stimuli(
    cache: CacheManager,
    stimulus_ids: list[str],
    maximize_roi: str,
    minimize_roi: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find stimuli that maximize one ROI while minimizing another.

    Parameters
    ----------
    cache : CacheManager
    stimulus_ids : list of str
    maximize_roi : str
        ROI group name to maximize.
    minimize_roi : str
        ROI group name to minimize.
    top_k : int

    Returns
    -------
    List of (stimulus_id, contrast_score) sorted descending.
    """
    scores = []
    for sid in stimulus_ids:
        roi_summary = cache.load_roi_summary(sid)
        if roi_summary is None:
            continue
        max_val = roi_summary.get(maximize_roi, 0.0)
        min_val = roi_summary.get(minimize_roi, 0.0)
        contrast = max_val - min_val
        scores.append((sid, contrast))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_match.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/match.py tests/test_match.py
git commit -m "feat: add match module for brain-state content matching"
```

---

## Task 8: Eval Module

**Files:**
- Create: `neurolens/eval.py`
- Test: `tests/test_eval.py`

Core logic for Module 3: Representational Similarity Analysis (RSA) comparing AI model embeddings to brain predictions.

- [ ] **Step 1: Write failing test**

```python
# tests/test_eval.py
import tempfile
from pathlib import Path

import numpy as np
import torch

from neurolens.cache import CacheManager
from neurolens.eval import (
    compute_rsa_score,
    compute_pairwise_similarity_matrix,
    compute_model_brain_alignment,
)


def _setup(tmp: Path) -> tuple[CacheManager, list[str]]:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()

    ids = []
    for i in range(10):
        sid = f"clip_{i:03d}"
        ids.append(sid)
        np.savez(preds_dir / f"{sid}.npz", preds=np.random.randn(3, 20484))

    for model_name in ["vjepa2", "clip", "whisper"]:
        emb_dir = tmp / "embeddings" / model_name
        emb_dir.mkdir(parents=True)
        for sid in ids:
            torch.save(torch.randn(256), emb_dir / f"{sid}.pt")

    return CacheManager(tmp), ids


def test_compute_pairwise_similarity_matrix():
    vecs = [np.random.randn(100) for _ in range(5)]
    mat = compute_pairwise_similarity_matrix(vecs)
    assert mat.shape == (5, 5)
    # Diagonal should be ~1
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-6)
    # Should be symmetric
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)


def test_compute_rsa_score():
    # Two identical similarity matrices should have RSA = 1
    mat = np.random.randn(5, 5)
    mat = (mat + mat.T) / 2
    score = compute_rsa_score(mat, mat)
    assert abs(score - 1.0) < 1e-6


def test_compute_rsa_score_uncorrelated():
    np.random.seed(42)
    mat_a = np.random.randn(10, 10)
    mat_a = (mat_a + mat_a.T) / 2
    mat_b = np.random.randn(10, 10)
    mat_b = (mat_b + mat_b.T) / 2
    score = compute_rsa_score(mat_a, mat_b)
    # Should be close to 0 for random matrices
    assert abs(score) < 0.5


def test_compute_model_brain_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        score = compute_model_brain_alignment(cm, "vjepa2", ids)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# neurolens/eval.py
"""Eval module: RSA-based comparison of AI model embeddings to brain predictions."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from neurolens.cache import CacheManager


def compute_pairwise_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a list of vectors.

    Parameters
    ----------
    vectors : list of np.ndarray
        Each array is 1D of the same length.

    Returns
    -------
    np.ndarray of shape (n, n) with cosine similarities.
    """
    mat = np.stack(vectors)  # (n, d)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_normed = mat / norms
    return mat_normed @ mat_normed.T


def compute_rsa_score(
    sim_matrix_a: np.ndarray,
    sim_matrix_b: np.ndarray,
) -> float:
    """Compute RSA score: Spearman correlation between upper triangles.

    Parameters
    ----------
    sim_matrix_a, sim_matrix_b : np.ndarray
        Square similarity matrices of the same size.

    Returns
    -------
    float : Spearman correlation coefficient.
    """
    n = sim_matrix_a.shape[0]
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(n, k=1)
    vec_a = sim_matrix_a[idx]
    vec_b = sim_matrix_b[idx]
    corr, _ = spearmanr(vec_a, vec_b)
    return float(corr)


def compute_model_brain_alignment(
    cache: CacheManager,
    model_name: str,
    stimulus_ids: list[str],
) -> float:
    """Compute overall brain alignment score for a model using RSA.

    Steps:
    1. Load embeddings for all stimuli → compute embedding similarity matrix
    2. Load brain predictions (time-averaged) → compute brain similarity matrix
    3. RSA score = Spearman correlation between the two upper triangles

    Parameters
    ----------
    cache : CacheManager
    model_name : str
    stimulus_ids : list of str

    Returns
    -------
    float : RSA alignment score in [-1, 1].
    """
    embeddings = []
    brain_vecs = []
    for sid in stimulus_ids:
        emb = cache.load_embedding(sid, model_name)
        preds = cache.load_brain_preds(sid)
        if emb is None or preds is None:
            continue
        embeddings.append(emb.numpy())
        brain_vecs.append(preds.mean(axis=0))  # Average across timesteps

    if len(embeddings) < 3:
        return 0.0

    emb_sim = compute_pairwise_similarity_matrix(embeddings)
    brain_sim = compute_pairwise_similarity_matrix(brain_vecs)
    return compute_rsa_score(emb_sim, brain_sim)


def compute_all_model_alignments(
    cache: CacheManager,
    stimulus_ids: list[str],
) -> dict[str, float]:
    """Compute brain alignment scores for all available models.

    Returns
    -------
    dict mapping model_name to RSA score, sorted descending.
    """
    models = cache.available_models()
    scores = {}
    for model_name in models:
        scores[model_name] = compute_model_brain_alignment(cache, model_name, stimulus_ids)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_eval.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add neurolens/eval.py tests/test_eval.py
git commit -m "feat: add eval module with RSA-based model-brain alignment scoring"
```

---

## Task 9: Precompute Notebook

**Files:**
- Create: `neurolens_precompute.ipynb`

This notebook runs on Colab with GPU. It:
1. Installs dependencies and loads TRIBE v2
2. Processes the stimulus library through TRIBE v2
3. Extracts embeddings from comparison models (CLIP, Whisper, GPT-2)
4. Saves everything to the cache directory

- [ ] **Step 1: Create the notebook**

Create `neurolens_precompute.ipynb` with the following cells:

**Cell 0 (Markdown):**
```markdown
# NeuroLens — Precompute Cache

Run this notebook **once** on a GPU runtime to generate the NeuroLens cache.
This processes the stimulus library through TRIBE v2 and comparison models.

**Requirements:** Colab GPU runtime, HuggingFace account (for LLaMA access)
```

**Cell 1 (Code) — Setup:**
```python
# Install dependencies
!pip install -e ".[plotting]"
!pip install open_clip_torch transformers[torch]

import json
import numpy as np
import torch
from pathlib import Path
from tribev2 import TribeModel

CACHE_DIR = Path("neurolens_cache")
CACHE_DIR.mkdir(exist_ok=True)
for subdir in ["stimuli", "brain_preds", "roi_summaries", "embeddings"]:
    (CACHE_DIR / subdir).mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
```

**Cell 2 (Code) — Define stimulus library:**
```python
# Define stimuli — replace paths with your actual video/audio files
# For demo, we use a small set. Add more as needed.
STIMULI = [
    {"id": "clip_001", "name": "Nature timelapse", "category": "Silence + Visuals",
     "media_type": "video", "duration_sec": 10.0, "path": "stimuli/nature.mp4"},
    {"id": "clip_002", "name": "TED talk excerpt", "category": "Speech",
     "media_type": "video", "duration_sec": 12.0, "path": "stimuli/ted_talk.mp4"},
    # Add more stimuli here...
]

# Save metadata (without paths — those are only needed during precompute)
metadata = [{k: v for k, v in s.items() if k != "path"} for s in STIMULI]
(CACHE_DIR / "stimuli" / "metadata.json").write_text(json.dumps(metadata, indent=2))
print(f"Saved metadata for {len(STIMULI)} stimuli")
```

**Cell 3 (Code) — Load TRIBE v2 and generate brain predictions:**
```python
# Load TRIBE v2
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

from neurolens.roi import summarize_by_roi_group

for stim in STIMULI:
    sid = stim["id"]
    print(f"Processing {sid}: {stim['name']}...")

    # Get events and predict
    media_kwarg = {f"{stim['media_type']}_path": stim["path"]}
    events = model.get_events_dataframe(**media_kwarg)
    preds, segments = model.predict(events)

    # Save brain predictions
    np.savez(CACHE_DIR / "brain_preds" / f"{sid}.npz", preds=preds)

    # Save ROI summary (average across timesteps)
    avg_pred = preds.mean(axis=0)
    roi_summary = summarize_by_roi_group(avg_pred)
    (CACHE_DIR / "roi_summaries" / f"{sid}.json").write_text(json.dumps(roi_summary))

    print(f"  -> {preds.shape[0]} timesteps, {preds.shape[1]} vertices")
```

**Cell 4 (Code) — Extract CLIP embeddings:**
```python
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

(CACHE_DIR / "embeddings" / "clip").mkdir(exist_ok=True)

from torchvision.io import read_video

for stim in STIMULI:
    sid = stim["id"]
    if stim["media_type"] != "video":
        continue
    # Read middle frame
    video, _, _ = read_video(stim["path"], pts_unit="sec")
    mid_frame = video[len(video) // 2]  # (H, W, C)
    img = preprocess(mid_frame.permute(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img).squeeze().cpu()
    torch.save(emb, CACHE_DIR / "embeddings" / "clip" / f"{sid}.pt")
    print(f"CLIP embedding for {sid}: {emb.shape}")
```

**Cell 5 (Code) — Extract Whisper embeddings:**
```python
from transformers import WhisperModel, WhisperFeatureExtractor
import soundfile as sf

whisper = WhisperModel.from_pretrained("openai/whisper-base").to(device).eval()
whisper_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

(CACHE_DIR / "embeddings" / "whisper").mkdir(exist_ok=True)

for stim in STIMULI:
    sid = stim["id"]
    # Extract audio path (either direct audio or extracted from video)
    audio_path = stim["path"].replace(".mp4", ".wav")  # Adjust as needed
    try:
        audio, sr = sf.read(audio_path)
        inputs = whisper_fe(audio, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            out = whisper.encoder(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze().cpu()
        torch.save(emb, CACHE_DIR / "embeddings" / "whisper" / f"{sid}.pt")
        print(f"Whisper embedding for {sid}: {emb.shape}")
    except Exception as e:
        print(f"Skipped Whisper for {sid}: {e}")
```

**Cell 6 (Code) — Extract GPT-2 embeddings:**
```python
from transformers import GPT2Model, GPT2Tokenizer

gpt2 = GPT2Model.from_pretrained("gpt2").to(device).eval()
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

(CACHE_DIR / "embeddings" / "gpt2").mkdir(exist_ok=True)

for stim in STIMULI:
    sid = stim["id"]
    # Use stimulus name + category as text input
    text = f"{stim['name']}. Category: {stim['category']}"
    inputs = gpt2_tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gpt2(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().cpu()
    torch.save(emb, CACHE_DIR / "embeddings" / "gpt2" / f"{sid}.pt")
    print(f"GPT-2 embedding for {sid}: {emb.shape}")
```

**Cell 7 (Markdown):**
```markdown
## Done!

Cache generated at `neurolens_cache/`. Download this folder or upload it to
Google Drive / HuggingFace Hub for use with the main `neurolens.ipynb` notebook.
```

- [ ] **Step 2: Verify notebook structure**

Run: `python -c "import json; nb = json.load(open('neurolens_precompute.ipynb')); print(f'{len(nb[\"cells\"])} cells')" `
Expected: `8 cells`

- [ ] **Step 3: Commit**

```bash
git add neurolens_precompute.ipynb
git commit -m "feat: add precompute notebook for generating NeuroLens cache"
```

---

## Task 10: Main Interactive Notebook

**Files:**
- Create: `neurolens.ipynb`

The main notebook with all three interactive modules. Uses `ipywidgets` for interactivity and loads everything from the cache.

- [ ] **Step 1: Create the notebook**

Create `neurolens.ipynb` with the following cells:

**Cell 0 (Markdown):**
```markdown
# NeuroLens

**An interactive neuroscience playground built on TRIBE v2**

Explore how the brain responds to video, audio, and text — and discover which AI models think most like humans.

Three modules:
1. **PREDICT** — See predicted brain activation for a stimulus
2. **MATCH** — Find content that triggers specific brain states
3. **EVAL** — Benchmark AI models against biological brain responses
```

**Cell 1 (Code) — Setup:**
```python
# Install if needed (uncomment in Colab)
# !pip install -e ".[plotting]"
# !pip install plotly ipywidgets

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

from neurolens.cache import CacheManager
from neurolens.stimulus import StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_num_timesteps, get_top_rois
from neurolens.match import find_similar_stimuli, build_target_from_regions, find_contrast_stimuli
from neurolens.eval import compute_all_model_alignments, compute_model_brain_alignment
from neurolens.roi import get_roi_group_names, ROI_GROUPS
from neurolens.viz import plot_brain_surface, make_radar_chart

# Load cache
CACHE_DIR = Path("neurolens_cache")
cache = CacheManager(CACHE_DIR)
library = StimulusLibrary(CACHE_DIR)
print(f"Loaded {len(library)} stimuli, {len(cache.available_models())} models")
```

**Cell 2 (Markdown):**
```markdown
---
## Module 1: PREDICT
Select a stimulus and explore how it activates different brain regions.
```

**Cell 3 (Code) — Predict module:**
```python
# Stimulus picker
stim_dropdown = widgets.Dropdown(
    options=library.dropdown_options(),
    description="Stimulus:",
    style={"description_width": "initial"},
)

# Time slider (updated dynamically)
time_slider = widgets.IntSlider(
    value=0, min=0, max=1, step=1,
    description="Timestep:",
    continuous_update=False,
)

# View selector
view_select = widgets.SelectMultiple(
    options=["left", "right", "medial_left", "medial_right", "dorsal"],
    value=["left", "right"],
    description="Views:",
)

output_predict = widgets.Output()

def update_predict(*args):
    sid = stim_dropdown.value
    n_steps = get_num_timesteps(cache, sid)
    time_slider.max = n_steps - 1

    with output_predict:
        clear_output(wait=True)
        data = get_prediction_at_time(cache, sid, time_slider.value)
        stim = library.get(sid)
        fig = plot_brain_surface(
            data,
            views=list(view_select.value),
            title=f"{stim.name} (t={time_slider.value})",
        )
        plt.show()

        # Top ROIs
        top = get_top_rois(cache, sid, k=5)
        print("\nTop activated regions:")
        for name, val in top:
            bar = "█" * int(abs(val) * 20)
            print(f"  {name:.<30s} {val:+.3f} {bar}")

stim_dropdown.observe(update_predict, names="value")
time_slider.observe(update_predict, names="value")
view_select.observe(update_predict, names="value")

display(widgets.VBox([
    widgets.HBox([stim_dropdown, time_slider]),
    view_select,
    output_predict,
]))
update_predict()
```

**Cell 4 (Markdown):**
```markdown
---
## Module 2: MATCH
Find content that activates specific brain regions, or discover neurally similar stimuli.
```

**Cell 5 (Code) — Match module:**
```python
# Tab 1: Region picker
match_mode = widgets.ToggleButtons(
    options=["Region Picker", "More Like This", "Contrast"],
    description="Mode:",
)

# Region picker controls
region_dropdowns = {}
for name in get_roi_group_names():
    region_dropdowns[name] = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.1,
        description=name, style={"description_width": "initial"},
        layout=widgets.Layout(width="400px"),
    )
region_box = widgets.VBox(list(region_dropdowns.values()))

# More Like This controls
source_dropdown = widgets.Dropdown(
    options=library.dropdown_options(),
    description="Source:",
    style={"description_width": "initial"},
)

# Contrast controls
max_roi = widgets.Dropdown(options=get_roi_group_names(), description="Maximize:")
min_roi = widgets.Dropdown(options=get_roi_group_names(), description="Minimize:", value=get_roi_group_names()[1])

output_match = widgets.Output()

def run_match(btn=None):
    with output_match:
        clear_output(wait=True)
        ids = library.ids()
        mode = match_mode.value

        if mode == "Region Picker":
            intensities = {name: slider.value for name, slider in region_dropdowns.items()}
            target = build_target_from_regions(intensities)
            results = find_similar_stimuli(cache, target, ids, top_k=5)
        elif mode == "More Like This":
            source_preds = cache.load_brain_preds(source_dropdown.value)
            target = source_preds.mean(axis=0)
            results = find_similar_stimuli(cache, target, ids, top_k=5)
        else:  # Contrast
            results = find_contrast_stimuli(
                cache, ids, max_roi.value, min_roi.value, top_k=5
            )

        print(f"Top matches ({mode}):\n")
        radar_data = {}
        for rank, (sid, score) in enumerate(results, 1):
            stim = library.get(sid)
            print(f"  {rank}. {stim.name} [{stim.category}] — score: {score:.3f}")
            roi = cache.load_roi_summary(sid)
            if roi and rank <= 3:
                radar_data[stim.name] = roi

        if radar_data:
            fig = make_radar_chart(radar_data, title="ROI Activation Profiles")
            plt.show()

match_btn = widgets.Button(description="Find Matches", button_style="primary")
match_btn.on_click(run_match)

display(widgets.VBox([
    match_mode,
    region_box,
    widgets.HBox([source_dropdown]),
    widgets.HBox([max_roi, min_roi]),
    match_btn,
    output_match,
]))
```

**Cell 6 (Markdown):**
```markdown
---
## Module 3: EVAL
Which AI model thinks most like a human brain?
Compare model representations against predicted brain responses.
```

**Cell 7 (Code) — Eval module:**
```python
output_eval = widgets.Output()

def run_eval(btn=None):
    with output_eval:
        clear_output(wait=True)
        ids = library.ids()
        print("Computing brain alignment scores (RSA)...\n")

        scores = compute_all_model_alignments(cache, ids)

        # Leaderboard
        print("=" * 50)
        print(f"{'Rank':<6}{'Model':<20}{'Brain Alignment':>15}")
        print("=" * 50)
        for rank, (model, score) in enumerate(scores.items(), 1):
            bar = "█" * int(max(0, score) * 30)
            print(f"{rank:<6}{model:<20}{score:>+.4f}  {bar}")
        print("=" * 50)

        # Radar comparison of top 3
        if len(scores) >= 2:
            print("\nSelect two models to compare:")

model_a = widgets.Dropdown(
    options=cache.available_models(),
    description="Model A:",
)
model_b = widgets.Dropdown(
    options=cache.available_models(),
    description="Model B:",
    value=cache.available_models()[-1] if len(cache.available_models()) > 1 else cache.available_models()[0],
)

output_compare = widgets.Output()

def compare_models(btn=None):
    with output_compare:
        clear_output(wait=True)
        ids = library.ids()
        # Per-ROI alignment would need per-ROI RSA — simplified version:
        # Show each model's average ROI activation correlation
        print(f"Comparing {model_a.value} vs {model_b.value}...\n")
        score_a = compute_model_brain_alignment(cache, model_a.value, ids)
        score_b = compute_model_brain_alignment(cache, model_b.value, ids)

        print(f"  {model_a.value}: RSA = {score_a:+.4f}")
        print(f"  {model_b.value}: RSA = {score_b:+.4f}")

        # Brain report card
        for model_name in [model_a.value, model_b.value]:
            print(f"\n--- Brain Report Card: {model_name} ---")
            score = compute_model_brain_alignment(cache, model_name, ids)
            pct = max(0, score) * 100
            print(f"  Overall brain alignment: {pct:.1f}%")

compare_btn = widgets.Button(description="Compare Models", button_style="info")
compare_btn.on_click(compare_models)

eval_btn = widgets.Button(description="Run Leaderboard", button_style="primary")
eval_btn.on_click(run_eval)

display(widgets.VBox([
    eval_btn,
    output_eval,
    widgets.HBox([model_a, model_b, compare_btn]),
    output_compare,
]))
```

**Cell 8 (Markdown):**
```markdown
---
## Explore Further

- **TRIBE v2 Paper:** [A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- **TRIBE v2 Demo:** [aidemos.atmeta.com/tribev2](https://aidemos.atmeta.com/tribev2/)
- **Add more stimuli:** Edit `neurolens_precompute.ipynb` and re-run
- **Add more models:** Extract embeddings from any HuggingFace model into the cache

Built with NeuroLens by Ansuman SS Bhujabala.
```

- [ ] **Step 2: Verify notebook loads**

Run: `python -c "import json; nb = json.load(open('neurolens.ipynb')); print(f'{len(nb[\"cells\"])} cells')" `
Expected: `9 cells`

- [ ] **Step 3: Commit**

```bash
git add neurolens.ipynb
git commit -m "feat: add main NeuroLens interactive notebook with Predict, Match, and Eval modules"
```

---

## Task 11: Package Exports and Final Wiring

**Files:**
- Modify: `neurolens/__init__.py`

- [ ] **Step 1: Update package init with public API**

```python
# neurolens/__init__.py
"""NeuroLens: Interactive neuroscience playground built on TRIBE v2."""

from neurolens.cache import CacheManager
from neurolens.stimulus import Stimulus, StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_num_timesteps, get_top_rois
from neurolens.match import find_similar_stimuli, build_target_from_regions, find_contrast_stimuli
from neurolens.eval import compute_all_model_alignments, compute_model_brain_alignment
from neurolens.roi import ROI_GROUPS, get_roi_group_names, summarize_by_roi_group
from neurolens.viz import plot_brain_surface, make_radar_chart

__all__ = [
    "CacheManager",
    "Stimulus",
    "StimulusLibrary",
    "get_prediction_at_time",
    "get_num_timesteps",
    "get_top_rois",
    "find_similar_stimuli",
    "build_target_from_regions",
    "find_contrast_stimuli",
    "compute_all_model_alignments",
    "compute_model_brain_alignment",
    "ROI_GROUPS",
    "get_roi_group_names",
    "summarize_by_roi_group",
    "plot_brain_surface",
    "make_radar_chart",
]
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add neurolens/__init__.py
git commit -m "feat: export full public API from neurolens package"
```

---

## Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

Verifies the full pipeline works end-to-end with a mock cache.

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test with a mock cache."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from neurolens.cache import CacheManager
from neurolens.stimulus import StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_top_rois
from neurolens.match import find_similar_stimuli, find_contrast_stimuli
from neurolens.eval import compute_all_model_alignments
from neurolens.viz import plot_brain_surface, make_radar_chart


def _build_mock_cache(tmp: Path) -> Path:
    """Build a complete mock cache for integration testing."""
    stimuli = [
        {"id": f"clip_{i:03d}", "name": f"Clip {i}", "category": cat,
         "media_type": "video", "duration_sec": 10.0}
        for i, cat in enumerate(["Speech", "Music", "Silence + Visuals",
                                  "Emotional", "Multimodal-rich"])
    ]
    (tmp / "stimuli").mkdir()
    (tmp / "stimuli" / "metadata.json").write_text(json.dumps(stimuli))

    (tmp / "brain_preds").mkdir()
    (tmp / "roi_summaries").mkdir()
    for s in stimuli:
        preds = np.random.randn(5, 20484).astype(np.float32)
        np.savez(tmp / "brain_preds" / f"{s['id']}.npz", preds=preds)
        roi = {"Visual Cortex": float(np.random.rand()),
               "Auditory Cortex": float(np.random.rand()),
               "Language Areas": float(np.random.rand())}
        (tmp / "roi_summaries" / f"{s['id']}.json").write_text(json.dumps(roi))

    for model in ["vjepa2", "clip", "whisper"]:
        (tmp / "embeddings" / model).mkdir(parents=True)
        for s in stimuli:
            torch.save(torch.randn(256), tmp / "embeddings" / model / f"{s['id']}.pt")

    return tmp


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _build_mock_cache(Path(tmp))
        cache = CacheManager(cache_dir)
        library = StimulusLibrary(cache_dir)

        # 1. Predict
        data = get_prediction_at_time(cache, "clip_000", time_idx=2)
        assert data.shape == (20484,)
        top = get_top_rois(cache, "clip_000", k=3)
        assert len(top) == 3

        # 2. Match
        target = data
        results = find_similar_stimuli(cache, target, library.ids(), top_k=3)
        assert len(results) == 3
        contrast = find_contrast_stimuli(
            cache, library.ids(), "Visual Cortex", "Auditory Cortex", top_k=3
        )
        assert len(contrast) == 3

        # 3. Eval
        scores = compute_all_model_alignments(cache, library.ids())
        assert len(scores) == 3
        assert all(isinstance(v, float) for v in scores.values())

        # 4. Visualization
        fig = plot_brain_surface(data, views=["left"])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

        radar_data = {"Clip 0": cache.load_roi_summary("clip_000")}
        fig2 = make_radar_chart(radar_data)
        assert fig2 is not None
        plt.close(fig2)
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for full NeuroLens pipeline"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Scaffolding | — |
| 2 | ROI utilities | 3 |
| 3 | Stimulus library | 5 |
| 4 | Cache manager | 6 |
| 5 | Visualization helpers | 3 |
| 6 | Predict module | 5 |
| 7 | Match module | 4 |
| 8 | Eval module | 4 |
| 9 | Precompute notebook | — |
| 10 | Main notebook | — |
| 11 | Package exports | — |
| 12 | Integration test | 1 |
| **Total** | **12 tasks** | **31 tests** |
