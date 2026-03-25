# PyAutoLens Workspace

This is the tutorial and example workspace for **PyAutoLens**, a Python library for strong gravitational lens modeling.

## Repository Structure

- `scripts/` - Runnable Python scripts organised by topic
  - `imaging/` - CCD imaging (HST, JWST, Euclid) lens modeling
  - `interferometer/` - ALMA/JVLA uv-plane lens modeling
  - `multi/` - Multi-wavelength simultaneous modeling
  - `guides/` - API guides: modeling, results, plotting, over-sampling, HPC
  - `howtolens/` - Tutorial lecture series
  - `slam_pipeline/` - Source, Light and Mass (SLaM) pipeline examples
- `notebooks/` - Jupyter notebook versions of scripts (mirrors `scripts/`)
- `config/` - PyAutoLens configuration YAML files
- `dataset/` - Input `.fits` files and other data
- `output/` - Model-fit results (generated, not committed)

## Running Scripts

Scripts are run from the repository root so relative paths to `dataset/` and `output/` resolve correctly:

```bash
python scripts/imaging/modeling/start_here.py
```

**Integration testing / fast mode**: Set `PYAUTOFIT_TEST_MODE=1` to skip non-linear search sampling:

```bash
PYAUTOFIT_TEST_MODE=1 python scripts/imaging/modeling/start_here.py
```

## Core API Patterns

**Imports** (standard across all scripts):
```python
import autofit as af
import autolens as al
import autolens.plot as aplt
```

**Standard modeling workflow**:
1. Load dataset via `al.Imaging.from_fits(...)`
2. Apply a 2D mask: `al.Mask2D.circular(...)`
3. Apply adaptive over-sampling
4. Compose a model using `af.Model` / `af.Collection` with `al.Galaxy`, light profiles and mass profiles
5. Configure a non-linear search (default: `af.Nautilus`)
6. Create an `al.AnalysisImaging(dataset=dataset, use_jax=True)` object
7. Run `search.fit(model=model, analysis=analysis)`

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the Python files in `scripts/`. **Always edit the `.py` scripts**, not the notebooks directly. The `# %%` marker alternates between code and markdown cells.

### Building Notebooks

Notebooks are generated from Python scripts using `generate.py` from the `PyAutoBuild` repo. Run from the workspace root:

```bash
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autolens
```

This converts every `.py` file in `scripts/` to a `.ipynb` in `notebooks/` by:
1. Converting triple-quoted docstrings into `# %%` Jupyter cell markers
2. Running `ipynb-py-convert` to produce `.ipynb` files
3. Restoring commented Jupyter magic commands (e.g. `# %matplotlib` → `%matplotlib`)

Use the `/generate_and_merge` skill to build notebooks, commit, push, raise a PR, and merge into `main`.

## Related Repos

- **PyAutoLens** source: `../PyAutoLens`
- **PyAutoGalaxy** source: `../PyAutoGalaxy`
- **PyAutoBuild**: `../PyAutoBuild` — notebook generation and CI/CD tooling
