# Compatibility notes

This file records the exact software versions the tutorials are built against, the API renames a reader will meet when comparing with older Cedalion material, and the platform overrides the environment needs. It was written on 2026-10-31 from measurements on the project environment.

## Environment

The environment is managed with uv from `pyproject.toml` and `uv.lock`. Cedalion is installed from its git tag rather than from PyPI, because the PyPI release (25.1.0, June 2025) lags the release branch. The dependency list in Cedalion's `pyproject.toml` is the one its authors use for Google Colab, so installing from the tag with pip or uv is a supported path even though their recommended local install is conda.

| Package | Version | Why this version |
|---|---|---|
| Python | 3.12.12 | Cedalion requires 3.11 or later; 3.12 is the lab default |
| cedalion | 26.5.1 (git tag `v26.5.1`) | latest stable tag; `v26.8.0-rc1` and `rc2` exist but are release candidates |
| mne | 1.9.0 | pinned by Cedalion (`mne==1.9.*`) |
| mne-nirs | 0.7.3 | pinned by Cedalion (`mne-nirs==0.7.*`) |
| numpy | 2.2.6 | pinned by Cedalion |
| scipy | 1.16.3 | override, see below (Cedalion pins 1.15.*) |
| statsmodels | 0.14.6 | override, see below (Cedalion pins 0.14.4) |
| nilearn | 0.11.1 | pinned by Cedalion |
| xarray | 2025.6.1 | pinned by Cedalion |
| pint-xarray | 0.4 | pinned by Cedalion |
| snirf | 0.8.0 | SNIRF validation; successor of pysnirf2 |
| nirsimple | 0.1.6 | reference MBLL implementation for the by-hand module |

## Book toolchain

The site is built with Jupyter Book 2, which is a Python package that bundles the MyST engine and runs it on Node.js. It is installed in the development dependency group, and the versions below were measured on 2026-11-11 on the Mac.

| Component | Version | Where the version comes from |
|---|---|---|
| jupyter-book | 2.1.7 | `uv run jupyter book --version` prints `v2.1.7`; pinned in `pyproject.toml` |
| MyST engine (mystmd) | bundled, not installed separately | `uv pip list` shows no `mystmd` package; the bundle `jupyter_book/dist/jupyter-book.cjs` contains `myst-cli` 1.11.0 (read from its source, since `jupyter book --version` reports only the Jupyter Book version) |
| Node.js | v26.12.0 on the Mac | `node --version` |
| nodeenv | 1.12.0 | `uv pip list`; used only when no suitable Node.js is found |
| D3 (interactive illustrations) | 7.9.2, ISC licence | `interactive/vendor/d3.min.js`, the file `dist/d3.min.js` of the npm package `d3` 7.9.2 (280,622 bytes, SHA-256 `912111098ea99503a6f4967361d2bbdc9dbf58cf9c678c8e27e5b02d1a07f0c1`), copied into the repository with its `LICENSE` and loaded by a relative address; `tests/test_interactive.py` pins the size and the hash |
| book-theme (site theme) | commit `08951401cf4aa6710f040fe34dcf0c1351a08639`, 2026-11-10, message "v2.0.0 from ea08e186" | the zip comment of the archive that the first build downloaded (`_build/templates/site/myst/book-theme/template.zip`); `myst.yml` pins the archive of this commit |

The theme repository `myst-templates/book-theme` has no tag, so the pin is a commit; the bare name `book-theme` would follow its `main` branch. The pinned archive was rebuilt from an empty `_build/templates` on 2026-11-11 and renders the same three pages.

The Node.js requirement is read from the Jupyter Book source, not from the documentation. The check in `jupyter_book/__main__.py` accepts Node.js 18, 20 or 22 and later, and its error message says "Jupyter Book requires node 18, 20, or 22+". When no suitable Node.js is on the path, the tool offers to download Node.js 22.17.0 with nodeenv into a user data folder, after a prompt that the environment variable `JB_ALLOW_NODEENV` answers in advance.

## Platform overrides (macOS 27, arm64)

Two of Cedalion's pins do not work on macOS 27, so `pyproject.toml` overrides them through `[tool.uv] override-dependencies`.

The scipy 1.15.x wheels for macOS arm64 fail to load on macOS 27. The dynamic loader rejects `scipy/sparse/linalg/_propack/_spropack.cpython-312-darwin.so` with the message "section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero". Versions 1.15.2 and 1.15.3 both fail; 1.16.3 and 1.17.1 load. Because Cedalion, MNE-NIRS and NIRSimple all import scipy's sparse linear algebra, none of them imports without the override.

Lifting scipy to 1.16 breaks statsmodels 0.14.4, which imports the private function `scipy._lib._util._lazywhere` that scipy 1.16 removed. The failure shows up when importing `cedalion.models.glm`, so the whole GLM module is unavailable until statsmodels is lifted to 0.14.6, the last 0.14 patch release.

With both overrides, Cedalion's own test suite at tag `v26.5.1` gives 302 passed and 4 skipped, with no failures (run on 2026-10-31). One warning deserves a later look: `test_global_component_subtract` emits "invalid value encountered in matmul" from scikit-learn's PCA. Whether it also appears with the original pins can only be checked on Linux, where the original pins install; this comparison should be made once CI exists. Linux (CI, Colab) is not affected by the scipy loading problem.

## Google Colab

Every notebook of the book opens on Google Colab from a link under its title, and its first code cell installs the environment and the data there. The cell is generated from `book/_templates/colab_setup.py` by `scripts/sync_colab_cells.py`, with the parameters of `config/colab.yaml`. On Colab the cell is shown folded, under the title "Setup for Google Colab" and a line "Show code". Colab folds a code cell that has both a title line, a comment that starts with `@title`, and the metadata key `cellView` with the value `form`; either one alone leaves the cell in full, and so does the tag `hide-input`, which is what folds the cell in the book. What follows was measured on 2026-11-11 and 2026-11-12 on Colab, with a Colab Pro account; a free account was not tried.

The notebooks ask Colab for the runtime version `2026.07`, through the key `colab.runtime_attributes.runtime_version` of the notebook metadata, which Colab honours for a notebook opened from GitHub. That version runs Python 3.12.13. The default runtime, "Latest (recommended)", runs Python 3.13.18, which the package refuses because `pyproject.toml` requires Python 3.12; on it the setup cell stops with a message that names the version to choose. Google states that a past runtime version stays available for one year after its release, so this pin has to move before the summer of 2027. Connecting to a past version took between 90 and 104 s on five connections, against 8 s for the default runtime on one.

The cell installs the package with `uv pip install --system --override`, and the override file holds the two overrides of `pyproject.toml`, scipy 1.16.3 and statsmodels 0.14.6. A plain `pip install` of the package does not read the overrides and gives Cedalion's own pins, scipy 1.15.3 and statsmodels 0.14.4. The install replaces nine packages of the Colab image (click, h5py, nibabel, numpy, pandas, pooch, pywavelets, scikit-learn and xarray) and leaves the packages the Colab kernel is built on, ipykernel 6.17.1 and ipython 7.34.0, untouched. Installing every version of `uv.lock` instead replaces those too, and the kernel then never comes back after a restart. The Colab environment is therefore not the lock file in full: a package that neither Cedalion nor the project pins keeps the version of the image, for instance matplotlib 3.10.0 where the lock has 3.10.9.

| Package | In the `2026.07` image | After the setup cell |
|---|---|---|
| numpy | 2.0.2 | 2.2.6 |
| scipy | 1.16.3 | 1.16.3 |
| statsmodels | 0.14.6 | 0.14.6 |
| pandas | 2.2.2 | 2.3.3 |
| xarray | 2025.12.0 | 2025.6.1 |
| scikit-learn | 1.6.1 | 1.7.2 |
| h5py | 3.16.0 | 3.14.0 |
| cedalion | absent | 26.5.1 |

The session has to restart once after the install. The Colab kernel imports numpy before the first cell runs, Cedalion requires numpy 2.2, and the image ships 2.0.2: without a restart, `import cedalion` fails with `ImportError: cannot import name '_center' from 'numpy._core.umath'`. `uv` does not make Colab offer a restart, so the cell ends its own kernel after it has announced it, and Colab then shows the notice "Your session crashed for an unknown reason", which is expected here. Run again, the cell finds the file it wrote after the install, installs nothing and prints "The environment is ready."

The cell follows Cedalion's own Colab script (`scripts/colab_setup.py` at the tag `v26.5.1`) on two points and departs from it on five.

- Like Cedalion's script, it installs with `uv` and ends the kernel process to restart the session, after a wait of two seconds that lets the last lines reach the browser.
- It does not mount the reader's Google Drive, so no sign-in is asked: the data are downloaded into the Colab machine by `scripts/download_data.py`, with their checksums.
- It installs a release tag of Cedalion through the dependencies of the package, with the two overrides, where Cedalion's script installs a branch.
- It does not uninstall `opt-einsum` and does not install the system package `tree`. The smoke notebook runs with `opt-einsum` in place; Cedalion's script gives no reason for removing it.
- It does not configure PyVista for a display without a screen, which Cedalion's script does on its second run. The book has no 3D view yet; the chapter that adds one has to add that configuration.
- It checks the Python version first, records what it installed so that a second run does nothing, and prints the time the install and the data download took.

On three runs from the deployed book, the setup cell took 88 s, 453 s and 115 s, and the smoke notebook then ran in 13 s, 17 s and 12 s. The second setup was slow because the Colab machine received the packages from PyPI at under 1 MB/s for about twelve minutes; half an hour later the same machine downloaded the largest of them, a 105 MB wheel, in 3 s. `uv` prints nothing while it downloads, so a slow install shows no progress for several minutes. The cell says so in a line it prints before the install starts.

## Cedalion API renames, 25.x to 26.5

Every API name the series uses was resolved by introspection on the installed 26.5.1 package. The names below changed between 25.1.0 and 26.5.1. Notebooks use the new names.

| 25.x | 26.5.1 |
|---|---|
| `cedalion.sigproc.motion_correct` | `cedalion.sigproc.motion` |
| `motion_correct_spline` | `spline` |
| `motion_correct_splineSG` | `spline_sg` |
| `motion_correct_wavelet` | `wavelet` |
| `motion_correct_PCA` | `pca` |
| `motion_correct_PCA_recurse` | `pca_recurse` |
| `cedalion.plots` | `cedalion.vis` |
| `cedalion.imagereco` | `cedalion.dot` |
| `cedalion.datasets` | merged into `cedalion.data` |
| `cedalion.nirs.int2od`, `od2conc` | `cedalion.nirs.cw.int2od`, `cedalion.nirs.cw.od2conc` (also `.fd`, `.td`) |
| `cedalion.sigdecomp` (ICA-EBM, ERBM, SPoC) | `cedalion.sigdecomp.unimodal` |

The old `motion_correct_*` function names still exist in `cedalion.sigproc.motion` as aliases, so older code keeps running. The GLM entry point is `cedalion.models.glm.fit(ts, design_matrix, noise_model="ols", ar_order=30, max_jobs=-1, verbose=False)`, with `noise_model` one of `ols`, `rls`, `wls`, `ar_irls`, `gls`, `glsar`.

## Other packages

`pysnirf2` 0.7.3 fails to import under numpy 2 because it uses `np.string_`, which numpy 2.0 removed. The `snirf` package (0.8.0) is its maintained successor and provides `snirf.validateSnirf`; the project uses it instead.

In NIRSimple, `cbsi` lives in `nirsimple.processing` (and is exported at top level as `nirsimple.cbsi`), not in `nirsimple.preprocessing`.
