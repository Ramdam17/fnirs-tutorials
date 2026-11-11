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

The Node.js requirement is read from the Jupyter Book source, not from the documentation. The check in `jupyter_book/__main__.py` accepts Node.js 18, 20 or 22 and later, and its error message says "Jupyter Book requires node 18, 20, or 22+". When no suitable Node.js is on the path, the tool offers to download Node.js 22.17.0 with nodeenv into a user data folder, after a prompt that the environment variable `JB_ALLOW_NODEENV` answers in advance.

## Platform overrides (macOS 27, arm64)

Two of Cedalion's pins do not work on macOS 27, so `pyproject.toml` overrides them through `[tool.uv] override-dependencies`.

The scipy 1.15.x wheels for macOS arm64 fail to load on macOS 27. The dynamic loader rejects `scipy/sparse/linalg/_propack/_spropack.cpython-312-darwin.so` with the message "section '__DATA/__thread_bss' has a zero-fill section type, but offset field is not zero". Versions 1.15.2 and 1.15.3 both fail; 1.16.3 and 1.17.1 load. Because Cedalion, MNE-NIRS and NIRSimple all import scipy's sparse linear algebra, none of them imports without the override.

Lifting scipy to 1.16 breaks statsmodels 0.14.4, which imports the private function `scipy._lib._util._lazywhere` that scipy 1.16 removed. The failure shows up when importing `cedalion.models.glm`, so the whole GLM module is unavailable until statsmodels is lifted to 0.14.6, the last 0.14 patch release.

With both overrides, Cedalion's own test suite at tag `v26.5.1` gives 302 passed and 4 skipped, with no failures (run on 2026-10-31). One warning deserves a later look: `test_global_component_subtract` emits "invalid value encountered in matmul" from scikit-learn's PCA. Whether it also appears with the original pins can only be checked on Linux, where the original pins install; this comparison should be made once CI exists. Linux (CI, Colab) is not affected by the scipy loading problem.

## Cedalion API renames, 25.x to 26.5

Every API name used in the brief was resolved by introspection on the installed 26.5.1 package. The names below changed between 25.1.0 and 26.5.1. Notebooks use the new names.

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
