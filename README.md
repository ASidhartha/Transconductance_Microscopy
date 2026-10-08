# IETS spectrum denoising and bond mapping

This project cleans up the second-derivative spectra (`DerivativeY2_2`, i.e. d²I/dV²)
measured on a grid of electrode pairs, finds the vibrational peaks, and matches them to
candidate chemical bonds using the FTIR table from
[instanano.com](https://instanano.com/all/characterization/ftir/ftir-functional-group-search/).

The full write-up, with before/after figures and the bond summary for both devices, is
**`outputs/IETS_Denoising_Report.docx`**.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Quick start](#2-quick-start)
3. [Input data](#3-input-data)
4. [Running the pipeline step by step](#4-running-the-pipeline-step-by-step)
5. [Outputs and how to read them](#5-outputs-and-how-to-read-them)
6. [Confidence tiers: how much to trust a peak](#6-confidence-tiers-how-much-to-trust-a-peak)
7. [Customising](#7-customising)
8. [Known issues with the current data](#8-known-issues-with-the-current-data)
9. [Troubleshooting](#9-troubleshooting)
10. [How it works (short version)](#10-how-it-works-short-version)
11. [Project layout](#11-project-layout)

---

## 1. What it does

For every electrode-pair CSV file, the pipeline:

1. **Splits the sweep into its two halves.** `Wvnmbr = 8066 × bias`, so the +V half and the
   mirrored −V half are two measurements of the same vibrational spectrum.
2. **Aligns the two halves.** In these data the halves are shifted and stretched relative to
   each other, by different amounts in different spectra. The alignment is only used when it
   clearly beats chance; otherwise the spectrum is flagged *not aligned*.
3. **Denoises both halves together** with a 1D U-Net neural network trained on simulated
   spectra whose noise was measured from the real data.
4. **Finds peaks** and labels each one *confirmed*, *single-half* or *tentative*
   (see [section 6](#6-confidence-tiers-how-much-to-trust-a-peak)).
5. **Matches each peak to candidate bonds** in the FTIR table, taking the position
   uncertainty into account.
6. **Produces** denoised spectra, a peak table, per-pair plots, band heatmaps and the report.

---

## 2. Quick start

Requires macOS or Linux with Python 3.11. An Apple-silicon or CUDA GPU speeds up training
but is not required. Node.js is only needed to rebuild the Word report.

```bash
git clone <repo-url> && cd <repo>
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Trained models are already in models/. To just re-run the analysis:
.venv/bin/python evaluate.py        # ~5 min   -> outputs/metrics.json (thresholds used below)
.venv/bin/python apply.py           # ~2 min   -> outputs/ (denoised spectra, peaks, PDFs)
```

To start from scratch (for example after adding new data), first run
`.venv/bin/python train.py`, which takes about 30–60 minutes.

---

## 3. Input data

```
data/
├── impure/   11_12_alpha_new.csv, 11_21_alpha_new.csv, ...  (55 files)
└── pure/     11_12_alpha_new.csv, ...                        (55 files)
```

- **File names** encode the electrode pair: `11_12` is the pair between grid node (1,1) and
  node (1,2). Each measurement is plotted at the midpoint of its two nodes (X = first digits,
  Y = second digits), as in the original heatmaps.
- **Columns** are read by position, so header spelling does not matter
  (`wvnmbr`/`Wvnmbr`/`wvnm`, `Current`/`currrent`, …). Expected order: wavenumber, voltage,
  current, DerivativeY1, DerivativeY1_1, DerivativeY2, DerivativeY2_1, DerivativeY2_2, Alpha.
- Files with an extra empty column (`Var3`, `null`) are handled automatically. These files
  have no `DerivativeY2_2`, so the last second-derivative column is used instead (none of the
  files in `data/` has this layout).
- **Units.** The derivative columns are derivatives *per sample*, not per volt
  (`DerivativeY1` equals MATLAB/NumPy `gradient(Current)` with unit spacing exactly). The
  loader divides them by the voltage step averaged over 40 samples (and its square) to give
  A/V and A/V². The mean step is 1.25 mV in every file. In ten impure files (row 1) the voltage
  is recorded on a 1 mV grid, so single steps go 1, 1, 1, 2 mV; a median step (1.0 mV) would
  overstate d²I/dV² there by 1.56×.
- **`Alpha` column.** In `data/` it equals (d²I/dV²)/(dI/dV) up to a file-specific constant and
  is filled for V > 0 only. It is not used. The coupling coefficient α = (d²I/dV²)/(2K₀) of the
  manuscript is computed by `alpha_analysis.py` (section 4). `matlab/alpha_calculation_updated.m`
  is the equivalent MATLAB script (2 mm electrode pitch), corrected to use the mean voltage step.
- **Adding a device:** add its folder to `DEVICES` in `iets/data.py`, then re-run
  `train.py`, `evaluate.py` and `apply.py`.

---

## 4. Running the pipeline step by step

Run from the project folder. On a Mac, prefix long jobs with `caffeinate -i` so the machine
does not sleep during them (sleeping slowed training 10-fold in testing).

| Step | Command | Time | What it produces |
|---|---|---|---|
| 1. Train | `caffeinate -i .venv/bin/python train.py` | 30–60 min | `models/unet_2branch.pt` (main model), `models/unet_1branch.pt` (validation only), `models/synth_test.npz`, and `outputs/alignment.json` on first run |
| 2. Evaluate | `.venv/bin/python evaluate.py` | ~5 min | `outputs/metrics.json`: benchmark results and the peak thresholds for each confidence tier |
| 3. Apply | `.venv/bin/python apply.py` | ~2 min | denoised spectra, `peaks.csv`, spectra and heatmap PDFs |
| 4. Report figures | `.venv/bin/python make_report_figures.py` | ~3 min | `outputs/report/`: figures, region and bond summaries |
| 4b. ±V polarity test | `.venv/bin/python polarity_check.py` | ~4 min | `outputs/polarity/`: does the −V half carry the same spectrum? |
| 4c. Coupling coefficient α | `.venv/bin/python alpha_analysis.py` | ~1 min | `outputs/alpha/`: α = (d²I/dV²)/(2K₀) before and after denoising |
| 4d. Manuscript figures | `.venv/bin/python make_paper_figures.py` | ~1 min | `outputs/paper_figures/` |
| 5. Word report | see below | seconds | `outputs/IETS_Denoising_Report.docx` |

Step 3 reads the thresholds written by step 2, so run them in order.

To rebuild the Word report:

```bash
npm install                      # once, installs docx from package.json
node report/build_report.js
```

**Alignment cache.** The alignment of every spectrum is cached in `outputs/alignment.json`.
Delete that file if the input data changes; it is recomputed on the next run (about 1 minute).

---

## 5. Outputs and how to read them

### `outputs/IETS_Denoising_Report.docx`
The full report: data issues, method, validation, before/after figures, and the bond
summary across grids and devices. Start here.

### `outputs/peaks.csv`: one row per peak

| Column | Meaning |
|---|---|
| `device`, `pair`, `x`, `y` | Which measurement, and its midpoint on the grid |
| `wavenumber` | Peak position (cm⁻¹) on the **common axis**, halfway between the two halves' axes |
| `wavenumber_pos_axis`, `wavenumber_neg_axis` | The same peak's position on the +V and on the −V half's own axis |
| `position_uncertainty` | ± cm⁻¹, half the disagreement between the two halves. Empty if the spectrum could not be aligned |
| `aligned` | Whether the two halves were aligned (see [section 8](#8-known-issues-with-the-current-data)) |
| `confidence` | `confirmed`, `single-half (+V/−V)` or `tentative (+V/−V)`, see [section 6](#6-confidence-tiers-how-much-to-trust-a-peak) |
| `symmetry` | For confirmed peaks: `odd` (halves have opposite sign) or `even` (same sign) |
| `strength` | Peak height in A/V² (geometric mean of both halves for confirmed peaks) |
| `snr` | The same height in noise units |
| `duplicated_across_devices` | `True` if this pair's data is identical in both device folders |
| `candidate_bonds` | Up to three FTIR bands matching the position (nearest first, then narrowest) |

`outputs/report/peak_bond_assignments.csv` has the same peaks plus a spectral region, the
single best FTIR match and *every* bond within the position uncertainty.

### `outputs/denoised/<device>/<pair>.csv`: the cleaned spectra
Columns `wavenumber_mid` (common axis), `wavenumber_pos_axis`, `wavenumber_neg_axis`,
`raw_pos`, `raw_neg`, `denoised_pos`, `denoised_neg` (A/V²). Values are empty where a half has
no data after alignment.

### `outputs/spectra_<device>.pdf`: before/after plots for every pair
Left: raw (detrended); right: denoised. Blue is the +V half and red is the mirrored −V half.
Vertical lines mark peaks: **solid = confirmed**, **dashed = single-half**,
**dotted = tentative**. Each panel title shows the alignment used, or says the halves could
not be aligned, and flags duplicated pairs.

A **nearly flat** denoised panel means nothing in that spectrum was consistent enough to be
told apart from noise. This typically happens when the raw trace is a series of similar-sized
wiggles that disagree between the halves. The raw data is still in the CSV if you want to
inspect it.

### `outputs/heatmaps_<device>.pdf`: band maps
One page per band from `iets/bands.py`. Colour shows the strongest denoised signal
magnitude in that band, averaged over the two halves. Magnitude is used because the sign
is not consistent between halves. **●** marks a confirmed peak in the band and **□** a
single-half peak.

### `outputs/alpha/`: coupling coefficient α
α = (d²I/dV²)/(2K₀) with K₀ = μ₀C_ox·W/L (constants in `iets/alpha.py`; L = 2 mm for side
pairs and 2.83 mm for diagonal pairs, so K₀ = 27.7 and 19.6 µA/V²).
- `<device>/<pair>.csv`: measured and denoised α of both halves on the common axis.
- `alpha_spectra_<device>.pdf`: for every pair, the original α of the +V half (as the MATLAB
  script computes it) and the denoised α with the removed background added back.
- `peaks_alpha.csv`: `peaks.csv` plus K₀, pair length and α at each peak (measured and denoised).
- `alpha_summary.json`: the device statistics quoted in the manuscript.
- `matlab_check.csv`: the original MATLAB `Alpha` output divided by the α computed here, per
  file (100 of 110 agree within 0.2%; the ten row-1 impure files differ by 1.56×, the
  median-step error fixed in `matlab/`). Regenerated only if the MATLAB output is present in
  `Alpha_2mm/`.

### Other files
- `outputs/metrics.json`: benchmark numbers and tier thresholds.
- `outputs/alignment.json`: per-spectrum stretch `k`, shift `c`, correlation and whether it was accepted.
- `outputs/report/region_summary.csv`, `bond_family_summary.csv`: the bond-summary tables in the report.

---

## 6. Confidence tiers: how much to trust a peak

Precision was measured on simulated spectra where the true peaks are known:

| Tier | Rule | Precision |
|---|---|---|
| **confirmed** | Found in both halves within 32 cm⁻¹ | ~97% |
| **single-half** | One half only, prominence ≥ 6× noise | ~87% |
| **tentative** | One half only, prominence 2–6× noise | ~68% |

Suggested use:
- **Bond assignments and conclusions:** confirmed and single-half peaks. The report's bond
  summary uses these.
- **Leads worth a second look:** tentative peaks.
- **Position:** a peak at 2400 ± 120 cm⁻¹ could match any band in 2280–2520 cm⁻¹. Check
  `position_uncertainty` before trusting a specific bond match.

On real data these figures are probably optimistic, because real noise and line shapes can
differ from the simulation.

---

## 7. Customising

| To change | Edit |
|---|---|
| Bands used for the heatmaps | `iets/bands.py` (the ranges behind the original heatmaps were not available, so these are chosen from the FTIR table; includes two sulfonate/sulfate windows) |
| Constants for α (μ₀, ε_r, t_ox, W, electrode pitch) | `iets/alpha.py` |
| FTIR bond table | `iets/ftir_bands.csv` (extracted automatically from instanano.com; worth spot-checking) |
| Analysis window (200–3648 cm⁻¹) | `GRID` in `iets/preprocess.py` (keep the length divisible by 16; retraining needed) |
| Alignment strictness | `NULL_Q` in `iets/pipeline.py` (default 95; lower aligns more spectra but admits more chance alignments). Delete `outputs/alignment.json` afterwards |
| Alignment search range | `K_RANGE`, `C_RANGE` in `iets/align.py` |
| Tentative-peak threshold | `TENTATIVE` in `evaluate.py` |
| Match tolerance for bonds | `tol` in `candidates()` / `assign()` in `iets/analysis.py` (default ±20 cm⁻¹ plus the position uncertainty) |

---

## 8. Known issues with the current data

1. **One near-duplicate pair.** In `data/`, pair `53_54` has nearly the same current in
   both device folders (max difference 0.7%; every other pair differs by ≥ 22%). It is
   flagged (`duplicated_across_devices`) and excluded from comparisons between the devices,
   which therefore use 54 pairs per device. Please check which device it belongs to.
2. **Derivative columns are per sample, not per volt** (see section 3). The loader converts
   them. Any earlier plot labelled A/V² that used the raw columns is off by about 6×10⁵ and
   is not comparable between pairs.
3. **The ±V halves are displaced.** The mirrored −V half reproduces the +V spectrum far above
   chance (`polarity_check.py`), but displaced by +96 cm⁻¹ (pure) and +216 cm⁻¹ (impure). A
   junction voltage offset V₀ gives a displacement of 2V₀. The measured zero-current offsets
   (≈4 mV, ≈60–70 cm⁻¹) explain most of the pure-device value but under a third of the
   impure one. 41 of 110 spectra could be aligned individually. Peak positions carry a
   median uncertainty of ±117 cm⁻¹ (impure) and ±35 cm⁻¹ (pure).
4. **Positive bias alone is not enough.** The `Alpha` column is filled for V > 0 only. Only
   58–62% of the peaks in the denoised +V half are reproduced in the −V half, and +V-only
   positions are biased by a median of +35 cm⁻¹ (pure) and +110 cm⁻¹ (impure). Always
   analyse both halves.
5. **Current jumps in impure pair `22_33`.** The −V current jumps abruptly at −0.354, −0.387 and
   −0.436 V (30–45× the typical step), so its −V half has |d²I/dV²| ≈ 500× the +V half above
   ~2500 cm⁻¹ and measured |α| up to 19.
6. **The data was already smoothed before analysis**, so the noise is smooth and peak-like
   (correlation half-width ≈ 56 cm⁻¹). This limits what any denoiser can do. Unsmoothed
   I–V exports would help.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` / `torch` | Use `.venv/bin/python`, not the system `python3` |
| Training suddenly very slow | The Mac went to sleep; run with `caffeinate -i` |
| `KeyError: 'tiers'` or missing `metrics.json` when running `apply.py` | Run `evaluate.py` first |
| Results do not change after replacing data files | Delete `outputs/alignment.json` (cached alignment) |
| `Cannot find module 'docx'` | `npm install` in the project folder |
| Word report shows an empty contents list | Right-click the contents list → *Update Field* |

---

## 10. How it works (short version)

- **Why a plain smoothing filter fails:** the noise is correlated over about 110 cm⁻¹ (FWHM), as
  wide as real peaks, so filtering by width removes both. What distinguishes real features
  is that they recur in both bias halves.
- **Alignment:** for each spectrum, search for the stretch `k` and shift `c` that make the
  −V half best match the +V half. Accept it only if its correlation beats the 95th percentile
  of the same search on 200 unrelated spectrum pairs. Both halves are then resampled onto a
  common, midpoint axis.
- **Network:** a 1D U-Net (5.8 M parameters) whose input is the two halves as two channels.
  It is trained on 40,000 simulated spectra: random-position peaks (deliberately *not* at
  FTIR positions), amplitudes 0.5–30× noise, some peaks in one half only, a small leftover
  misalignment, and noise taken from the unshared part of real aligned spectra.
- **Validation:** on simulated data, peak-finding F1 is 0.76 for the U-Net versus 0.61 for
  Savitzky–Golay smoothing. On real data, where there is no ground truth, each half is
  denoised on its own and the peaks are compared between halves. The U-Net is best at every
  operating point. The gain is largest at 4–5 peaks per half (+0.06–0.07 over Savitzky–Golay) and
  negligible at 2–3 peaks. See section 5 of the report.

---

## 11. Project layout

```
.
├── README.md
├── requirements.txt          Python dependencies
├── package.json              Node dependency (docx) for the Word report
├── train.py                  simulate training data, train both models
├── evaluate.py               benchmarks + confidence-tier thresholds
├── apply.py                  run on all spectra, write outputs
├── make_report_figures.py    figures and summary tables for the report
├── polarity_check.py         does the mirrored −V half carry the same spectrum?
├── alpha_analysis.py         coupling coefficient α before and after denoising -> outputs/alpha/
├── make_paper_figures.py     manuscript + SI figures -> outputs/paper_figures/
├── report/build_report.js    builds the Word report
├── iets/
│   ├── data.py               robust CSV loader, duplicate detection
│   ├── preprocess.py         branch split, detrending, normalisation
│   ├── align.py              ±V alignment and chance threshold
│   ├── pipeline.py           load → align → prepare (shared by all scripts)
│   ├── synth.py              simulated training spectra
│   ├── model.py              1D U-Net
│   ├── analysis.py           denoising, peak tiers, bond matching
│   ├── bands.py              heatmap bands
│   ├── alpha.py              K₀ and α for the 2 mm electrode geometry
│   └── ftir_bands.csv        FTIR functional-group table
├── data/                     input spectra (impure/, pure/), 55 electrode pairs each
├── matlab/                   MATLAB α script (2 mm pitch, mean voltage step)
├── models/                   trained weights (unet_2branch.pt, unet_1branch.pt), simulated test set
└── outputs/                  all results (see section 5); polarity/ for the ±V test,
                              paper_figures/ for manuscript figures
```
