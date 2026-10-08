// Builds outputs/IETS_Denoising_Report.docx from outputs/report/* (run make_report_figures.py first).
// From the project root:  NODE_PATH=<dir containing node_modules/docx> node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const R = "outputs/report";
const data = JSON.parse(fs.readFileSync(path.join(R, "report_data.json"), "utf8"));
const M = data.metrics, ST = data.stats, SY = M.synthetic, TI = M.synthetic.tiers, MT = M.matched;
const NU = data.n_unique.impure;
const POL = JSON.parse(fs.readFileSync("outputs/polarity/polarity_summary.json", "utf8"));
const PP = POL.pure.alpha_full, PI = POL.impure.alpha_full;

const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins
const FONT = "Calibri", ACCENT = "1F4E8C";

// ------------------------------------------------------------------ helpers
const run = t => (typeof t === "string" ? new TextRun(t) : t);
const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
  children: (Array.isArray(text) ? text : [text]).map(run) });
const b = t => new TextRun({ text: t, bold: true });
const it = t => new TextRun({ text: t, italics: true });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = text => p(text, { numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 264 } });
let numInst = 0;
const numbered = items => { numInst += 1; return items.map(t => p(t, { numbering: { reference: "numbers", level: 0, instance: numInst }, spacing: { after: 60, line: 264 } })); };
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

let figNo = 0, tabNo = 0;
const figRef = {}, tabRef = {};
function figure(name, caption, widthFrac = 1) {
  const buf = fs.readFileSync(path.join(R, name + ".png"));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const maxW = (CONTENT_W / 1440) * 96 * widthFrac;
  const width = Math.round(maxW), height = Math.round(width * h / w);
  figNo += 1; figRef[name] = figNo;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
        altText: { title: `Figure ${figNo}`, description: caption, name } })] }),
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18 }),
      new TextRun({ text: caption, size: 18 })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(key, headers, rows, widths, caption, fontSize = 18) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (text, head, w, shade) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "DCE6F1", type: ShadingType.CLEAR } : shade ? { fill: "F5F7FA", type: ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: head, size: fontSize })] })],
  });
  tabNo += 1; tabRef[key] = tabNo;
  return [
    new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, size: 18 }), new TextRun({ text: caption, size: 18 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((t, k) => cell(t, true, widths[k])) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, k) => cell(t, false, widths[k], ri % 2 === 1)) }))] }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}
const f2 = x => x.toFixed(2), f3 = x => x.toFixed(3), sg = x => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const pct = x => Math.round(100 * x) + "%";

// Figures and tables are numbered in order of appearance; references below rely on that order.
// Order: 1 raw, 2 alignment, 3 alignment params, 4 synthetic, 5 benchmark, 6-7 before/after,
// 8-9 heatmaps, 10 regions, 11 families, 12 spatial.
const FIG = { raw: 1, align: 2, alignp: 3, syn: 4, bench: 5, baImp: 6, baPure: 7, hmImp: 8, hmPure: 9, reg: 10, fam: 11, spat: 12 };

// ------------------------------------------------------------------ derived values
const methods = ["raw", "savitzky-golay", "gaussian", "branch-average", "unet-1branch", "unet-2branch"];
const mName = { "raw": "Raw (no filtering)", "savitzky-golay": "Savitzky–Golay (11 pt, cubic)", "gaussian": "Gaussian smoothing (σ = 2.5 pt)",
  "branch-average": "Sign-aligned ±V average", "unet-1branch": "U-Net, one half only", "unet-2branch": "U-Net, both halves (final model)" };
const synRows = methods.map(k => [mName[k], f3(SY[k].mse), f3(SY[k].f1), f3(SY[k].precision), f3(SY[k].recall), f3(SY[k].recall_large)]);
const mk = ["raw", "savitzky-golay", "gaussian", "unet-1branch"];
const matchedRows = mk.map(k => [mName[k], ...["2", "3", "4", "5", "6"].map(t => sg(MT[k][t]))]);
const ret = SY.retention;
const nAligned = ST.impure.aligned + ST.pure.aligned;

const regRows = data.regions.map(r => [r.region, r.description, `${r.impure_high} (${r.impure_any})`, `${r.pure_high} (${r.pure_any})`, `${r.dup_high} (${r.dup_any})`]);
const famRows = data.families.filter(f => f.total > 0).map(f => [
  f.family === "no match in table" ? "No match in FTIR table" : f.family,
  String(f.impure_pairs), String(f.pure_pairs), String(f.dup_pairs),
  f.impure_wavenumbers || "–", f.pure_wavenumbers || "–"]);
const reg = Object.fromEntries(data.regions.map(r => [r.region, r]));

const peakRows = (dev, dup) => data.peaks.filter(r => r.device === dev && r.tier !== "tentative" && r.duplicated_across_devices === dup)
  .map(r => [r.pair, `(${r.x}, ${r.y})`, String(Math.round(r.wavenumber)),
    r.position_uncertainty === null ? "n/a" : "±" + Math.round(r.position_uncertainty),
    r.tier === "confirmed" ? "confirmed" : r.confidence.replace("single-half ", "single "),
    f2(r.strength_nA_V2), r.best_match]);

// ------------------------------------------------------------------ content
const C = [];
C.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "Denoising IETS Spectra to Identify Bond Vibrations", bold: true, size: 46, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 400 }, children: [new TextRun({ text: "Alignment, deep-learning denoising and bond mapping across the electrode grids of two semiconductor devices", size: 26, color: "404040" })] }),
  p([b("Data: "), "data/impure and data/pure (corrected export), 55 electrode pairs each"]),
  p([b("Signal: "), "DerivativeY2_2 (numerical d²I/dV², converted from per-sample to A/V²) versus Wvnmbr, both bias polarities"]),
  p([b("Date: "), "3 October 2026 (refreshed on the corrected data)"]),
  pageBreak(),
  h1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  p(it("If the contents list is empty, right-click it in Word and choose Update Field.")),
  pageBreak(),
);

// 1 Summary
C.push(
  h1("1. Summary"),
  p("The goal was to remove noise from second-derivative (d²I/dV²) spectra, measured on a grid of electrode pairs on two devices, so that vibrational peaks can be found reliably and matched to chemical bonds using the FTIR table from instanano.com."),
  p(b("What we found about the data")),
  bullet([b("Both bias halves are needed. "), `Because Wvnmbr = 8066 × bias, each sweep holds the spectrum twice, once at +V and once (mirrored) at −V. The −V half reproduces the +V spectrum far above chance, but displaced by +${PP.best_shift_cm} cm⁻¹ (pure) and +${PI.best_shift_cm} cm⁻¹ (impure). Using +V alone (as the Alpha column does) biases peak positions and leaves about 40% of +V peaks unconfirmed (Section 8).`]),
  bullet([b("The derivative columns are per sample, not per volt. "), "They are converted to A/V and A/V² using the mean voltage step (1.25 mV in every file). Without this, amplitudes are off by about 6×10⁵."]),
  bullet([b(`${data.duplicated.length} of the 55 electrode pairs (${data.duplicated.join(", ")}) has nearly identical data in both device folders. `), "Comparisons between the devices therefore use the other " + (55 - data.duplicated.length) + " pairs (Section 2.3)."]),
  bullet([b("Much of the apparent noise is shaped like peaks. "), "The noise is about as wide as real features, so a smoothing filter cannot separate them. Some spectra consist entirely of evenly sized oscillations that do not agree between the two halves."]),
  p(b("What the method does")),
  bullet(`Aligns the two halves of each spectrum (stretch plus shift) when the match clearly beats chance: ${nAligned} of 110 spectra.`),
  bullet("Denoises both halves together with a one-dimensional U-Net neural network trained on simulated spectra, whose noise is estimated from the real data."),
  bullet(`Reports every peak with a confidence tier. On simulated data, where the truth is known, the tiers are right ${pct(TI.confirmed_precision)} (confirmed: found in both halves), ${pct(TI.single_precision)} (single-half: strong, one half only) and ${pct(TI.tentative_precision)} (tentative: moderate, one half only) of the time.`),
  p(b("How well it works")),
  bullet(`On simulated spectra the network clearly beats classical smoothing: peak-finding F1 ${f2(SY["unet-2branch"].f1)} versus ${f2(SY["savitzky-golay"].f1)} for Savitzky–Golay, and error (MSE) ${f2(SY["unet-2branch"].mse)} versus ${f2(SY["savitzky-golay"].mse)}.`),
  bullet(`On the real data the gain from the network is modest. At 4–5 peaks per half, peaks agree between the two halves ${sg(MT["unet-1branch"]["4"])} above chance, versus ${sg(MT["savitzky-golay"]["4"])} for the best classical filter. At 2–3 peaks per half, all methods are within 0.02 of each other.`),
  bullet(`${pct(ret["both-branch"].kept / ret["both-branch"].total)} of the features that are obvious in both halves of the raw data are kept after denoising. Features obvious in only one half are kept less often (${pct(ret.all.kept / ret.all.total)} of all obvious features), because the data cannot confirm them.`),
  p(b("Potential bonds (non-duplicated pairs)")),
  bullet([b("Impure device: "), `mainly high-wavenumber features. There is a peak at 2300–2500 cm⁻¹ in ${reg["2300–2500"].impure_high} of ${NU} pairs, a range with no entry in the FTIR table (adsorbed CO₂ at ≈2349 cm⁻¹ is one candidate), versus ${reg["2300–2500"].pure_high} on the pure device (Fisher test, Holm-corrected p = ${reg["2300–2500"].p_holm.toFixed(3)}). There are also peaks at 2500–3000 cm⁻¹ (carboxylic O–H, C–H, S–H stretch) in ${reg["2500–3000"].impure_high} pairs and at 3000–3650 cm⁻¹ (O–H, N–H stretch) in ${reg["3000–3650"].impure_high} pairs.`]),
  bullet([b("Pure device: "), `mainly low-wavenumber features, at 400–900 cm⁻¹ (${reg["400–900"].pure_high} pairs) and in the 900–1350 cm⁻¹ fingerprint region (C–O, C–N stretch; ${reg["900–1350"].pure_high} pairs, versus ${reg["900–1350"].impure_high} on the impure device).`]),
  bullet([b("Both devices: "), `similar numbers of pairs with C=C / C=O / N–H-bend features at 1500–1800 cm⁻¹ (${reg["1500–1800"].impure_high} impure, ${reg["1500–1800"].pure_high} pure).`]),
  bullet([b("Caution: "), `peak positions carry an uncertainty from the misalignment of the halves (median ±${Math.round(ST.impure.unc_median)} cm⁻¹ on the impure device and ±${Math.round(ST.pure.unc_median)} cm⁻¹ on the pure device), and FTIR bands overlap. Bond assignments are therefore candidates, not identifications.`]),
);

// 2 Data
C.push(
  h1("2. Data"),
  h2("2.1 Measurements and grid layout"),
  p("Each device was measured on a 5 × 4 grid of electrode nodes. Each CSV file is one bias sweep between two nodes; the file name gives the pair (11_12 is the pair between node (1,1) and node (1,2)). As in the existing heatmaps, each measurement is placed at the midpoint of its two nodes, giving X from 1 to 5 and Y from 1 to 4."),
  p("Each file holds 800 points from −0.5 V to +0.5 V. Wvnmbr equals 8066 × voltage, the conversion from electron-volts to cm⁻¹, so the axis maps bias directly to vibrational energy, as in inelastic electron tunnelling spectroscopy (IETS). In the corrected data, the derivative columns are numerical derivatives of Current with respect to the sample index (DerivativeY1 equals gradient(Current) exactly), smoothed with a Savitzky–Golay filter. In this export the Alpha column equals (d²I/dV²)/(dI/dV) up to a constant and is filled for V > 0 only. The newer α = (d²I/dV²)/(2K₀) for the 2 mm electrode geometry is produced by alpha_analysis.py (outputs/alpha/)."),
  h2("2.2 Data-quality issues handled automatically"),
  bullet("Column headers are spelled more than a dozen ways (wvnmbr, Wvnmbr, wvnm, vnmbr; Current, current, currrent, CUrrent …). Columns are read by position."),
  bullet("Derivatives are per sample. They are divided by the voltage step averaged over 40 samples (and its square) to give A/V and A/V². The mean step is 1.25 mV in every file; in ten impure files (row 1) the voltage is recorded on a 1 mV grid, so single steps are 1 or 2 mV and a median step (1.0 mV) would overstate d²I/dV² and α by 1.56×."),
  bullet("Everything is resampled onto a uniform 8 cm⁻¹ grid."),
  bullet("Large artefacts at |wavenumber| > ≈3650 cm⁻¹ (the ends of the sweep) and the zero-bias region below 200 cm⁻¹ are excluded, leaving an analysis window of 200–3648 cm⁻¹."),
  bullet("Amplitudes vary about 100-fold between pairs, so each spectrum is normalised by a robust noise estimate and restored to A/V² afterwards."),
  ...figure("fig01_raw_examples", "Raw DerivativeY2_2 for four pairs. Grey regions are excluded; the sweep-edge artefacts would otherwise dominate the scale."),
  h2("2.3 Nearly identical data in both device folders"),
  p(["The earlier export had 22 pairs with identical data in both folders. In the corrected export, all 55 impure files and 6 pure files were replaced. One pair remains suspicious: ", it(data.duplicated.join(", ")), " has currents within 0.7% of each other in the two folders, whereas every other pair differs by at least 22%. Until its origin is confirmed:"]),
  bullet(`All comparisons between the devices in this report use only the ${NU} non-duplicated pairs per device.`),
  bullet("Duplicated pairs are still denoised and shown, marked with a square in the spatial maps and flagged in all tables and CSV files."),
);

// 3 The problem
C.push(
  h1("3. Why the noise is hard to remove"),
  p("Three properties of these spectra rule out a simple smoothing filter."),
  ...numbered([
    [b("The noise looks like peaks. "), "The data was already filtered before we received it, so the remaining noise is smooth. It is correlated over ≈110 cm⁻¹ (full width at half maximum), about the width of real features. Removing it by frequency would remove the peaks too."],
    [b("The two bias halves are misaligned. "), `In an ideal junction the −V half would mirror the +V half. In these data the halves share features, but at shifted and stretched positions. Figure ${FIG.align} shows a clear case where one feature appears at clearly different positions in the two halves. The mapping between halves (−V at w matches +V at k·w + c) varies from spectrum to spectrum (Figure ${FIG.alignp}). The median shift is ${Math.round(ST.impure.c_median)} cm⁻¹ on the impure device and ${Math.round(ST.pure.c_median)} cm⁻¹ on the pure device, with little stretch (median k ≈ ${f2(ST.impure.k_median)}). A junction voltage offset V₀ would give a shift of 2V₀. The measured zero-current offsets (≈4 mV, ≈60–70 cm⁻¹) explain most of the pure-device shift but under a third of the impure one (Section 8).`],
    [b("Some spectra are made of evenly sized oscillations. "), "In several pairs (for example pure 11_12 and impure 13_22) the whole trace is a series of similar-sized wiggles that do not agree between the halves, even after the best alignment. The data cannot tell whether any one of them is real."],
  ]),
  p("Signs are also inconsistent: after alignment, the halves have the same sign (even symmetry) more often than the opposite sign expected for an ideal junction. The method therefore compares peak positions and ignores sign."),
  ...figure("fig02_alignment", "Top: impure pair 14_23 as measured; the same large feature appears at different wavenumbers in the two halves. Bottom left: after alignment onto a common axis. Bottom right: the best alignment correlation for each spectrum (blue) versus the best achievable between unrelated spectra (grey, chance). Alignments right of the dashed line (95th percentile of chance) are accepted."),
  ...figure("fig03_alignment_params", "Fitted stretch k and shift c for the accepted alignments.", 0.6),
);

// 4 Method
C.push(
  h1("4. Method"),
  h2("4.1 Preprocessing and alignment"),
  ...numbered([
    "Split each sweep into its +V and −V halves, mirror the −V half onto positive wavenumbers, and resample both onto a uniform grid from 200 to 3648 cm⁻¹ in 8 cm⁻¹ steps (432 points).",
    "Remove the slowly varying background from each half with a robust (Huber-weighted) cubic polynomial fit.",
    "Search for the mapping between the halves (stretch k = 0.85–1.15, shift c = ±400 cm⁻¹, either sign) that maximises their correlation.",
    `Accept the mapping only if its correlation beats the 95th percentile of the same search run on 200 randomly paired, unrelated spectra (|r| > ${f2(data.threshold)}), and it is not at the edge of the search range. ${nAligned} of 110 spectra pass; by chance alone about 5–6 would. Spectra that fail are kept as measured and flagged as not aligned.`,
    "Resample both halves onto a common midpoint axis, halfway between each half's own axis, since there is no way to tell which polarity is correctly calibrated. Half the distance between the two halves' positions is reported as the position uncertainty.",
    "Divide by a robust noise scale (1.4826 × median absolute deviation), which is stored so results can be converted back to A/V².",
  ]),
  h2("4.2 Training data"),
  p("No noise-free reference spectra exist, so the network was trained on 40,000 simulated two-half spectra:"),
  bullet([b("Features: "), "0–8 per spectrum (10% contain none). They have Gaussian or dispersive (derivative-of-Gaussian) shapes, widths σ = 12–70 cm⁻¹, and amplitudes of 0.5–30 × the noise, matching the real range (the largest real features are typically about 8× the noise and up to about 30×). Positions are uniform across the window, not taken from the FTIR table, so the network cannot learn to invent peaks at expected bond positions."]),
  bullet([b("Coupling between halves: "), "80% of features appear in both halves, with a random amplitude ratio and odd, even or mixed sign. 10% appear only in the +V half and 10% only in the −V half; the target keeps them. The −V half carries a small leftover misalignment (shift σ = 12 cm⁻¹, stretch σ = 2%), as after the real alignment step."]),
  bullet([b("Noise: "), "estimated from the real aligned spectra as the part of one half not shared with the other, then phase-randomised. This keeps the real noise's correlation length without copying any real features."]),
  h2("4.3 Network and training"),
  p("The network is a one-dimensional U-Net: an encoder–decoder convolutional network with skip connections, a standard architecture for denoising. It has 4 down-sampling levels with 32 to 512 channels, kernel size 7, GroupNorm and GELU activations, and 5.8 million parameters. The input is the two aligned halves as two channels, and the output is the two denoised halves. At inference, results are averaged over sign-flipped and half-swapped copies of the input."),
  p("Training used AdamW, a one-cycle learning-rate schedule (peak 2 × 10⁻³), batch size 128, 8 epochs and a Huber loss, with random sign flips and half swaps. The Huber loss keeps the largest peaks from dominating training. A second, one-half model was trained the same way and is used only for the real-data test in Section 5.2."),
  h2("4.4 Peak detection and confidence tiers"),
  p(`Peaks (maxima and minima) are found in each denoised half with a minimum separation of 40 cm⁻¹. Each peak is then placed in a tier:`),
  bullet([b("Confirmed: "), `found in both halves within 32 cm⁻¹, with prominence ≥ ${f2(TI.threshold)} noise units in each.`]),
  bullet([b("Single-half: "), `found in one half only, with prominence ≥ ${TI.single_threshold.toFixed(1)} noise units. This threshold was chosen to give about 90% precision on simulated data.`]),
  bullet([b("Tentative: "), `found in one half only, with prominence between ${TI.tentative_threshold.toFixed(1)} and ${TI.single_threshold.toFixed(1)} noise units. These are shown in plots and listed in peaks.csv, but are kept separate from the bond summary.`]),
  h2("4.5 Bond assignment"),
  p("Each peak is matched against the 104 FTIR bands from instanano.com. Because bands overlap and positions carry an uncertainty, results are given at two levels. First, by spectral region (for example 1500–1800 cm⁻¹: C=C, C=O and N–H bending), which is robust to both problems. Second, by the best-matching single band, the narrowest band containing the peak; every band within the position uncertainty is listed in peak_bond_assignments.csv."),
);

// 5 Validation
C.push(
  h1("5. Validation"),
  h2("5.1 Simulated test set (ground truth known)"),
  p("The simulated test set has 1,000 held-out spectra, and both halves are scored. For fairness, each method's detection threshold was tuned on a separate set of 1,000 spectra. A predicted peak is correct if it lies within 24 cm⁻¹ of a true peak."),
  ...table("syn", ["Method", "MSE", "F1", "Precision", "Recall", "Recall ≥5× noise"], synRows, [3438, 1200, 1200, 1200, 1200, 1400],
    "Performance on simulated spectra (MSE in normalised units; the noisy input has MSE ≈ 0.87)."),
  ...table("tier", ["Tier", "Rule", "Precision", "Peaks in test set"], [
    ["Confirmed", "both halves", pct(TI.confirmed_precision), String(TI.n_confirmed)],
    ["Single-half", `one half, ≥ ${TI.single_threshold.toFixed(1)}× noise`, pct(TI.single_precision), String(TI.n_single)],
    ["Tentative", `one half, ${TI.tentative_threshold.toFixed(1)}–${TI.single_threshold.toFixed(1)}× noise`, pct(TI.tentative_precision), String(TI.n_tentative)],
  ], [2000, 3638, 2000, 2000], "Precision of each confidence tier on simulated data."),
  p("The network clearly beats classical filters on simulated data and almost never misses large features (recall above 99% for features ≥ 5× the noise). Using both halves helps a little over using one."),
  ...figure("fig04_synthetic_example", "Two held-out simulated spectra: Savitzky–Golay smoothing (gold) keeps the noise wiggles, while the U-Net (blue) follows the true signal (thick grey)."),
  h2("5.2 Real data (no ground truth)"),
  p(`For a test on real data, each half of the ${nAligned} aligned spectra was denoised on its own with the one-half model, so the model never saw the other half. Peaks were then compared between halves. Noise is independent between halves, so a better method finds more peaks that agree. Chance agreement was measured with randomly paired spectra. Because methods report different numbers of peaks, the comparison is made at matched operating points (the same number of peaks per half).`),
  ...table("matched", ["Method", "2 peaks", "3 peaks", "4 peaks", "5 peaks", "6 peaks"], matchedRows, [3638, 1200, 1200, 1200, 1200, 1200],
    "Agreement of peaks between independently denoised halves, above chance, at a matched number of peaks per half."),
  p([b("Interpretation. "), `Aligning the halves is what makes the real data agree: after alignment, even raw peaks agree ${sg(MT.raw["3"])} above chance. The network gives the highest agreement at every operating point. The margin is largest at 4–5 peaks per half (where most spectra sit) and negligible at 2–3 peaks. Of the features obvious in both halves of the raw data (≥4× noise), ${ret["both-branch"].kept} of ${ret["both-branch"].total} (${pct(ret["both-branch"].kept / ret["both-branch"].total)}) are kept. Counting features obvious in only one half as well, ${ret.all.kept} of ${ret.all.total} (${pct(ret.all.kept / ret.all.total)}) are kept. Most dropped features are only about 3× the noise, and most come from spectra that could not be aligned.`]),
  ...figure("fig05_benchmark", "Left: peak-finding F1 on simulated data. Right: agreement between halves on real data at matched operating points."),
);

// 6 Before/after
C.push(
  h1("6. Results: before and after denoising"),
  p(`Figures ${FIG.baImp} and ${FIG.baPure} show raw (left) and denoised (right) spectra on the common axis. Each device shows its three pairs with the strongest confirmed peaks, one pair with a strong single-half peak, and one pair where the output is nearly flat. In aligned pairs the main features line up between the halves and are kept, while smaller wiggles that disagree between the halves are removed.`),
  p([b("Nearly flat outputs. "), `The bottom row of each figure shows a pair whose denoised output is nearly flat. Its raw fluctuations are of similar size throughout and do not agree between the two halves, so the model treats them as noise. This is a judgement the data cannot settle either way; the raw traces are kept in the output files for anyone who wants to examine such spectra.`]),
  ...figure("fig06_before_after_impure", "Impure device, before (left) and after (right) denoising. Blue: +V half; red: mirrored −V half. Lines: solid = confirmed, dashed = single-half, dotted = tentative."),
  ...figure("fig06_before_after_pure", "Pure device, before (left) and after (right) denoising."),
  p(`Figures ${FIG.hmImp} and ${FIG.hmPure} compare band maps before and after denoising. Because signs are not always consistent between halves, these maps show magnitude (signed maps restricted to sign-reproducible pairs are in the manuscript, Fig. 4): the strongest absolute signal in the band, averaged over the two halves. Contours are interpolated between measurement points (dots) and should not be read at finer resolution than the grid.`),
  ...figure("fig07_heatmaps_impure", "Impure device band maps, before (left) and after (right) denoising: C=C stretch, isothiocyanate and H-bonded O–H."),
  ...figure("fig07_heatmaps_pure", "Pure device band maps, before (left) and after (right) denoising."),
  p("Full plots for all 110 spectra are in outputs/spectra_impure.pdf and outputs/spectra_pure.pdf, and maps for all seven heatmap bands are in outputs/heatmaps_impure.pdf and outputs/heatmaps_pure.pdf."),
);

// 7 Bonds
C.push(
  h1("7. Potential bonds across the grids"),
  h2("7.1 By spectral region"),
  p(`Table 4 counts, per device, how many of the ${NU} non-duplicated pairs have a peak in each spectral region. The first number counts high-confidence peaks (confirmed or single-half); the number in brackets also includes tentative peaks. The duplicated pairs are counted separately.`),
  ...table("reg", ["Region (cm⁻¹)", "Candidate vibrations", `Impure (${NU})`, `Pure (${NU})`, `Duplicated (${data.duplicated.length})`],
    regRows, [1300, 4838, 1150, 1150, 1200], "Pairs with a peak in each spectral region: high-confidence count, with the count including tentative peaks in brackets."),
  ...figure("fig08_regions", "Share of non-duplicated pairs with a peak in each region. Solid bars: confirmed or single-half peaks; pale bars: including tentative peaks."),
  p(b("Main differences between the devices")),
  bullet([b("2300–2500 cm⁻¹ (impure). "), `High-confidence peaks in ${reg["2300–2500"].impure_high} of ${NU} impure pairs and ${reg["2300–2500"].pure_high} pure pair (the only region difference that survives Holm correction, p = ${reg["2300–2500"].p_holm.toFixed(3)}). The FTIR table has no band here: the nearest are C≡N and C≡C stretching (2190–2260 cm⁻¹) and the broad carboxylic O–H band from 2500 cm⁻¹. Candidates worth testing are adsorbed or trapped CO₂ (asymmetric stretch ≈2349 cm⁻¹), P–H or Si–H stretching (≈2280–2440 cm⁻¹), and B–H. Impure-device positions carry about ±${Math.round(ST.impure.unc_median)} cm⁻¹ of uncertainty from the alignment, so the neighbouring C≡N / C≡C and O–H assignments cannot be ruled out.`]),
  bullet([b("2500–3650 cm⁻¹ (mostly impure). "), `Stretching vibrations of O–H, C–H, N–H and S–H: ${reg["2500–3000"].impure_high + reg["3000–3650"].impure_high} region-hits on the impure device versus ${reg["2500–3000"].pure_high + reg["3000–3650"].pure_high} on the pure device. This fits an impure surface carrying hydroxyl, carboxylic or organic contamination.`]),
  bullet([b("400–1350 cm⁻¹ (mostly pure). "), `Metal–oxygen, C–O, C–N and C=C bending and fingerprint modes: ${reg["400–900"].pure_high} and ${reg["900–1350"].pure_high} pure pairs, versus ${reg["400–900"].impure_high} and ${reg["900–1350"].impure_high} impure pairs.`]),
  bullet([b("1500–1800 cm⁻¹ (both). "), `C=C and C=O stretching and N–H bending appear on both devices at similar rates (${reg["1500–1800"].impure_high} impure, ${reg["1500–1800"].pure_high} pure pairs).`]),
  h2("7.2 By best-matching bond"),
  p(`Table 5 gives the best-matching FTIR vibration for every high-confidence peak. Bands overlap and positions are uncertain, so one peak often fits several vibrations. The counts show which bonds are compatible with the data, not which are confirmed.`),
  ...table("fam", ["Best-matching vibration", "Impure", "Pure", "Dupl.", "Impure positions (cm⁻¹)", "Pure positions (cm⁻¹)"], famRows,
    [1900, 700, 700, 700, 2819, 2819], "Pairs with a high-confidence peak whose best FTIR match is each vibration (non-duplicated pairs per device; duplicated pairs counted once).", 15),
  ...figure("fig09_bond_families", "Best-matching FTIR vibration for high-confidence peaks, non-duplicated pairs.", 0.72),
  h2("7.3 Across the grid"),
  ...figure("fig10_spatial_peaks", "High-confidence peaks at each electrode-pair midpoint, coloured by wavenumber. The square marks the pair whose data is nearly identical in both device folders."),
  bullet([b("Impure device: "), "high-wavenumber features (2300–3600 cm⁻¹, green–yellow) concentrate in the centre of the grid (X ≈ 2.5–3.5, Y ≈ 1–3.5). The left edge (X ≈ 1–2, Y ≈ 3.5–4) carries features at ≈2000–2500 cm⁻¹, and the right-hand column (X ≈ 4.5) mostly low-wavenumber features."]),
  bullet([b("Pure device: "), "low-wavenumber features below 1500 cm⁻¹ (purple–blue) occur across most of the grid. High-wavenumber features above 2500 cm⁻¹ are fewer and scattered."]),
  bullet([b("Near-duplicate pair: "), "the square (53_54, X = 5, Y = 3.5) is nearly the same measurement in both folders and is excluded from device comparisons."]),
);

// 8 Revision note and limitations
C.push(
  h1("8. Are the negative-voltage readings needed?"),
  p("The Alpha column ((d²I/dV²)/(dI/dV)) was computed for positive voltages only, on the view that only positive bias carries the bond information. We tested this directly on both d²I/dV² and the normalised signal."),
  bullet([b("The −V half carries the same spectrum. "), `Averaged over the 55 pairs of a device, the +V half and the mirrored −V half of the same pair correlate at r = ${PP.mean_r_at_shift.toFixed(2)} (pure) and ${PI.mean_r_at_shift.toFixed(2)} (impure). For unrelated pairs the corresponding value is at most ${PP.chance_max_abs_r_99.toFixed(2)} (p < 0.001). This also holds above 800 cm⁻¹, away from the zero-bias region. Physically this is expected: a process with energy ħω switches on at |eV| = ħω for either polarity.`]),
  bullet([b("But it is displaced. "), `Best agreement needs the −V half to be shifted by +${PP.best_shift_cm} cm⁻¹ (pure) and +${PI.best_shift_cm} cm⁻¹ (impure). A voltage offset V₀ gives a shift of 2V₀; the measured zero-current offsets (≈4 mV) account for most of the pure-device shift but not the impure one. Most features have the same sign in both halves (even symmetry).`]),
  bullet([b("So the student is half right. "), "The −V half does not contain different bonds. However, it is needed to (i) confirm a peak: only 58–62% of +V peaks are reproduced at −V; (ii) correct positions: +V-only positions sit a median 35 cm⁻¹ (pure) and 110 cm⁻¹ (impure) above the two-polarity midpoint, enough to change the FTIR match; and (iii) roughly double the data for denoising. Alpha should be computed for both polarities."]),
  ...figure("fig13_polarity", "a, One full sweep. b, Normalised signal at +V and mirrored −V, as measured and after displacement. c, Mean ±V correlation versus displacement, same pair (lines) versus unrelated pairs (99% band). d, Best displacement per pair."),
  h1("9. Limitations and next steps"),
  h2("9.1 Data refresh note"),
  p("This report was regenerated on the corrected data in data/. Results from an earlier export (22 duplicated pairs; all impure files since replaced) should be discarded."),
  h2("9.2 Limitations"),
  bullet([b("Modest real-data gain from the network. "), "Most of the improvement comes from aligning the halves; the network adds a modest gain in the useful range and does worse than simple smoothing in some settings."]),
  bullet([b("Polarity displacement. "), "Partly explained by a ≈4 mV voltage offset; the rest is unknown. Neither polarity's axis can be trusted absolutely, hence the midpoint axis and the position uncertainty."]),
  bullet([b("Unaligned spectra. "), `For the ${110 - nAligned} spectra whose halves could not be aligned, peaks cannot be confirmed; only strong single-half peaks are reported with high confidence.`]),
  bullet([b("Near-duplicate data. "), `${data.duplicated.join(", ")} is nearly identical in both device folders.`]),
  bullet([b("IETS versus FTIR. "), "Selection rules and intensities differ, and vibrational energies can shift when molecules bond to a surface. FTIR matches are candidates, not identifications."]),
  bullet([b("Inputs chosen by us. "), "The heatmap band ranges (iets/bands.py) were chosen from the FTIR table because the ranges behind the original heatmaps were not available. The FTIR table was extracted automatically from instanano.com and should be spot-checked."]),
  h2("9.3 Recommended next steps"),
  ...numbered([
    "Confirm which device pair 53_54 belongs to, and replace the other file if a separate measurement exists.",
    "Record forward and reverse sweeps for a few pairs and measure the zero-bias offset, to explain the ±V displacement. A correction at source would remove the largest uncertainty in this analysis.",
    "Repeat the sweep on a few pairs. Two independent sweeps of the same pair would allow a direct test of reproducibility and a stronger training signal.",
    "Measure a reference sample with a known vibrational spectrum to calibrate the wavenumber axis and check the bond mapping.",
    "Check the impure-device 2300–2500 cm⁻¹ feature with FTIR or Raman spectroscopy on the same sample.",
  ]),
);

// 9 Files
C.push(
  h1("10. Files and reproducibility"),
  ...table("files", ["File / folder", "Contents"], [
    ["README.md", "How to install, run and read the outputs"],
    ["iets/", "Python package: loader, preprocessing, alignment, simulation, U-Net, analysis, band definitions"],
    ["train.py", "Builds the simulated data and trains both models (about 30 minutes on an Apple-silicon GPU)"],
    ["evaluate.py", "Benchmarks and confidence-tier calibration → outputs/metrics.json"],
    ["apply.py", "Alignment, denoising and peaks for all spectra → denoised CSVs, peaks.csv, spectra and heatmap PDFs"],
    ["make_report_figures.py", "Figures and summary tables for this report → outputs/report/"],
    ["polarity_check.py", "±V test: does the mirrored −V half carry the same spectrum? → outputs/polarity/"],
    ["make_paper_figures.py", "Manuscript and SI figures → outputs/paper_figures/"],
    ["outputs/alignment.json", "Per-spectrum alignment (k, c, correlation, accepted) and the chance distribution"],
    ["outputs/peaks.csv", "Every peak: position on the common and on each half's axis, uncertainty, tier, strength, candidate bonds"],
    ["outputs/report/peak_bond_assignments.csv", "Every peak with its best-matching band and all bonds within its uncertainty"],
    ["outputs/report/region_summary.csv, bond_family_summary.csv", "Tables 4 and 5 in machine-readable form"],
  ], [3900, 5738], "Project files."),
);

// Appendix
C.push(
  pageBreak(),
  h1("Appendix A. High-confidence peaks by electrode pair"),
  p("Confirmed and single-half peaks only; tentative peaks are listed in outputs/peaks.csv. Position is on the common axis, and ± is the uncertainty from the alignment (n/a where the halves could not be aligned). Strength is in nA/V² (10⁻⁹ A/V²). The best match is the narrowest FTIR band containing the position."),
  h2("A.1 Impure device (non-duplicated pairs)"),
  ...table("a1", ["Pair", "(X, Y)", "cm⁻¹", "±", "Tier", "nA/V²", "Best FTIR match"], peakRows("impure", false), [700, 850, 650, 600, 1000, 750, 5088], "Impure device.", 14),
  h2("A.2 Pure device (non-duplicated pairs)"),
  ...table("a2", ["Pair", "(X, Y)", "cm⁻¹", "±", "Tier", "nA/V²", "Best FTIR match"], peakRows("pure", false), [700, 850, 650, 600, 1000, 750, 5088], "Pure device.", 14),
  h2("A.3 Near-duplicate pair"),
  ...table("a3", ["Pair", "(X, Y)", "cm⁻¹", "±", "Tier", "nA/V²", "Best FTIR match"], peakRows("impure", true), [700, 850, 650, 600, 1000, 750, 5088], "Near-duplicate pair (impure-folder copy shown).", 14),
);

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "IETS denoising project", title: "Denoising IETS Spectra to Identify Bond Vibrations",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: ACCENT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "333333" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
    children: C,
  }],
});

// sanity check: hard-coded figure numbers must match the order figures were added
const expected = { fig02_alignment: FIG.align, fig03_alignment_params: FIG.alignp, fig06_before_after_impure: FIG.baImp,
  fig06_before_after_pure: FIG.baPure, fig07_heatmaps_impure: FIG.hmImp, fig07_heatmaps_pure: FIG.hmPure };
for (const [k, v] of Object.entries(expected)) if (figRef[k] !== v) throw new Error(`figure ${k} is ${figRef[k]}, text says ${v}`);
if (tabRef.reg !== 4 || tabRef.fam !== 5) throw new Error(`table numbering changed: reg=${tabRef.reg} fam=${tabRef.fam}`);

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("outputs/IETS_Denoising_Report.docx", buf);
  console.log(`wrote outputs/IETS_Denoising_Report.docx (${figNo} figures, ${tabNo} tables)`);
});
