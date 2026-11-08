"""Render verified Large/native outcomes and same-cohort comparisons offline.

No images, model calls or new operating-point selection. ReportLab is imported
only for rendering; numerical table helpers remain usable in the CPU test suite.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from render_precise_shared_report import percentage

ARMS = ("frozen_dinov2", "adapted_dinov2")
HANDS = ("left", "right", "thumbs")


def operating_rows(data):
    """Keep attempted denominators, failure policies and binary provenance clear."""
    rows = []
    for key, label in (("prior_small_same_cohort", "DINOv2-small / 448"), ("dino", "DINOv2-Large / native")):
        for arm in ARMS:
            item = data[key][arm]
            for group, summary in [("pooled", item["results"]["pooled"]),
                                   *[(h, item["results"]["by_slap_type"][h]) for h in HANDS]]:
                rates = summary["all_attempts_failure_as_reject"]
                tp, fp = rates["genuine_accepts"], rates["false_accepts"]
                g, i = rates["genuine_comparisons"], rates["impostor_comparisons"]
                tn, fn = i - fp, g - tp
                rows.append({"method": label + " / " + arm.split("_")[0], "population": group,
                    "TP": tp, "TN": tn, "FP": fp, "FN": fn, "attempts": g + i,
                    "accuracy": (tp + tn) / (g + i), "tar": rates["tar"], "far": rates["far"],
                    "failures": sum(summary["failures"].values()), "threshold": item["threshold"]["cutoff"],
                    "policy": "training_impostors_nominal_1pct_FAR_failure_as_reject"})
    for kind in ("base", "finetuned"):
        item = data["qwen"][kind]
        for group, summary in [("pooled", item["operating_point"]),
                               *[(h, item["by_slap_type"][h]) for h in HANDS]]:
            rates = summary["all_attempts_failure_as_reject"]
            tp, fp = rates["genuine_accepts"], rates["false_accepts"]
            g, i = rates["genuine_comparisons"], rates["impostor_comparisons"]
            # The collaborator's all-attempt accuracy counts INVALID/ERROR as
            # incorrect for both classes. Never credit a failed impostor as TN.
            tn = i - summary["failures"]["impostor"] - fp
            fn = g - summary["failures"]["genuine"] - tp
            accuracy = (tp + tn) / (g + i)
            if group == "pooled" and (item["confusion_matrix"] != {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
                                      or item["accuracy_all_attempts"] != accuracy):
                raise ValueError("Qwen matrix and all-attempt accuracy differ from source review.")
            rows.append({"method": "Qwen3-VL-8B / supplied " + kind, "population": group,
                "TP": tp, "TN": tn, "FP": fp, "FN": fn, "attempts": g + i, "accuracy": accuracy,
                "tar": rates["tar"], "far": rates["far"], "failures": sum(summary["failures"].values()),
                "threshold": None, "policy": "native_AB_failures_incorrect_run_provenance_incomplete"})
    return rows


def continuous_curves(data):
    curves = []
    for key, label, colors in (("prior_small_same_cohort", "Small/448", ("#7396a6", "#c5a17b")),
                               ("dino", "Large/native", ("#12617b", "#c45422"))):
        for arm, color in zip(ARMS, colors, strict=True):
            summary = data[key][arm]["results"]["pooled"]
            curves.append((label + " " + arm.split("_")[0], summary["descriptive_valid_score_discrimination"],
                           sum(summary["attempts"].values()) - sum(summary["failures"].values()), color))
    for arm, label, color in (("verifinger_fusion", "VeriFinger fused", "#497440"),
                              ("verifinger_sum", "VeriFinger sum", "#785690")):
        summary = data["sdk_descriptive"][arm]["pooled"]
        curves.append((label, summary["discrimination"], summary["valid"], color))
    return curves


def make_roc(data, font):
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, Line, String
    from reportlab.lib import colors

    drawing = Drawing(528, 310)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 46, 80, 458, 202
    active = [(name, curve, color) for name, curve, _, color in continuous_curves(data) if curve]
    plot.data = [[(point["far"], point["tar"]) for point in curve["roc_det_rates"]] for _, curve, _ in active]
    plot.joinedLines = 1
    for axis in (plot.xValueAxis, plot.yValueAxis):
        axis.valueMin, axis.valueMax, axis.valueStep = 0, 1, .2
        axis.labels.fontName, axis.labels.fontSize = font, 8
        axis.visibleGrid = True
        axis.gridStrokeColor = colors.HexColor("#e5e7eb")
        axis.gridStrokeWidth = .4
    for index, (name, _, color) in enumerate(active):
        plot.lines[index].strokeColor = colors.HexColor(color)
        plot.lines[index].strokeWidth = 1.4
        if name.startswith("Small"):
            plot.lines[index].strokeDashArray = [3, 2]
    drawing.add(Line(46, 80, 504, 282, strokeColor=colors.HexColor("#cdd4d9"), strokeDashArray=[2, 3]))
    drawing.add(plot)
    drawing.add(String(275, 54, "False accept rate (valid scores)", textAnchor="middle", fontName=font, fontSize=9))
    drawing.add(String(46, 298, "True accept rate", fontName=font, fontSize=9))
    for index, (name, _, color) in enumerate(active):
        x, y = (46 if index % 2 == 0 else 300), 36 - 14 * (index // 2)
        drawing.add(Line(x, y + 2, x + 17, y + 2, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x + 23, y, name, fontName=font, fontSize=8))
    return drawing


def render_report(data, output, aggregate_digest):
    from reportlab.graphics import renderSVG
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    if (data.get("scope") != "native_large_numeric_group_holdout_with_source_verified_qwen_limits"
            or data.get("model_profile") != "dinov2_large_native_v1"
            or data.get("binary_qwen_roc_auc_eer") is not None):
        raise ValueError("Expected the verified native-large report scope with binary Qwen limits.")
    rows = operating_rows(data)
    output = Path(output)
    paths = {key: output / filename for key, filename in {
        "pdf": "professor-report.pdf", "markdown": "professor-report.md", "csv": "operating_points.csv",
        "roc": "pooled_roc.svg"}.items()}
    if any(path.exists() for path in paths.values()):
        raise ValueError("Native report rendering never overwrites artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    font, bold = "Helvetica", "Helvetica-Bold"
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    if all((fonts / p).exists() for p in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")):
        for name, filename in (("PreciseSans", "DejaVuSans.ttf"), ("PreciseSans-Bold", "DejaVuSans-Bold.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
        pdfmetrics.registerFontFamily("PreciseSans", normal="PreciseSans", bold="PreciseSans-Bold")
        font, bold = "PreciseSans", "PreciseSans-Bold"
    styles = getSampleStyleSheet()
    for name, size, leading, color in (("TitleNative", 24, 29, "#173b4b"), ("HeadingNative", 14, 18, "#173b4b"),
                                       ("BodyNative", 10, 14, "#243243"), ("SmallNative", 8, 11, "#54616d"),
                                       ("CellNative", 8, 10.5, "#243243")):
        styles.add(ParagraphStyle(name, fontName=bold if name in {"TitleNative", "HeadingNative"} else font,
                                  fontSize=size, leading=leading, textColor=colors.HexColor(color), spaceAfter=8))
    story, markdown = [], []

    def para(text, small=False):
        story.append(Paragraph(text, styles["SmallNative" if small else "BodyNative"]))
        markdown.append(text.replace("<b>", "**").replace("</b>", "**"))

    def heading(text):
        story.append(Paragraph(escape(text), styles["HeadingNative"]))
        markdown.append("## " + text)

    def table(headers, values, widths):
        t = Table([[Paragraph(escape(str(cell)), styles["CellNative"]) for cell in row]
                   for row in (headers, *values)], colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#9ab0ba")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.extend([t, Spacer(1, 8)])
        markdown.append("\n".join(["| " + " | ".join(map(str, headers)) + " |",
                                   "| " + " | ".join("---" for _ in headers) + " |",
                                   *["| " + " | ".join(map(str, row)) + " |" for row in values]]))

    synthetic = data.get("_synthetic", False)
    title = "SYNTHETIC LAYOUT TEST" if synthetic else "Precise: DINOv2-Large"
    story.append(Paragraph(title, styles["TitleNative"]))
    markdown.append("# " + title)
    para("Bibesh Pyakurel | Original-resolution evaluation | " + escape(data["study_date"]), small=True)
    if synthetic:
        para("<b>Synthetic layout fixtures only. These are not Large/native research results.</b>")
    para(f"Frozen and adapted DINOv2-Large use all {data['training_pairs']:,} supplied training pairs and "
         f"the same {data['testing_pairs']:,} testing attempts as the completed small-model and supplied Qwen results. "
         "The test set has 2,100 genuine and 2,100 impostor pairs, including 700 of each class per left, right and two-thumb type.")
    primary = [row for row in rows if row["population"] == "pooled"]
    heading("Actual accuracy before and after adaptation")
    table(["Method / checkpoint", "Accuracy", "TAR", "FAR"],
          [[r["method"], percentage(r["accuracy"]), percentage(r["tar"]), percentage(r["far"])] for r in primary],
          [264, 88, 88, 88])
    frozen, adapted = [r for r in primary if r["method"].startswith("DINOv2-Large")]
    para("Large/native adapted-minus-frozen changes: "
         f"accuracy {100 * (adapted['accuracy'] - frozen['accuracy']):+.2f}, "
         f"TAR {100 * (adapted['tar'] - frozen['tar']):+.2f}, "
         f"FAR {100 * (adapted['far'] - frozen['far']):+.2f} percentage points.", small=True)
    para("Accuracy = (TP + TN) / all attempts. TAR is the accepted fraction of genuine pairs; FAR is the accepted "
         "fraction of impostor pairs. With balanced classes and zero failures, standard and balanced accuracy "
         "coincide; failure counts and policies are shown below.", small=True)
    heading("Confusion matrices on supplied labels")
    table(["Method", "TP", "TN", "FP", "FN", "Failures"],
          [[r["method"], r["TP"], r["TN"], r["FP"], r["FN"], r["failures"]] for r in primary], [218, 62, 62, 62, 62, 62])
    para("DINO uses an arm-specific cutoff selected at nominal 1% training-impostor FAR, with failures rejected. "
         "Qwen uses native A/B answers; source accuracy counts invalid/errors as incorrect. A rejected failed impostor "
         "is not evidence of a successful biometric comparison. These are different operating policies, <b>not matched test FAR</b>.", small=True)

    story.append(PageBreak())
    heading("Larger model, original pixels and measured training time")
    cfg = data["configuration"]
    elapsed = data["training_elapsed_seconds"]
    hours, remainder = divmod(int(elapsed), 3600);minutes, seconds = divmod(remainder, 60)
    para("The new model is <b>facebook/dinov2-large: 304,368,640 parameters</b>. Only blocks 22/23 and the final "
         "layer norm are adapted (25,198,592 parameters). Both arms start from the same pinned pretrained base; "
         "neither prior task checkpoint initializes this training.")
    para("All 6,778 benchmark images are 1600 x 1500 pixels. Each entire grayscale raster is centered on a white "
         "1610 x 1512 canvas: 5 pixels on each horizontal side and 6 on each vertical side. RGB replication and "
         "normalization follow. <b>There is no resampling, cropping or stretching, and original files are unchanged.</b> "
         "The model receives 12,420 patches of 14 x 14 pixels; its positional embeddings are interpolated to this grid.")
    para(f"The fresh adaptation took <b>{hours}h {minutes}m {seconds}s</b> from its recorded attempt start to final "
         "checkpoint selection. This elapsed time includes the training pipeline and checkpoints; it excludes the "
         "separate frozen/adapted calibration and final-test phases.")
    table(["Fixed item", "Executed setting"], [
        ["Schedule / selection", f"{cfg['epochs']} complete epochs; final epoch chosen before execution"],
        ["Pairs / optimizer updates", f"{data['training_pairs'] * cfg['epochs']:,} pair presentations / {data['training']['optimizer_updates_completed']:,} updates"],
        ["Batch / accumulation", f"{cfg['batch_pairs']} pair (2 images) per microbatch / {cfg['accumulation']} microbatches"],
        ["Optimizer", f"AdamW; LR {cfg['learning_rate']:g} to {cfg['minimum_learning_rate']:g}; warmup {cfg['warmup_steps']}; matrix weight decay {cfg['weight_decay']:g}"],
        ["Loss / clipping / seed", f"Cosine contrastive margin {cfg['margin']}; norm clip 1; seed {cfg['seed']}"],
        ["Numerics", "CUDA/BF16 for training and inference; FP32 normalized CLS/cosines; forced Flash SDPA; deterministic algorithms"],
        ["Memory", "Nonreentrant gradient checkpointing; no reduced-resolution or kernel fallback"],
    ], [155, 373])
    heading("Frozen training-calibrated thresholds")
    table(["Native arm", "Strict score > cutoff", "Training false accepts", "Training impostors"], [
        [arm.split("_")[0].title(), f"{data['dino'][arm]['threshold']['cutoff']:.9f}",
         data["dino"][arm]["threshold"]["training_false_accepts"], data["dino"][arm]["threshold"]["training_valid_impostors"]]
        for arm in ARMS], [108, 148, 136, 136])
    para("The integer false-accept budget is 185 of 18,597 training impostors; ties at the boundary reject. "
         "Both selections were sealed before final testing. This is resubstitution calibration, not independent "
         "development calibration or a guarantee of 1% test FAR.", small=True)

    story.append(PageBreak())
    heading("Left hands, right hands and two thumbs")
    para("Each row contains 700 genuine and 700 impostor attempts. Every DINO slap type uses its arm's pooled "
         "training cutoff. Qwen per-type rates are reproduced from the original parsed decisions on those same pairs.")
    table(["Method", "Type", "Accuracy", "TAR", "FAR", "Failures"], [
        [r["method"], r["population"], percentage(r["accuracy"]), percentage(r["tar"]), percentage(r["far"]), r["failures"]]
        for r in rows if r["population"] != "pooled" and not r["method"].startswith("DINOv2-small")],
        [190, 48, 80, 72, 72, 66])
    para("The companion CSV also includes all small/448 per-type results and every confusion-matrix count. "
         "No type-specific tuning, exclusion of thumbs, segmentation, augmentation or pair regeneration was performed.", small=True)
    heading("Label and selection treatment")
    para("All supplied labels and pairs remain intact. The new cohort has zero disagreements between supplied labels "
         "and numeric-prefix equality, so the label-exclusion sensitivity equals the full population. This does not "
         "confirm biological identity semantics. SDK zeros remain zero; positive-component-sum selection and unknown "
         "template-extraction status limit any claim about performance on the full upstream pool.")

    story.append(PageBreak())
    heading("Continuous-score discrimination: descriptive only")
    roc = make_roc(data, font)
    story.append(roc);renderSVG.drawToFile(roc, str(paths["roc"]))
    markdown.append("![Descriptive pooled score ROC](pooled_roc.svg)")
    table(["Continuous-score method", "Valid scores", "AUC", "Interpolated EER"], [
        [name, valid, f"{curve['auc']:.4f}" if curve else "unavailable",
         percentage(curve["eer_linear_interpolation"]) if curve else "unavailable"]
        for name, curve, valid, _ in continuous_curves(data)], [234, 90, 90, 114])
    para("These test-score curves describe ordering; they select no operating cutoff or checkpoint. The SDK "
         "arms have no verified non-test calibration here, so only descriptive curves are reported. Qwen supplies "
         "binary answers and <b>has no continuous-score ROC, AUC or EER</b> in this report.")
    para("A lower descriptive EER need not improve TAR at the training-selected cutoff. Larger architecture, native "
         "resolution and BF16/Flash execution change together; any observed difference cannot be attributed solely "
         "to model size or resizing. Extra ablations were outside the approved proposal.", small=True)

    story.append(PageBreak())
    heading("What this cohort supports")
    dep = data["dependence"]
    table(["Population property", "Training", "Testing"], [
        ["Supplied pairs", dep["training"]["pairs"], dep["testing"]["pairs"]],
        ["Unique images", dep["training"]["images"], dep["testing"]["images"]],
        ["Numeric-prefix groups", dep["training"]["numeric_groups_unconfirmed_identity_semantics"], dep["testing"]["numeric_groups_unconfirmed_identity_semantics"]],
        ["Images reused in multiple pairs", dep["training"]["images_in_multiple_pairs"], dep["testing"]["images_in_multiple_pairs"]],
        ["Maximum pairs using one image", dep["training"]["maximum_pairs_using_one_image"], dep["testing"]["maximum_pairs_using_one_image"]],
    ], [310, 109, 109])
    para("Training/testing share zero exact pairs, image paths, numeric-prefix groups, raw-file hashes and decoded "
         "image hashes. All 61 leading-zero alias groups are confined to training. <b>This is numeric-group holdout; "
         "biological subject identity and independent captures remain unconfirmed.</b>")
    para("Within-phase pairs reuse images and groups. All testing images recur in multiple pairs. Rates describe "
         "this finite cohort; no independent-pair confidence intervals, population low-FAR claims or significance "
         "tests are made. One false accept changes pooled FAR by 1/2,100 = 0.0476 percentage points.")
    heading("Paired Large/native frozen-to-adapted decisions")
    paired = data["paired_native_dino"]["pooled"]
    table(["Supplied class", "Frozen only accepts", "Adapted only accepts", "Change in acceptance"], [
        [label.title(), paired[label]["frozen_only_accept"], paired[label]["adapted_only_accept"],
         f"{100 * paired[label]['adapted_minus_frozen_acceptance_rate']:+.2f} percentage points"]
        for label in ("genuine", "impostor")], [108, 134, 134, 152])
    para("A positive genuine change increases TAR. A positive impostor change increases FAR. These are paired "
         "descriptive counts. The aggregate retains additional paired DINO-to-Qwen counts, without turning binary "
         "answers into confidence scores.", small=True)
    para("Prior results for this dataset were known before this follow-up. This is not an untouched-test or "
         "independent-replication claim. No checkpoint, cutoff or hyperparameter was chosen using these final-test summaries.")

    story.append(PageBreak())
    heading("Reproducibility and collaborator evidence")
    counts = data["training"]["counts"]
    para(f"Offline checkpoint verification found <b>{counts['changed_frozen_tensors']} changed frozen tensors</b> "
         f"out of {counts['frozen_tensors']}; {counts['changed_trainable_tensors']} of {counts['trainable_tensors']} "
         "selected trainable tensors changed. Full epoch membership counts, optimizer update count, final selection, "
         "base/checkpoint digests, source commits and saved test completion hashes were verified. This is one fixed-seed run; "
         "training-seed variability is not estimated.")
    para("Pinned Large revision: 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c.<br/>"
         "Base weights SHA-256: 399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23.", small=True)
    first = data["dino"]["frozen_dinov2"]
    para("Execution source commit: " + escape(first["code_commit"]) + ". Reporting-only commits may be newer; "
         "every recorded protected runtime file is checked against its execution commit.", small=True)
    para("Runtime versions: " + escape(", ".join(f"{k} {v}" for k, v in sorted(first["packages"].items()))) + ".", small=True)
    heading("What the supplied Qwen results establish")
    para("All four preserved Qwen prediction files cover the exact new test membership and supplied labels once; "
         "their source summaries, prompts and confusion matrices reproduce. The supplied training JSON is byte-identical "
         "to the DINO cohort. Adapter bytes and trainer state were inspected without executing collaborator code or "
         "unpickling training arguments. Konain owns Qwen fine-tuning; it was not duplicated here.")
    para("The supplied Qwen evaluation code starts with original RGB images but invokes its processor, whose saved "
         "configuration has resizing enabled. It therefore does not establish unresampled network input. Qwen and DINO "
         "preprocessing and operating policies must be described separately.")
    heading("Remaining actual-run provenance limits")
    para("Konain clarified that a prompt JSON is optional because the prompts are built in. The source confirms "
         "this for --prompt_json. Separately, the training script writes binary_training_config.json with selected "
         "counts and actual argument values. The fine-tuned summary references that saved file; its contents "
         "have not been supplied. No additional prompt JSON is required to reproduce the verified prompt strings.", small=True)
    for text in data["qwen_provenance_limits"]:
        if not text.startswith("Exact pairs") and not text.startswith("Qwen native") and not text.startswith("No population"):
            para(escape(text), small=True)
    para("The public report contains aggregate counts, rates and reproducibility digests only. Original images, "
         "filenames, private pair rows, model weights and raw responses remain local. All earlier reports and "
         "approved decisions are preserved.", small=True)
    para("Aggregate SHA-256: " + aggregate_digest, small=True)

    def footer(canvas, document):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#d3dee3"));canvas.line(42, 39, 570, 39)
        canvas.setFont(font, 8);canvas.setFillColor(colors.HexColor("#667581"))
        canvas.drawString(42, 26, "SYNTHETIC LAYOUT TEST | No Large/native results" if synthetic
                          else "SLAPBench | Large/native | Numeric-group holdout")
        canvas.drawRightString(570, 26, str(document.page));canvas.restoreState()

    document = SimpleDocTemplate(str(paths["pdf"]), pagesize=(612, 792), leftMargin=42, rightMargin=42,
        topMargin=40, bottomMargin=52, title=title, author="Bibesh Pyakurel",
        subject="Precise original-resolution DINOv2-Large and source-verified Qwen comparison")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    paths["markdown"].write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    with paths["csv"].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.aggregate.read_bytes()
    render_report(json.loads(raw), args.output_dir, hashlib.sha256(raw).hexdigest())
    print("Native report rendered offline; inspect every final page before publication.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
