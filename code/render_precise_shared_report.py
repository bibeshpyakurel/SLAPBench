"""Render reviewed aggregate JSON as a professor report, CSV and ROC figure.

Uses ReportLab's standard chart/document tools. No private inputs or model calls.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape


def percentage(value, places=2):
    return "unavailable" if value is None else f"{100 * value:.{places}f}%"


def summary_rows(data):
    rows = []
    for arm in ("frozen_dinov2", "adapted_dinov2"):
        result = data["dino"][arm]
        for group, summary in [("pooled", result["results"]["pooled"]),
                               *result["results"]["by_slap_type"].items(),
                               ("excluding_two_flagged_test_pairs" if result["results"]["label_ambiguity_sensitivity"]["excluded_pairs"] == 2 else "label_ambiguity_sensitivity", result["results"]["label_ambiguity_sensitivity"]["remaining"])]:
            rates = summary["all_attempts_failure_as_reject"]
            curve = summary["descriptive_valid_score_discrimination"] or {}
            rows.append({"method": arm, "population": group, "threshold": result["threshold"]["cutoff"],
                **{key: rates[key] for key in ("genuine_comparisons", "impostor_comparisons", "genuine_accepts",
                                              "false_accepts", "tar", "far", "balanced_accuracy")},
                "auc": curve.get("auc"), "eer": curve.get("eer_linear_interpolation"),
                "failed_genuine": summary["failures"]["genuine"], "failed_impostor": summary["failures"]["impostor"]})
    return rows


def make_roc(data, font="Helvetica"):
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, Line, String
    from reportlab.lib import colors

    methods = [
        ("Frozen DINOv2", data["dino"]["frozen_dinov2"]["results"]["pooled"]["descriptive_valid_score_discrimination"], "#16637d"),
        ("Adapted DINOv2", data["dino"]["adapted_dinov2"]["results"]["pooled"]["descriptive_valid_score_discrimination"], "#c55b27"),
        ("VeriFinger fused", data["sdk_descriptive"]["verifinger_fusion"]["pooled"]["discrimination"], "#4f7a43"),
        ("VeriFinger sum", data["sdk_descriptive"]["verifinger_sum"]["pooled"]["discrimination"], "#7a5e9a"),
    ]
    if "metrics" in data.get("historical_qwen_scores", {}):
        methods.append(("Historical Qwen score export", data["historical_qwen_scores"]["metrics"]["pooled"]["discrimination"], "#727983"))
    drawing = Drawing(520, 310)
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 54, 84, 440, 202
    active = [(name, curve, color) for name, curve, color in methods if curve]
    chart.data = [[(p["far"], p["tar"]) for p in curve["roc_det_rates"]] for _, curve, _ in active]
    chart.joinedLines = 1
    for axis in (chart.xValueAxis, chart.yValueAxis):
        axis.valueMin, axis.valueMax, axis.valueStep = 0, 1, .2
        axis.labels.fontName, axis.labels.fontSize = font, 8
        axis.visibleGrid = True
        axis.gridStrokeColor = colors.HexColor("#e5e7eb")
        axis.gridStrokeWidth = .4
    for index, (_, _, color) in enumerate(active):
        chart.lines[index].strokeColor = colors.HexColor(color)
        chart.lines[index].strokeWidth = 1.5
        if "Historical" in active[index][0]:
            chart.lines[index].strokeDashArray = [3, 2]
    drawing.add(Line(54, 84, 494, 286, strokeColor=colors.HexColor("#d1d5db"), strokeDashArray=[2, 3]))
    drawing.add(chart)
    drawing.add(String(274, 59, "False accept rate (valid scores)", textAnchor="middle", fontName=font, fontSize=9))
    drawing.add(String(54, 300, "True accept rate", fontName=font, fontSize=9))
    for index, (name, _, color) in enumerate(active):
        x, y = (54 if index % 2 == 0 else 288), 38 - 14 * (index // 2)
        drawing.add(Line(x, y + 2, x + 17, y + 2, strokeColor=colors.HexColor(color), strokeWidth=2))
        drawing.add(String(x + 23, y, name, fontName=font, fontSize=8, fillColor=colors.HexColor("#243243")))
    return drawing


def render_report(data, output, aggregate_digest):
    from reportlab.graphics import renderSVG
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = Path(output)
    font, bold = "Helvetica", "Helvetica-Bold"
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    if (fonts / "DejaVuSans.ttf").exists() and (fonts / "DejaVuSans-Bold.ttf").exists():
        pdfmetrics.registerFont(TTFont("SLAPSans", str(fonts / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("SLAPSans-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("SLAPSans", normal="SLAPSans", bold="SLAPSans-Bold")
        font, bold = "SLAPSans", "SLAPSans-Bold"
    paths = {key: output / name for key, name in {
        "pdf": "professor-report.pdf", "markdown": "professor-report.md", "csv": "dino_metrics.csv", "roc": "pooled_roc.svg"}.items()}
    if any(path.exists() for path in paths.values()):
        raise ValueError("Report rendering never overwrites existing artifacts.")
    rows = summary_rows(data)
    output.mkdir(parents=True, exist_ok=True)
    with paths["csv"].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName=bold, fontSize=25, leading=29,
                              textColor=colors.HexColor("#173b4b"), spaceAfter=10))
    styles.add(ParagraphStyle("ReportSubtitle", fontName=font, fontSize=11, leading=16, textColor=colors.HexColor("#5a6874"), spaceAfter=15))
    styles.add(ParagraphStyle("ReportBody", fontName=font, fontSize=10, leading=14, spaceAfter=9, alignment=TA_LEFT))
    styles.add(ParagraphStyle("ReportSmall", fontName=font, fontSize=8, leading=11, spaceAfter=7, textColor=colors.HexColor("#54616d")))
    styles.add(ParagraphStyle("ReportH2", fontName=bold, fontSize=13, leading=17,
                              spaceBefore=10, spaceAfter=7, textColor=colors.HexColor("#173b4b")))
    styles.add(ParagraphStyle("Cell", fontName=font, fontSize=8.5, leading=11))
    story = []
    markdown = []

    def para(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))
        markdown.append(text.replace("<b>", "**").replace("</b>", "**").replace("<br/>", "\n"))

    def heading(text):
        story.append(Paragraph(escape(text), styles["ReportH2"]))
        markdown.append("\n## " + text + "\n")

    def table(headers, values, widths=None):
        cells = [[Paragraph(escape(str(value)), styles["Cell"]) for value in row] for row in [headers, *values]]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#9ab0ba")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(t); story.append(Spacer(1, 9))
        md_table = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        md_table.extend("| " + " | ".join(map(str, row)) + " |" for row in values)
        markdown.append("\n".join(md_table))

    group_holdout = data.get("benchmark_scope") == "numeric_group_disjoint_fixed_pairs"
    date = data.get("study_date", "2026-11-01")
    synthetic = data.get("_synthetic", False)
    title = "SYNTHETIC LAYOUT TEST" if synthetic else "Precise DINOv2 contribution"
    story.append(Paragraph(title, styles["ReportTitle"]))
    markdown.extend(["# " + title, ""])
    para(f"Bibesh Pyakurel | {'Numeric-group holdout' if group_holdout else 'Shared-pair benchmark'} | {date}", "ReportSubtitle")
    if synthetic:
        para("<b>Synthetic fixtures only. This document contains no research results.</b>")
    if group_holdout:
        para(f"Frozen and freshly adapted DINOv2 use the same {data['testing_pairs']:,} supplied testing attempts: "
             "2,100 genuine and 2,100 impostor pairs, including right hands, left hands and two thumbs. "
             f"Adaptation uses all {data['training_pairs']:,} supplied training pairs. "
             "<b>Numeric filename-prefix groups, images and exact pairs are held out; biological subject "
             "identity and capture independence remain unconfirmed.</b>")
    else:
        para("Frozen and adapted DINOv2 are evaluated on the same 4,200 supplied attempts, including right hands, "
         "left hands and two thumbs. The approved adaptation uses all 39,290 supplied training pairs. "
         "<b>Testing holds out pairs, while reusing training images and identity groups.</b>")
    heading("Primary operating-point results")
    primary = [r for r in rows if r["population"] == "pooled"]
    table(["DINOv2 arm", "TAR", "FAR", "Balanced accuracy", "Valid / attempted"], [
        [r["method"].split("_")[0].title(), percentage(r["tar"]), percentage(r["far"]), percentage(r["balanced_accuracy"]),
         f"{r['genuine_comparisons'] + r['impostor_comparisons'] - r['failed_genuine'] - r['failed_impostor']} / "
         f"{r['genuine_comparisons'] + r['impostor_comparisons']}"] for r in primary], [118, 76, 76, 112, 146])
    para("Each arm uses its own pooled cutoff selected from training impostors at nominal 1% training FAR. "
         "This is training calibration with resubstitution optimism, not independent development calibration "
         "or a guarantee of 1% test FAR. Failures remain attempted comparisons and are rejected in this primary view.")
    table(["Arm", "Training-selected cutoff", "Genuine accepts / attempts", "False accepts / attempts"], [
        [r["method"].split("_")[0].title(), f"{r['threshold']:.8f}",
         f"{r['genuine_accepts']} / {r['genuine_comparisons']}", f"{r['false_accepts']} / {r['impostor_comparisons']}"]
        for r in primary], [85, 145, 149, 149])
    frozen, adapted = primary
    para(f"Observed adapted-minus-frozen changes: TAR {(adapted['tar'] - frozen['tar']) * 100:+.2f} percentage points; "
         f"FAR {(adapted['far'] - frozen['far']) * 100:+.2f} percentage points. At this fixed cutoff the adapted arm "
         f"still rejects {adapted['genuine_comparisons'] - adapted['genuine_accepts']} of "
         f"{adapted['genuine_comparisons']} genuine attempts. Score ordering alone does not establish high acceptance "
         "at a strict operating point or statistical superiority.")
    heading("LLM comparison status")
    if group_holdout:
        para("<b>Matching fine-tuned Qwen3-VL A/B outputs and actual run provenance for these new pairs "
             "have not been supplied.</b> The older score exports cover different testing attempts and are "
             "not substituted. No superiority to Konain's model is claimed. Konain owns VLM fine-tuning.")
    else:
        para("<b>Konain's matching fine-tuned Qwen3-VL A/B results are not ready</b>, as confirmed by the user on "
         "2026-11-01. No claim of superiority to that model is made. The supplied older Qwen files are "
         "0-100 score exports using a different prompt; their descriptive results appear separately. "
         "Bibesh's work does not duplicate Konain's VLM fine-tuning.")
    heading("Claim supported by this milestone")
    if group_holdout:
        para("These are descriptive measurements of agreement with supplied labels on the declared numeric-group "
             "holdout. The model starts from the pinned pretrained base; the earlier adapted checkpoint is not "
             "reused. Numeric-group separation alone does not prove unseen biological subjects, independent "
             "acquisitions, population confidence bounds, or the cause of a VLM limitation.")
    else:
        para("These are descriptive measurements of agreement with supplied benchmark labels for held-out pairs "
         "of seen images and identities. They do not establish unseen-person generalization, independent "
         "captures, population confidence bounds, or the cause of any VLM limitation.")

    story.append(PageBreak())
    heading("Protocol and per-type evaluation")
    cfg = data["configuration"]
    para(f"DINOv2-small uses the pinned pretrained revision and whole-image grayscale-to-RGB letterboxing at 448 x 448. "
         f"Adaptation updates only the final two blocks and final layer norm. The fixed schedule is {cfg['epochs']} full epochs; "
         f"{cfg['batch_pairs']} explicit pairs per microbatch; accumulation {cfg['accumulation']}; AdamW LR {cfg['learning_rate']:g}, "
         f"minimum LR {cfg['minimum_learning_rate']:g}; weight decay {cfg['weight_decay']:g}; seed {cfg['seed']}; {cfg['precision'].upper()}. "
         "The final epoch checkpoint is selected in advance. No test-based tuning, extra cross-pairs, segmentation "
         "or SDK-zero filtering is used.")
    table(["Arm / slap type", "Genuine / impostor attempts", "TAR", "FAR", "Failures"], [
        [r["method"].split("_")[0].title() + " / " + r["population"],
         f"{r['genuine_comparisons']} / {r['impostor_comparisons']}", percentage(r["tar"]), percentage(r["far"]),
         r["failed_genuine"] + r["failed_impostor"]]
        for r in rows if r["population"] in {"right", "left", "thumbs"}], [145, 139, 80, 80, 84])
    para("All slap types use the same arm-specific pooled training cutoff. Type-specific cutoffs are not selected.", "ReportSmall")
    heading("Label ambiguity sensitivity")
    if group_holdout:
        para("There are zero supplied-label disagreements with numeric-prefix equality in training or testing. "
             "All supplied labels and pairs are retained. The ambiguity-exclusion view therefore equals the "
             "primary result. Leading-zero aliases remain unconfirmed identity semantics, not corrected ground truth.")
    else:
        para("Twenty-four training and two testing impostor labels disagree with numeric-identity equality for "
         "leading-zero aliases. All labels remain as supplied. The secondary result below excludes the two "
         "flagged testing rows only; it neither corrects identity truth nor changes training, checkpoints or cutoffs.")
    table(["Arm", "Remaining genuine / impostor", "TAR", "FAR"], [
        [r["method"].split("_")[0].title(), f"{r['genuine_comparisons']} / {r['impostor_comparisons']}",
         percentage(r["tar"]), percentage(r["far"])]
        for r in rows if r["population"] in {"excluding_two_flagged_test_pairs", "label_ambiguity_sensitivity"}], [118, 210, 100, 100])
    heading("Paired frozen-to-adapted changes")
    paired = data["paired_dino"]["pooled"]
    table(["Supplied class", "Frozen only accepts", "Adapted only accepts", "Adapted - frozen accept rate"], [
        [label.title(), paired[label]["frozen_only_accept"], paired[label]["adapted_only_accept"],
         percentage(paired[label]["adapted_minus_frozen_acceptance_rate"])] for label in ("genuine", "impostor")],
          [105, 130, 130, 163])
    para("A positive genuine difference increases TAR. A positive impostor difference increases FAR. "
         "These are paired descriptive counts, not significance tests.", "ReportSmall")

    story.append(PageBreak())
    heading("Continuous-score discrimination: descriptive only")
    markdown.append("![Pooled descriptive ROC curves](pooled_roc.svg)")
    roc = make_roc(data, font)
    story.append(roc)
    renderSVG.drawToFile(roc, str(paths["roc"]))
    curves = [("Frozen DINOv2", data["dino"]["frozen_dinov2"]["results"]["pooled"]["descriptive_valid_score_discrimination"]),
              ("Adapted DINOv2", data["dino"]["adapted_dinov2"]["results"]["pooled"]["descriptive_valid_score_discrimination"]),
              ("VeriFinger fused", data["sdk_descriptive"]["verifinger_fusion"]["pooled"]["discrimination"]),
              ("VeriFinger sum", data["sdk_descriptive"]["verifinger_sum"]["pooled"]["discrimination"])]
    if "metrics" in data.get("historical_qwen_scores", {}):
        curves.append(("Historical Qwen score export", data["historical_qwen_scores"]["metrics"]["pooled"]["discrimination"]))
    table(["Saved continuous-score arm", "Valid-score AUC", "Interpolated EER"], [
        [name, f"{curve['auc']:.4f}" if curve else "unavailable",
         percentage(curve["eer_linear_interpolation"]) if curve else "unavailable"] for name, curve in curves], [288, 120, 120])
    if group_holdout:
        para("Curves and interpolated EER describe test score ordering, with no cutoff or checkpoint selection. "
             "The SDK arms have no verified non-test calibration here; no operational threshold is chosen from "
             "their testing scores. Binary Qwen A/B decisions would supply one operating point, not a score ROC/EER.")
        para("SDK values are supplied fused scores or per-finger sums. Raw zeros remain zero. Selection uses "
             "positive component sums; template-extraction status is unresolved. Score presence is not proof "
             "of successful extraction, and this selected cohort is not a full upstream-pool evaluation.", "ReportSmall")
    else:
        para("Curves and interpolated EER summarize test score ordering. They do not choose an operational cutoff "
         "or checkpoint. SDK and historical Qwen arms have no verified non-test calibration in this report; "
         "no thresholded comparison is constructed from their test results.")
    if not group_holdout:
        para("SDK values are supplied fused scores or per-finger sums. All raw zeros remain zero; template-extraction "
         "status is unresolved, so score presence is not proof of successful extraction. Historical Qwen scores "
         "have a four-finger confidence prompt despite thumbs in the population, unpinned model/run provenance, "
         "and no supplied raw generated replies or adapter attestation.", "ReportSmall")

    story.append(PageBreak())
    heading("Dependence, uncertainty and reproducibility")
    dependence = data["dependence"]
    table(["Population property", "Training", "Testing"], [
        ["Pairs", dependence["training"]["pairs"], dependence["testing"]["pairs"]],
        ["Distinct images", dependence["training"]["images"], dependence["testing"]["images"]],
        ["Numeric groups (identity semantics unconfirmed)",
         dependence["training"]["numeric_groups_unconfirmed_identity_semantics"],
         dependence["testing"]["numeric_groups_unconfirmed_identity_semantics"]],
        ["Images appearing in multiple pairs", dependence["training"]["images_in_multiple_pairs"],
         dependence["testing"]["images_in_multiple_pairs"]],
        ["Maximum pairs using one image", dependence["training"]["maximum_pairs_using_one_image"],
         dependence["testing"]["maximum_pairs_using_one_image"]]], [304, 112, 112])
    overlap = dependence["train_test_overlap"]
    para(f"Training/testing overlap: {overlap['exact_pairs']} exact pairs; {overlap['images']} images; "
         f"{overlap['numeric_groups']} numeric groups. Reused images and groups invalidate an independent-pair "
         "binomial interpretation. This report supplies exact denominators, reuse counts and label sensitivity; "
         "it does not publish population intervals, p-values or a low-FAR guarantee. One pooled false accept is "
         "1/2,100 = approximately 0.048 percentage points; this resolution is not a population bound.")
    if group_holdout:
        para("Cross-phase raw-file and decoded-grayscale hash overlap is zero. All 61 inventoried padding-alias "
             "groups occur in training, limiting holdout representativeness. Earlier adaptation saw all new "
             "testing images/groups and 1,918 exact testing pairs; that checkpoint is excluded from this run. "
             "Prior dataset findings are known, so this is not an untouched-dataset or independent-replication claim.", "ReportSmall")
    heading("Execution and preservation")
    first = data["dino"]["adapted_dinov2"]
    para(f"The user reviewed the concrete protocol and authorized training, calibration and testing on {data.get('review_date', '2026-11-01')}. "
         "That is not a new professor attestation or a confirmation of canonical identities. Original data, "
         "labels, scores and collaborator scripts remain unchanged. Private hashes, approval record, epoch "
         "checkpoints, calibration scores, test outcomes and failure records remain in ignored local storage.")
    if "training" in data:
        counts = data["training"]["counts"]
        if group_holdout:
            para(f"Fresh pinned-base initialization; {data['training']['optimizer_updates_completed']:,} optimizer updates "
                 "completed across three full epochs; final checkpoint selected in advance.", "ReportSmall")
        para(f"Offline tensor verification found {counts['changed_frozen_tensors']} changed frozen tensors out of "
             f"{counts['frozen_tensors']}; {counts['changed_trainable_tensors']} of {counts['trainable_tensors']} selected "
             "trainable tensors changed. This is one fixed-seed adaptation run; training-seed variability is not estimated.", "ReportSmall")
    para("Execution source commit: " + escape(first["code_commit"]) + ". "
         "Public reproduction uses the shared-pair protocol note, run CLI, aggregate verifier and report renderer. "
         "The private input bundle and model/image assets are required separately.", "ReportSmall")
    packages = first.get("packages", {})
    para("Runtime versions: " + escape(", ".join(f"{key} {value}" for key, value in sorted(packages.items()))) + ".", "ReportSmall")
    heading("Remaining collaborator contribution")
    para("When Konain's results are available, verify exact testing attempts, actual training membership, "
         "prompt and processor, base revision, adapter/checkpoint digest, checkpoint-selection rule, raw replies, "
         "parsing and failures. Then add the fixed A/B operating point on these same attempts. A binary answer "
         "does not supply a continuous-score ROC/EER. No additional DINO tuning or local Qwen fine-tuning is "
         "implied by that comparison.")
    para("Aggregate SHA-256: " + aggregate_digest, "ReportSmall")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d3dee3")); canvas.line(42, 39, 570, 39)
        canvas.setFont(font, 8); canvas.setFillColor(colors.HexColor("#667581"))
        canvas.drawString(42, 26, "SYNTHETIC LAYOUT TEST | No research results" if synthetic
                          else "SLAPBench | Precise | " + ("Numeric-group holdout" if group_holdout else "Descriptive shared-pair evaluation"))
        canvas.drawRightString(570, 26, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(paths["pdf"]), pagesize=(8.5 * inch, 11 * inch),
        leftMargin=42, rightMargin=42, topMargin=40, bottomMargin=52,
        title=title, author="Bibesh Pyakurel", subject="Reviewed Precise shared-pair DINOv2 contribution")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    paths["markdown"].write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.aggregate.read_bytes()
    render_report(json.loads(raw), args.output_dir, hashlib.sha256(raw).hexdigest())
    print("Professor report, metric table and ROC figure rendered; visual review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
