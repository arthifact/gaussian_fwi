"""Rebuild the illustrated walkthrough document from this repository.

    python docs/build_walkthrough.py --output "Gaussian FWI Walkthrough.docx"

Every code excerpt is copied verbatim from the package; the registry below
records which file each came from, and tests/unit/test_walkthrough.py fails if
the source changes without the document following. Figures 1, 2, 4 and 8 are
constructed illustrations kept in docs/figures/. Figures 3, 5, 6 and 7 are
produced by the scripts in models/ and read from the results directory.

Needs the document extra: pip install -e ".[docs]"
"""

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

# Code shown in the document, with the file each block is copied from. Every
# entry must appear verbatim and contiguously in that file; tests/unit/
# test_walkthrough.py fails if the source moves on without the document.
EXCERPTS = {
    "forward calculation": ("gaussian_fwi/inversion.py", [
        "raw = model.raw()",
        "velocity = model.bound_velocity(raw).reshape(model.grid.shape)",
        "prediction = observations.acquisition.simulate(velocity)",
    ]),
    "training objective": ("gaussian_fwi/inversion.py", [
        'train_bands = objective.losses(prediction, active, "train")',
        "train = train_bands.mean()",
    ]),
    "block parameters": ("gaussian_fwi/field.py", [
        "self.centers = nn.Parameter(centers.clone())  # (n, d) metres: (x, z) or (x, y, z)",
        "self.log_scales = nn.Parameter(scales.log().clone())  # (n, d) log widths in metres",
        "self.shears = nn.Parameter(centers.new_zeros((n, d * (d - 1) // 2)))  # tilt, 1 in 2D",
        "self.amplitudes = nn.Parameter(centers.new_zeros(n))  # (n,) signed m/s contribution",
    ]),
    "depth background": ("gaussian_fwi/field.py", [
        "points = self._points if points is None else points",
        "depth = points[:, -1] / self.grid.extent[-1]",
        "value = self.background[0] + (self.background[1] - self.background[0]) * depth",
    ]),
    "active cutoffs": ("gaussian_fwi/inversion.py", [
        "active = config.cutoffs[:stage_index + 1]",
    ]),
    "clone or split": ("gaussian_fwi/refinement.py", [
        "radius = float(torch.linalg.eigvalsh(values[kernel_id][1].double())[-1].sqrt())",
        "if radius <= threshold:",
        "    edit = topology.clone(kernel_id)",
        "else:",
        "    # A separate deterministic stream per parent/event avoids global",
        "    # RNG mutation and makes rejected proposals replayable.",
        "    seed = (cfg.seed + 104729 * global_step + 1000003 * kernel_id) % (2**63)",
        "    edit = topology.split(kernel_id, seed=seed)",
    ]),
    "validation selection": ("gaussian_fwi/inversion.py", [
        "if eligible and best.consider(validation, step, model, optimizer):",
        "    best_topology = controller.topology.state_dict()",
    ]),
}

# Lines quoted individually rather than as a contiguous block.
SINGLE_LINES = {
    "backward": ("gaussian_fwi/inversion.py", "loss.backward()"),
    "observe": ("gaussian_fwi/inversion.py", "controller.observe(model)"),
    "step": ("gaussian_fwi/inversion.py", "optimizer.step()"),
    "project": ("gaussian_fwi/inversion.py", "model.project_()"),
    "after update": ("gaussian_fwi/inversion.py",
                     "event = controller.after_update(model, optimizer, step + 1)"),
}

# One source line is too wide for the page. The wrapped form shown in the
# document must parse to the same syntax tree as the line it replaces.
WRAPPED = {
    "residual = lowpass(prediction[:, ids], self.dt, cutoff)"
    " - self.targets[cutoff][shots][:, ids]": [
        "residual = (",
        "    lowpass(prediction[:, ids], self.dt, cutoff)",
        "    - self.targets[cutoff][shots][:, ids]",
        ")",
    ],
}
WRAPPED_SOURCE = "gaussian_fwi/core/physics.py"


def build(output, figures, results):
    """Write the document, reading figures and fit results from disk."""
    FIG, RES = Path(figures), Path(results)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(.72), Inches(.68)
    section.left_margin, section.right_margin = Inches(.8), Inches(.8)
    section.footer_distance = Inches(.30)

    styles = doc.styles
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption",
                 "List Number", "List Bullet"):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        for c in style.element.xpath(".//w:color"):
            for attr in ("themeColor", "themeTint", "themeShade"):
                c.attrib.pop(qn("w:" + attr), None)
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
        fonts = style.element.get_or_add_rPr().rFonts
        for attr in list(fonts.attrib):
            if "theme" in attr.lower():
                del fonts.attrib[attr]
        for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn("w:" + attr), "Times New Roman")

    normal = styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.space_after = Pt(7)
    styles["Title"].font.size = Pt(26)
    styles["Title"].font.bold = False
    styles["Title"].paragraph_format.space_after = Pt(6)
    styles["Subtitle"].font.size = Pt(13)
    styles["Subtitle"].font.italic = False
    styles["Subtitle"].paragraph_format.space_after = Pt(16)
    for name, size in (("Heading 1", 15), ("Heading 2", 12)):
        st = styles[name]
        st.font.size = Pt(size)
        st.font.bold = True
        st.paragraph_format.space_before = Pt(15)
        st.paragraph_format.space_after = Pt(7)
        st.paragraph_format.keep_with_next = True
    cap = styles["Caption"]
    cap.font.size = Pt(10)
    cap.font.italic = False
    cap.font.bold = False
    cap.paragraph_format.line_spacing = 1.05
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(11)
    code = styles.add_style("Code", 1)
    code.base_style = normal
    code.font.name = "Courier New"
    code.font.size = Pt(9)
    code.paragraph_format.line_spacing = 1.04
    code.paragraph_format.space_after = Pt(0)
    code.paragraph_format.left_indent = Inches(.08)
    code.paragraph_format.keep_together = True
    for name in ("List Bullet", "List Number"):
        styles[name].paragraph_format.space_after = Pt(5)
        styles[name].paragraph_format.line_spacing = 1.10

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    run = footer.add_run()
    run.font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    field.append(r)
    footer._p.append(field)

    doc.core_properties.title = "Gaussian full waveform inversion"
    doc.core_properties.subject = "An illustrated walkthrough of the research code"
    doc.core_properties.keywords = "Gaussian fields, full waveform inversion, research code"


    def inline(paragraph, text):
        import re
        for part in re.split(r"(\*\*.*?\*\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                paragraph.add_run(part[2:-2]).bold = True
            elif part.startswith("`") and part.endswith("`"):
                run = paragraph.add_run(part[1:-1])
                run.font.name = "Courier New"
                run.font.size = Pt(9.3)
            else:
                paragraph.add_run(part)


    def body(text, style=None):
        p = doc.add_paragraph(style=style)
        inline(p, text)
        return p


    def heading(text, page_break=False):
        p = doc.add_paragraph(text, style="Heading 1")
        if page_break:
            p.paragraph_format.page_break_before = True
        return p


    def block(lines):
        for i, line in enumerate(lines):
            p = doc.add_paragraph(line, style="Code")
            p.paragraph_format.keep_with_next = i < len(lines) - 1
            if i == 0:
                p.paragraph_format.space_before = Pt(4)
            if i == len(lines) - 1:
                p.paragraph_format.space_after = Pt(9)


    def picture(path, alt, width=6.9):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = True
        pic = p.add_run().add_picture(str(path), width=Inches(width))
        pic._inline.docPr.set("descr", alt)


    def caption(text):
        p = doc.add_paragraph(style="Caption")
        inline(p, text)
        p.paragraph_format.keep_together = True


    def table(rows, widths, numeric_from=None):
        tab = doc.add_table(rows=0, cols=len(rows[0]))
        tab.alignment = WD_TABLE_ALIGNMENT.CENTER
        tab.autofit = False
        for column, width in zip(tab.columns, widths):
            column.width = Inches(width)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement("w:" + edge)
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "4")
            b.set(qn("w:color"), "D9D9D9")
            borders.append(b)
        tab._tbl.tblPr.append(borders)
        for ri, row in enumerate(rows):
            cells = tab.add_row().cells
            tr = tab.rows[-1]._tr.get_or_add_trPr()
            tr.append(OxmlElement("w:cantSplit"))
            if ri == 0:
                tr.append(OxmlElement("w:tblHeader"))
            for ci, (cell, text, width) in enumerate(zip(cells, row, widths)):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cp = cell._tc.get_or_add_tcPr()
                margins = OxmlElement("w:tcMar")
                for edge, value in (("top", "85"), ("bottom", "85"),
                                    ("left", "105"), ("right", "105")):
                    item = OxmlElement("w:" + edge)
                    item.set(qn("w:w"), value)
                    item.set(qn("w:type"), "dxa")
                    margins.append(item)
                cp.append(margins)
                if ri == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "EAF0F3")
                    cp.append(shade)
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.06
                if numeric_from is not None and ci >= numeric_from and ri > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                inline(p, text)
                for r in p.runs:
                    r.font.size = Pt(10)
                    if ri == 0:
                        r.bold = True
                p.paragraph_format.keep_with_next = ri == 0
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)


    doc.add_paragraph("Gaussian full waveform inversion", style="Title")
    doc.add_paragraph("A walkthrough of the research code", style="Subtitle")

    body("The source and receiver positions are known and the waveforms are recorded. The problem "
         "is to find a velocity model that reproduces those recordings. The model here is a depth "
         "background plus adjustable Gaussian components: waves are simulated through it, the "
         "mismatch with the recordings is measured, and gradients of that mismatch change the "
         "components.")
    body("The whole method follows one chain:")
    p = body("Gaussian parameters → velocity field → simulated recordings → waveform loss")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(3)
    body("Gradients carry the loss sensitivity back to the parameters.")
    body("A separate editing step can clone, split, or remove components. The continuous parameters "
         "are differentiable; those discrete editing decisions are not.")
    body("The method lives in the `gaussian_fwi` package. `gaussian_fwi/field.py` holds the "
         "representation, `gaussian_fwi/inversion.py` the training loop, `gaussian_fwi/refinement.py` "
         "the population controller, and `gaussian_fwi/core/` the acoustic solver, objective and "
         "file formats. Every excerpt below is copied verbatim from that implementation; the file "
         "and purpose are named alongside each one.")

    heading("1 The inverse problem")
    body("A receiver records a signal over time. Changing underground velocity changes how waves "
         "travel, which changes the predicted signal. Full-waveform inversion adjusts the model by "
         "comparing simulated and recorded waveforms.")
    body("The essential forward calculation is in `invert()`:")
    block(EXCERPTS["forward calculation"][1])
    body("The first two lines build the velocity map; the third asks what recordings it would "
         "produce. "
         "`Acquisition.simulate()` in `gaussian_fwi/core/physics.py` calls `deepwave.scalar` to solve "
         "the acoustic wave equation on the grid.")
    body("The field can be evaluated at continuous physical coordinates. The wave solver still uses "
         "a numerical grid; a continuous representation does not eliminate discretization.")

    heading("2 The controls of one Gaussian")
    body("Think of a localized, smooth bump. It has a center, a size, an orientation, and a signed "
         "amplitude. A positive amplitude raises the raw field; a negative amplitude lowers it. "
         "Several overlapping components describe one model.")
    picture(FIG / "gaussian-geometry.png",
            "A Gaussian component's center, long and short widths, tilt, and signed amplitude.")
    caption("**Figure 1. The controls of one Gaussian.** The center sets where it acts; the widths "
            "and tilt set its shape; the signed amplitude sets how strongly it raises or lowers the "
            "raw field. The ellipses are one- and two-standard-deviation contours, not the edge of "
            "its support. Constructed illustration.")
    body("In `gaussian_fwi/field.py`, the learnable quantities live in `GaussianBlock`:")
    block(EXCERPTS["block parameters"][1])
    body("`nn.Parameter` tells PyTorch that the optimizer may change these tensors. In 2D, each "
         "component has six numbers: two center coordinates, two log scales, one shear, and one "
         "amplitude. The depth background has two more numbers: its top and bottom values.")
    body("Widths are exponentiated so they remain positive. Scales and shears form a lower "
         "triangular matrix L; the covariance is L @ L.T. This makes the Gaussian ellipse positive "
         "definite while allowing it to stretch and tilt. The stored diagonal scales are not "
         "necessarily the principal widths of the tilted ellipse.")
    body("The kernel is Gaussian through five normalized standard deviations, then smoothly "
         "vanishes by six (`kernel()` in `gaussian_fwi/raster.py`). Its finite support lets the "
         "decoder evaluate nearby component–point pairs in bounded batches.")
    body("The unknowns are therefore the shapes and strengths of these components. Their amplitudes "
         "are signed velocity contributions in m/s, not probabilities or occupancies.")

    heading("3 Building the velocity field")
    body("The field is a depth background plus the sum of signed Gaussian contributions, passed "
         "through a smooth bounding function that maps the raw sum to physical velocity. Written "
         "out, with two components:")
    p = body("u(x) = b + a\u2081\u03c6\u2081(x) + a\u2082\u03c6\u2082(x),      v(x) = B(u(x))")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table([["Symbol", "What it is"],
           ["x", "A position in the model, in metres: (x, z) in 2D, (x, y, z) in 3D"],
           ["b", "The background: a straight ramp from the top value to the bottom one, m/s"],
           ["\u03c6\u1d62(x)", "Component i's shape: 1 at its own center, falling smoothly to 0 "
            "outward. Dimensionless"],
           ["a\u1d62", "Component i's signed amplitude, m/s. Positive raises velocity, negative "
            "lowers it"],
           ["u(x)", "The raw field, m/s. A plain sum, which may fall outside physical limits"],
           ["B", "A smooth bound that maps the raw field into the allowed velocity range"],
           ["v(x)", "The physical velocity the wave solver uses, m/s"]],
          [1.0, 5.9])
    body("Two of those deserve a sentence each, because they are where the shapes come from.")
    body("**The component shape \u03c6.** For a component with center \u03bc and covariance \u03a3, let "
         "q(x) = (x \u2212 \u03bc)\u1d40 \u03a3\u207b\u00b9 (x \u2212 \u03bc). This is squared distance from the center, measured in "
         "units of the component's own widths, so q = 1 one standard deviation away and q = 4 two "
         "away, in whichever direction the ellipse points. Then \u03c6 = exp(\u2212q/2), multiplied by a "
         "taper that equals exactly 1 out to five standard deviations and exactly 0 beyond six. At "
         "the center q = 0 and \u03c6 = 1, so a component contributes exactly a\u1d62 m/s at its own center "
         "and less further out. Past six standard deviations it contributes nothing at all, which is "
         "what makes the sum cheap to evaluate: each query point only sees nearby components.")
    body("**The bound B.** Velocity has to stay inside the physical range the solver accepts, but a "
         "hard clamp would have zero gradient outside the range and stop the optimizer from "
         "recovering. B is a softplus pair: it leaves the field essentially untouched well inside "
         "the bounds, then bends it smoothly as it approaches them. The field can always be pushed "
         "back, and every step remains differentiable.")
    picture(FIG / "gaussian-field.png",
            "Two signed Gaussian contributions added to a constant background.")
    caption("**Figure 2. Add the contributions, then apply the bounds.** Read left to right: the "
            "positive component lifts the field above the 2,500 m/s background; the negative one "
            "lowers it. The black curve is the resulting physical velocity. This constructed 1D "
            "example makes the addition visible; the same operation is used in 2D and 3D.")
    body("The depth background in `GaussianField.raw()` is a straight interpolation between its "
         "top and bottom values:")
    block(EXCERPTS["depth background"][1])
    body("The component sum is added to that background. `bound_velocity()` then applies a smooth "
         "bounding function. Consequently, a coefficient of +100 m/s need not produce exactly "
         "+100 m/s in physical velocity near a bound.")
    body("The seeds start at zero amplitude, so the initial field is just the bounded background. "
         "Amplitude gradients can activate them. At exactly zero amplitude, a component's center and "
         "covariance have zero gradient through its field contribution; geometry learning becomes "
         "active after its amplitude changes.")

    heading("4 Comparing simulated and observed waveforms")
    body("This line in `invert()` computes the data-fitting part:")
    block(EXCERPTS["training objective"][1])
    body("Inside `WaveformObjective.losses()`, the code selects the relevant receivers, filters the "
         "predicted and observed traces, and measures their normalized squared difference:")
    block(next(iter(WRAPPED.values()))
          + ["weighted = residual * self.weights[cutoff][shots][:, ids]",
             "value = weighted.square().mean() / self.denominators[cutoff][split]"])
    body("`ids` selects the receivers of the requested partition; `shots` selects the shot range, "
         "which is every shot unless a caller requests a single batch. The first statement is wrapped "
         "here for width and is one line in the source.")
    body("The denominator uses observed signal energy. Optional time gain and trace balancing change "
         "how samples contribute; the trace-balancing reference comes from training data. The full "
         "training objective also includes the configured spatial regularization and a penalty on "
         "the raw field outside its bounds.")
    picture(RES / "marmousi/waveforms.png",
            "One recorded training trace against the initial and selected predictions, "
            "and the RMS residual over all training traces.")
    caption("**Figure 3. Look at the error, not just the overlapping traces.** Left: one recorded "
            "training trace against the predictions of the initial and selected models, low-passed "
            "at 20 Hz. Right: root mean square residual over all 40 training receivers and 8 shots "
            "at each time. Fitting removes most of the reflected energy between 0.15 and 0.6 s, a "
            "43% reduction in mean residual, but the curve does not approach zero: the recorded "
            "traces carry noise, and that noise sets a floor no velocity model can pass. Amplitudes "
            "are in arbitrary simulation units. The plotted RMS is a diagnostic; the training loss "
            "averages squared error and normalizes it by observed energy. From the fit in "
            "section 10.")
    body("The inversion receives recordings and acquisition geometry. It does not receive the "
         "correct velocity map. `load_observations()` rejects a bundle that carries one, and the "
         "runner records `reference_velocity_loaded: false` for every fit.")

    heading("5 Learning through gradients")
    body("These are the update lines from `invert()`; the finite-value checks are omitted here:")
    block([SINGLE_LINES[name][1] for name in
           ("backward", "observe", "step", "project", "after update")])
    body("In order:")
    body("`backward()` propagates loss derivatives through wave propagation and the field.",
         style="List Number")
    body("`observe()` records the center-gradient magnitudes for later refinement decisions.",
         style="List Number")
    body("Adam updates the background, amplitudes, centers, scales, and shears.", style="List Number")
    body("`project_()` enforces the center and principal-width limits.", style="List Number")
    body("`after_update()` checks whether a population-editing event is scheduled.",
         style="List Number")
    body("The waveform mismatch therefore determines how the loss changes when a component is "
         "moved, reshaped or strengthened, and Adam uses that information to update the model.")

    heading("6 Introducing waveform detail in stages")
    body("The accepted profile uses cumulative cutoffs of 4, 7, 12, and 20 Hz. At each stage, the "
         "objective averages the losses for all cutoffs introduced so far:")
    block(EXCERPTS["active cutoffs"][1])
    body("At the 12 Hz stage, for example, the active filtered losses are 4, 7, and 12 Hz. This "
         "gradually introduces more temporal waveform detail. These are overlapping low-pass views, "
         "not disjoint frequency bands or hard cutoffs.")
    body("Waveform frequency and Gaussian width are different quantities. A low-frequency stage does "
         "not create a separate “macro model,” and low-frequency continuation does not guarantee "
         "convergence to the correct velocity.")

    heading("7 Cloning, splitting and pruning components")
    body("`RefinementController._event()` in `gaussian_fwi/refinement.py` first proposes removing "
         "eligible weak components, then ranks the remaining candidates by their mean center-gradient "
         "norm over recent updates.")
    body("A large score means the training objective is locally sensitive to moving that component. "
         "It is a refinement heuristic, not a measured benefit from adding children.")
    body("The clone-versus-split decision is directly visible in the code:")
    block(EXCERPTS["clone or split"][1])
    table([["Operation", "Selection idea", "What actually happens"],
           ["Prune", "Small absolute amplitude, subject to age/count limits",
            "Remove the component"],
           ["Clone", "High-ranked candidate with small principal width",
            "Keep the parent and append an identical child"],
           ["Split", "High-ranked candidate with large principal width",
            "Replace the parent with two sampled, narrower children"]],
          [.95, 2.55, 3.4])
    picture(FIG / "population-edits.png",
            "Before-and-after contribution curves for clone, split and prune.")
    caption("**Figure 4. Editing the population changes the field immediately.** A clone adds "
            "another full contribution at the same location. A split replaces the parent with two "
            "narrower contributions; dotted curves show the individual children. Pruning removes the "
            "contribution. These are constructed examples of the operations. The code also supports "
            "negative amplitudes.")
    body("A clone initially overlaps its parent and copies the full signed amplitude. A split "
         "samples two centers from the parent Gaussian, divides covariance by 1.6**2, and gives each "
         "child the parent's signed amplitude. Neither operation preserves the field. Neither "
         "guarantees a lower waveform loss.")
    body("That is why the code measures the whole event's velocity change against its pre-event "
         "field. The accepted profile allows at most 25 m/s at the propagation-grid samples. If the "
         "batch exceeds that limit, the code undoes it and retries with fewer proposals. This is a "
         "sampled bound, not a guarantee between grid points or on waveform error.")
    body("Surviving components keep their Adam history. New children start with fresh history. The "
         "algorithm differentiates through the field, not through the discrete choice to clone, "
         "split, or prune.")

    heading("8 Settling and selecting a result")
    body("Each frequency stage has an initial refinement window followed by settling: the number of "
         "components stays fixed, while their continuous parameters keep learning.")
    body("The accepted profile offers edits at updates 100, 150, …, 450. At update 500, population "
         "editing has stopped. Optimization continues through update 1,000.")
    body("Validation may be measured earlier, but it can select a checkpoint only during settling:")
    block(EXCERPTS["validation selection"][1])
    body("`eligible` is true only once the update index has passed the refinement stop step, so a "
         "checkpoint from the editing window can never be selected.")
    table([["Data", "Role"],
           ["Training receivers", "Drive gradients and refinement rankings"],
           ["Validation receivers", "Select a checkpoint from the settling phase"],
           ["Test receivers", "Evaluate the selected result at the end"]],
          [1.60, 5.3])
    body("These receiver sets are disjoint, and the loader rejects overlapping partitions before a "
         "fit starts.")

    heading("9 Running it: one recorded, verified call", page_break=True)
    body("The package exposes the whole workflow as a single call:")
    block(["import gaussian_fwi as gfwi",
           "",
           "fit = gfwi.run(\"data/marmousi_real.pt\", output=\"results/marmousi\")",
           "print(fit.summary())",
           "velocity = fit.velocity          # (z, x) in m/s, on the acquisition grid"])
    body("`run()` does more than optimize. Before fitting it records the complete profile, the "
         "SHA-256 of every executing source file, the dependency versions, and the content identity "
         "of the observation bundle. After fitting it reloads the saved field from disk, propagates "
         "it again with a fresh solve, and compares that prediction with the one it saved. "
         "`fit.verified` is true only when that independent replay reproduced the saved prediction "
         "exactly. The runner also re-checks the source hashes afterwards, so a fit whose code "
         "changed underneath it fails rather than reporting a result.")
    body("Any departure from the accepted profile is declared, validated immediately, and stored "
         "with the run:")
    block(["profile = gfwi.baseline(steps_per_stage=400, cutoffs=[4.0, 7.0])",
           "fit = gfwi.run(bundle, output=\"results/fit02\", profile=profile)"])
    body("`baseline()` rejects an unknown setting, a decreasing cutoff sequence, or a horizon too "
         "short to contain warm-up, a refinement opportunity and settling — at the call, rather than "
         "part way through a long fit. A finished directory reads back with `gfwi.Run.open(path)`, "
         "exposing the report, per-update history, refinement events and provenance as ordinary "
         "Python objects.")

    heading("10 A demonstration on Marmousi")
    body("`models/marmousi.npy` is the 70 × 70 development model, resampled and normalized to "
         "1500–4500 m/s at 10 m spacing. `models/make_observations.py` turns it into an "
         "observation-only bundle; `models/run_marmousi.py` fits that bundle and scores the result.")
    body("Fitting data you generated with the same solver, on the same grid, without noise, from the "
         "exact source wavelet is an inverse crime: the forward operator is invertible almost by "
         "construction, and any method looks strong. The generator removes those advantages by "
         "default.")
    table([["Condition", "Default", "Why"],
           ["Modelling grid", "2× finer than the inversion's, resampled onto it",
            "The inversion's operator is not the one that made the data"],
           ["Noise", "Band-limited, survey signal-to-noise 10", "Real traces are not clean"],
           ["Source wavelet", "8% peak-frequency error", "The source is estimated, not known"],
           ["Velocity bounds", "1400–5000 m/s, not the reference's 1500–4500",
            "The true range is not known in advance"]],
          [1.35, 2.35, 3.2])
    body("The survey itself is fixed for every fit reported here:")
    table([["Acquisition", "Value"],
           ["Model and grid", "70 \u00d7 70 samples at 10 m, 690 \u00d7 690 m"],
           ["Sources", "8, at 20 m depth, x = 40\u2013650 m"],
           ["Receivers", "68, at 20 m depth, x = 10\u2013680 m, every 10 m"],
           ["Maximum offset", "640 m (0.91 \u00d7 model depth)"],
           ["Record", "800 samples at 1 ms = 0.8 s"],
           ["Source wavelet", "Ricker, 12 Hz peak; the inversion is given 12.5 Hz"],
           ["Inversion cutoffs", "4 / 7 / 12 / 20 Hz, cumulative"],
           ["Noise", "band-limited Gaussian, survey signal-to-noise 10"],
           ["Per-band signal-to-noise", "4.4 at 4 Hz, 8.5 at 7 Hz, 12.1 at 12 Hz, 12.0 at 20 Hz"],
           ["Modelling grid", "2\u00d7 finer: 139 \u00d7 139 at 5 m, 0.5 ms, then resampled"],
           ["Receiver partitions", "40 training, 14 validation, 14 test, disjoint"],
           ["Solver", "4th-order acoustic, 20-cell PML, maximum velocity 5,000 m/s"]],
          [2.25, 4.65])
    body("The per-band figures matter more than the survey average. Low frequencies carry the "
         "large-scale velocity update, and a 12 Hz Ricker puts little energy at 4 Hz while the noise "
         "is spread across the band, so the first stage works at a signal-to-noise of about four.")
    body("Together these leave a 16% relative waveform difference from the self-consistent case. The "
         "cost of the crime is measurable: at a matched 240-update horizon, velocity RMSE falls from "
         "512.8 to 395.8 m/s under the crime and from 512.8 to 512.2 m/s under realistic conditions, "
         "and the final training waveform loss reaches about 1e-4 per band in the first case against "
         "0.16–0.42 in the second. Under the crime the data can be fit to near machine precision; "
         "under realistic conditions the misfit floors at a genuine noise and physics limit.")
    body("**A limitation of this survey, which the result cannot be read without.** Every source "
         "and receiver sits at 20 m depth, and the maximum source–receiver offset is 640 m against "
         "700 m of model depth. Diving waves therefore turn at roughly 210–320 m.")
    body("Below that depth the survey is not blind: reflections from deeper interfaces still travel "
         "down, return, and are recorded. But no ray turns there, so nothing samples that depth in "
         "transmission. Reflections constrain the position and strength of impedance contrasts, and "
         "constrain the smooth background velocity only weakly, because a slower model with shallower "
         "interfaces predicts nearly the same reflection traveltimes at these offsets. “Above 300 m” "
         "and “below 300 m” in what follows mean *with* and *without* diving-wave coverage — not "
         "with and without recorded data.")
    body("Errors are therefore reported separately for the two depth ranges. A single whole-model "
         "number is dominated by the part the survey constrains weakly, and reads as total failure "
         "even where recovery is real.")
    picture(RES / "marmousi/comparison.png",
            "Reference, initial and recovered velocity with the signed difference, realistic conditions.")
    caption("**Figure 5. A fit under realistic conditions.** Reference (evaluation only), initial "
            "model, and the field recovered from waveforms alone, on one velocity scale, with the "
            "signed difference. Stars mark the eight sources and dots the sixty-eight receivers, all "
            "at 20 m depth. The recovered field follows the dipping structure in the upper section; "
            "the largest errors sit below the depth diving waves reach. 240 updates, 6% of the "
            "accepted budget.")

    heading("11 What extra capacity does")
    body("A grid spends one parameter per cell, everywhere, whether or not the data constrain that "
         "cell. This representation decouples capacity from the grid, so the same survey can be "
         "fitted with a few hundred parameters or a few thousand. `models/capacity_study.py` does "
         "exactly that and reports error inside and outside the illuminated depth.")
    table([["Capacity", "Params", "Per cell", "RMSE all", "RMSE <300 m", "RMSE >300 m", "Roughness"],
           ["initial", "—", "—", "512.8", "231.4", "648.0", "0.0"],
           ["8 × 8", "380", "0.08", "500.2", "158.8", "647.2", "37.8"],
           ["16 × 16", "1532", "0.31", "522.3", "167.0", "675.6", "53.7"],
           ["32 × 32", "6068", "1.24", "512.2", "184.8", "658.4", "92.5"]],
          [.95, .75, .8, .95, 1.15, 1.15, .95], numeric_from=1)
    body("Roughness in the last column is the mean absolute second difference of the velocity "
         "field, averaged over both axes, in m/s per sample. It measures how much the field "
         "oscillates between neighbouring samples: a smooth field scores near zero, and speckle "
         "raises it. It says nothing about whether the field is correct, only how rough it is.")
    body("All three capacities reach the same waveform fit: final training bands within a few "
         "percent of 0.43 / 0.24 / 0.16 / 0.18. The extra 5,688 parameters therefore buy no data "
         "fit. They are spent on noise, and on structure in the unilluminated zone — where the "
         "380-parameter field instead stays at its prior, 647.2 m/s against the starting model's "
         "648.0. The larger fields actively corrupt that region.")
    body("Note the third column. A 32 × 32 seed lattice gives 1.24 parameters per grid cell on a "
         "70 × 70 model — more freedom than a grid of the same model. At that setting the "
         "representation supplies no dimensionality reduction at all, which is why it speckles.")
    picture(RES / "capacity/capacity.png",
            "Reference, initial and recovered fields at three capacities on one velocity scale.")
    caption("**Figure 6. Equal waveform fit, decreasing capacity.** Stars and dots mark the surface "
            "sources and receivers; the dashed line at 300 m marks the depth below which no diving "
            "wave turns. At 63 Gaussians the field is smooth and follows the dipping structure; at "
            "1,011 it is visibly speckled. Same data, same waveform fit, sixteen times the "
            "parameters.")
    body("A conventional smoothness prior does not substitute for this. Raising the TV weight on the "
         "32 × 32 field, at the same horizon:")
    table([["Setup", "Params", "RMSE all", "RMSE <300 m", "RMSE >300 m", "Roughness"],
           ["8 × 8, tv = 1e-4", "380", "500.2", "158.8", "647.2", "37.8"],
           ["32 × 32, tv = 1e-4", "6068", "512.2", "184.8", "658.4", "92.5"],
           ["32 × 32, tv = 1e-3", "6044", "511.8", "182.5", "658.3", "83.2"],
           ["32 × 32, tv = 1e-2", "6038", "538.2", "168.6", "696.9", "48.2"]],
          [1.5, .8, 1.05, 1.2, 1.2, .95], numeric_from=1)
    body("A hundredfold TV weight smooths the field and improves the illuminated zone, but degrades "
         "the unilluminated zone and the model as a whole, and still does not reach the low-capacity "
         "result. TV penalizes gradients uniformly: it blurs real structure where the data constrain "
         "it, while still permitting noise-driven structure where they do not. Lowering capacity "
         "removes the freedom instead of penalizing its use.")
    picture(RES / "capacity_vs_tv.png",
            "Recovered fields and error metrics for a small population, a large one, "
            "and the large one under a hundredfold TV weight.")
    caption("**Figure 7. A smoothness prior is not a substitute for capacity.** Top: the recovered "
            "fields, all reaching the same waveform fit; the dashed line marks the limit of "
            "diving-wave illumination. Bottom: velocity error above and below that depth, and field "
            "roughness, with the surface geometry overlaid. Strong TV smooths the large population "
            "and improves the zone diving waves reach, "
            "but pushes the unilluminated zone furthest from the truth \u2014 697 m/s against the "
            "starting model's 648 \u2014 while the 380-parameter field sits on the starting model "
            "there. TV penalizes gradients everywhere; reducing capacity removes the freedom "
            "instead.")
    body("**What this does and does not establish.** It is regularization by parameterization, "
         "measured on one model, one noise realization and one horizon. It is not a grid-FWI "
         "baseline: the control is a 32 × 32 Gaussian field with a stronger penalty, not 4,900 free "
         "pixels. And every run here learns geometry, so the study varies component count rather "
         "than whether geometry is movable; the fixed-versus-learned comparison belongs to the "
         "supervised study in the manuscript.")

    heading("12 Understanding macro and detail")
    body("The code learns one field. It does not contain separate macro/detail optimizers or a "
         "separate broad/fine gradient-weighting branch. For analysis a macro view can be defined by "
         "spatially smoothing the physical velocity, with detail as the residual. That decomposition "
         "is a diagnostic, distinct from the temporal waveform filtering described above.")
    body("Applied to a saved field, the diagnostic is:")
    block(["from scipy.ndimage import gaussian_filter",
           "from gaussian_fwi import GaussianField",
           "",
           "model = GaussianField.load(\"results/marmousi/fit/field.pt\")",
           "velocity = model().detach().cpu().numpy()",
           "macro = gaussian_filter(",
           "    velocity, sigma=80.0 / model.grid.spacing, mode=\"reflect\"",
           ")",
           "detail = velocity - macro"])
    body("Here 80 m is the smoothing standard deviation, converted to grid samples. It is not a "
         "sharp wavelength cutoff. The boundary rule is explicitly reflect; on a small domain it "
         "matters. This diagnostic is not part of the training objective.")
    picture(FIG / "macro-detail.png",
            "A compact local velocity change and its wider, lower-amplitude smoothed response.")
    caption("**Figure 8. A local edit has a wider effect on the smoothed model.** Left: turn on one "
            "fine component while keeping the background and broad component fixed. Right: smooth "
            "the resulting velocity change with an 80 m Gaussian. The 120 m/s local peak becomes a "
            "35.8 m/s smoothed peak, and the response extends beyond the fine component's support. "
            "Both panels use the same vertical scale. Constructed 1D example.")
    body("The local edit is zero outside its support, but its smoothed response extends farther. "
         "Holding broad-component parameters fixed does not hold the smoothed model fixed. Broad and "
         "fine components are not an exact low/high-frequency decomposition.")

    heading("13 How the code is checked")
    body("`python -m unittest discover -s tests/unit` runs 69 checks: analytic derivatives against "
         "finite differences, acoustic gradients against directional differences, density-control "
         "transactions and their rollback, optimizer-state transfer across clone/split/prune, "
         "restart identity, data separation, and saved-field replay.")
    body("`python models/verify.py` exercises the shipped interface end to end and reports twenty "
         "checks: profile validation, fitting, independent replay, provenance, reload, off-grid "
         "physical queries, pause and continue, stage restart and its identity rejection, output "
         "protection, rejection of a bundle carrying a reference velocity, train/test separation, "
         "float32 and float64, 2D and 3D, plotting, and the command line.")
    body("One check is worth naming explicitly. A test multiplies the test-partition traces by one "
         "hundred and refits; the selected model is bitwise identical. Held-out data cannot reach the "
         "optimizer or the checkpoint selection.")
    body("These are software correctness checks. They establish that the implementation does what it "
         "claims, not that the method is accurate on new data.")

    heading("Figure definitions")
    body("Figures 1, 2, 4 and 8 illustrate explicitly defined constructed examples; they are not "
         "inversion results. Figure 3 uses a synthetic demonstration fit. Figures 5, 6 and 7 use the "
         "Marmousi fits described in sections 10 and 11, under the realistic conditions tabulated "
         "there.")
    body("Figure 1: center (x, z) = (400, 300) m; principal widths 145 and 58 m; tilt 25° from +x "
         "toward +z. Contours have q = 1 and q = 4. The profiles show alternative amplitudes +240 "
         "and −180 m/s, using offset divided by the corresponding principal width. The code stores "
         "equivalent Cholesky parameters, rather than an explicit angle and principal widths.",
         style="List Bullet")
    body("Figure 2: 1D domain 0–1,200 m, sampled every 1 m. Components have centers 400 and 800 m, "
         "widths 120 and 80 m, and amplitudes +240 and −180 m/s. The background is 2,500 m/s. The "
         "soft bounds are 1,500–4,500 m/s with scale 20 m/s.", style="List Bullet")
    body("Figure 3: shot 4, receiver 34 (a training receiver), and the root mean square residual "
         "across all 40 training receivers and 8 shots at each time, from results/marmousi. Both "
         "panels are low-passed at 20 Hz, the highest cutoff the fit used.", style="List Bullet")
    body("Figure 4: a 1D parent centered at zero, width 30 m, amplitude +10 m/s. The illustrative "
         "split centers are −22 and +26 m; both children have width 30 / 1.6 m and amplitude "
         "+10 m/s. These chosen positions illustrate one possible geometry; the actual algorithm "
         "samples centers. Curves show raw contributions before bounding.", style="List Bullet")
    body("Figure 5: reference, initial and selected velocity from `results/marmousi`, with the "
         "signed difference on its own scale. 8 shots, 68 receivers at 10 m spacing, 800 samples at "
         "1 ms, signal-to-noise 10, modelled on a 2× finer grid with an 8% source error. 240 "
         "updates over cumulative cutoffs 4 / 7 / 12 / 20 Hz.", style="List Bullet")
    body("Figure 6: the same bundle fitted at 8 × 8, 16 × 16 and 32 × 32 seed lattices, all other "
         "settings equal, from `results/capacity`. The dashed line at 300 m marks the approximate "
         "limit of diving-wave illumination for this survey.", style="List Bullet")
    body("Figure 7: the same bundle fitted at an 8 × 8 lattice (tv = 1e-4), a 32 × 32 lattice "
         "(tv = 1e-4), and a 32 × 32 lattice at tv = 1e-2, from results/capacity/seeds_08, "
         "results/marmousi and results/tv_0.01. Roughness is the mean absolute second difference "
         "of the field, a scale-free speckle measure. All three reach the same waveform fit.",
         style="List Bullet")
    body("Figure 8: 1D domain 0–1,200 m, sampled every 2 m; the display crops to 300–1,000 m. "
         "Background 2,500 m/s. The fixed broad component has center 600 m, width 220 m, and "
         "amplitude +160 m/s. The fine component has center 650 m, width 25 m, and amplitude "
         "+120 m/s. Smoothing has standard deviation 80 m, truncation at six standard deviations, "
         "and nearest-value boundary extension.", style="List Bullet")

    settings = doc.settings.element
    settings.append(OxmlElement("w:doNotAutoCompressPictures"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Gaussian FWI Walkthrough.docx")
    parser.add_argument("--figures", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    written = build(args.output, args.figures, args.results)
    print(f"Saved {written}")


if __name__ == "__main__":
    main()
