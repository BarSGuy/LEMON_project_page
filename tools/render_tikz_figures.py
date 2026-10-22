"""Render TikZ figures from the paper sources into web images for the project page.

Each figure is compiled on its own page with the paper's full preamble (so all
macros and colors match), with the float wrapper and caption removed, then
cropped to its ink and saved as SVG (vector) and PNG (fallback/preview).

The paper sources are only READ, never modified.

Usage (needs tectonic + pymupdf; the base conda python has pymupdf):
    python tools/render_tikz_figures.py --paper_dir ../ICLR_2027 \
        --figs fig_head_perm_symmetry fig_wreath_action fig_attention_tensor_symmetry
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "static" / "images"

# Turn floats into plain centered boxes and drop captions/labels, so only the drawing remains.
OVERRIDES = r"""
\iclrfinalcopy  % turns off the review-mode margin line numbers
\makeatletter
\renewenvironment{wrapfigure}[3][]{\par\noindent\begin{minipage}{#3}\centering}{\end{minipage}\par}
\renewenvironment{figure}[1][]{\par\noindent\begin{minipage}{\textwidth}\centering}{\end{minipage}\par}
\renewenvironment{figure*}[1][]{\par\noindent\begin{minipage}{\textwidth}\centering}{\end{minipage}\par}
\renewcommand{\caption}[2][]{}
\renewcommand{\label}[1]{}
\makeatother
\pagestyle{empty}
"""


def build_tex(paper_tex: Path, fig_path: Path) -> str:
    src = paper_tex.read_text()
    preamble = src[: src.index(r"\begin{document}")]
    return (
        preamble
        + OVERRIDES
        + "\\begin{document}\n"
        + f"\\input{{{fig_path.as_posix()}}}\n"
        + "\\end{document}\n"
    )


def render(paper_dir: Path, fig: str, scale: float, pad: float) -> None:
    fig_path = paper_dir / "tkiz_figures" / f"{fig}.tex"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fig.tex").write_text(build_tex(paper_dir / "main_iclr_2027.tex", fig_path))
        subprocess.run(
            ["tectonic", "-Z", f"search-path={paper_dir}", "fig.tex"],
            cwd=tmp, check=True, capture_output=True,
        )
        doc = pymupdf.open(tmp / "fig.pdf")
        page = doc[0]
        # crop to the union of everything drawn: text, vector drawings, images
        rects = [b[:4] for b in page.get_text("blocks")]
        rects += [d["rect"] for d in page.get_drawings()]
        rects += [page.get_image_bbox(i) for i in page.get_images(full=True)]
        bbox = pymupdf.Rect(rects[0])
        for r in rects[1:]:
            bbox |= pymupdf.Rect(r)
        bbox = pymupdf.Rect(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad) & page.rect
        page.set_cropbox(bbox)

        out_svg = OUT_DIR / f"{fig}.svg"
        out_png = OUT_DIR / f"{fig}.png"
        out_svg.write_text(page.get_svg_image(text_as_path=True))
        page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(out_png)
        print(f"{fig}: {bbox.width:.0f}x{bbox.height:.0f}pt -> {out_svg.name}, {out_png.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_dir", type=Path, required=True)
    ap.add_argument("--figs", nargs="+", required=True)
    ap.add_argument("--scale", type=float, default=4.0, help="PNG zoom (4 = 288 dpi)")
    ap.add_argument("--pad", type=float, default=4.0, help="crop padding in pt")
    args = ap.parse_args()
    assert shutil.which("tectonic"), "tectonic not on PATH"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fig in args.figs:
        render(args.paper_dir.resolve(), fig, args.scale, args.pad)


if __name__ == "__main__":
    main()
