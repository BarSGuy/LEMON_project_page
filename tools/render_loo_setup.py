"""Render the six-model leave-one-out evaluation setup as a GIF and PNG.

Usage: python project_page/tools/render_loo_setup.py
The six one-second frames rotate the held-out model. Separate monitors are
trained for the hallucination and jailbreak tasks; this is not joint training.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from render_online_demo import arrow, blend, label


OUT_DIR = Path(__file__).resolve().parents[1] / "static" / "images"
WIDTH, HEIGHT = 1800, 730
MODELS = (
    "Llama-2-7B", "Llama-2-13B", "Mistral-7B",
    "Mistral-Nemo-12B", "Qwen2.5-3B", "Qwen2.5-7B",
)
# Colors identify models, so each keeps its color when its role changes.
MODEL_COLORS = ("#2f6bd0", "#b95715", "#087f73", "#8750ae", "#b7476b", "#445b78")
INK = "#1f2937"
MUTED = "#667085"
BLUE = "#3273dc"
RULE = "#e2e7ef"


def render(held_out):
    frame = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(frame)
    held_color = MODEL_COLORS[held_out]
    held_ink = blend(held_color, "#000000", 0.15)
    label(draw, (44, 33), "Zero-shot evaluation on unseen LLMs", 43, bold=True)
    label(draw, (46, 103),
          "For a fair evaluation, train on five models, hold out the sixth, and repeat across all six.", 28, MUTED)
    draw.rounded_rectangle((1510, 36, 1758, 96), radius=20,
                           fill=blend("white", held_color, 0.08),
                           outline=blend("white", held_color, 0.45), width=2)
    label(draw, (1634, 66), f"split {held_out + 1} / 6", 27,
          held_ink, bold=True, anchor="mm")
    draw.line((44, 151, 1758, 151), fill=RULE, width=2)

    # Only source-model data and labels enter monitor training.
    draw.rounded_rectangle((40, 181, 590, 540), radius=21,
                           fill="white", outline=RULE, width=2)
    label(draw, (72, 209), "Train on five LLMs", 31, bold=True)
    training = [(i, name) for i, name in enumerate(MODELS) if i != held_out]
    for row, (index, name) in enumerate(training):
        y = 258 + row * 47
        color = MODEL_COLORS[index]
        draw.rounded_rectangle((72, y, 558, y + 38), radius=9,
                               fill=blend("white", color, 0.08),
                               outline=blend("white", color, 0.30), width=1)
        draw.ellipse((86, y + 12, 100, y + 26), fill=color)
        label(draw, (116, y + 19), name, 25, INK, anchor="lm")
    label(draw, (315, 514), "Attention features + training labels", 22,
          MUTED, anchor="mm")

    # One task-specific trained monitor is applied to the unseen model.
    arrow(draw, [(609, 360), (704, 360)], BLUE, width=4, head=14)
    label(draw, (656, 328), "train", 25, BLUE, anchor="mm")
    draw.rounded_rectangle((725, 269, 1105, 462), radius=25,
                           fill="#fff8d9", outline="#d4a300", width=3)
    label(draw, (915, 321), "LEMON", 47, "#a07c00", bold=True, anchor="mm")
    label(draw, (915, 425), "Freeze after training", 23, MUTED, anchor="mm")
    arrow(draw, [(1125, 360), (1220, 360)], BLUE, width=4, head=14)
    label(draw, (1172, 328), "apply", 25, BLUE, anchor="mm")

    draw.rounded_rectangle((1240, 181, 1760, 540), radius=21,
                           fill="white", outline=RULE, width=2)
    label(draw, (1272, 209), "Evaluate on the sixth", 31, bold=True)
    label(draw, (1500, 279), "Held-out LLM", 25, held_ink, anchor="mm")
    draw.rounded_rectangle((1272, 310, 1728, 400), radius=15,
                           fill=blend("white", held_color, 0.08),
                           outline=held_color, width=3)
    label(draw, (1500, 355), MODELS[held_out], 31, held_ink, bold=True, anchor="mm")
    label(draw, (1500, 448), "No target training labels", 25, INK, anchor="mm")
    label(draw, (1500, 493), "No retraining or adaptation", 23, MUTED, anchor="mm")

    label(draw, (900, 588), "Rotate the held-out LLM across all six splits", 28,
          INK, anchor="mm")
    for index in range(6):
        x, y = 740 + 64 * index, 648
        current = index == held_out
        color = MODEL_COLORS[index]
        if current:
            draw.ellipse((x - 30, y - 30, x + 30, y + 30),
                         outline=color, width=3)
        draw.ellipse((x - 23, y - 23, x + 23, y + 23),
                     fill=color if current else blend("white", color, 0.08),
                     outline=color if current else blend("white", color, 0.45), width=2)
        label(draw, (x, y), str(index + 1), 24,
              "white" if current else blend(color, "#000000", 0.15),
              bold=current, anchor="mm")

    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    originals = [render(index) for index in range(6)]
    originals[0].save(OUT_DIR / "lemon_loo_setup.png")
    # Sample every split so the shared palette preserves all six model colors.
    palette_source = Image.new("RGB", (WIDTH, HEIGHT * len(originals)))
    for index, frame in enumerate(originals):
        palette_source.paste(frame, (0, HEIGHT * index))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
              for frame in originals]
    output = OUT_DIR / "lemon_loo_setup.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=1000, loop=0, disposal=1, optimize=False)
    with Image.open(output) as gif:
        assert gif.n_frames == 6 and gif.size == (WIDTH, HEIGHT)
        assert gif.info["loop"] == 0
        for index in range(gif.n_frames):
            gif.seek(index)
            assert gif.info["duration"] == 1000
        print(f"{output.name}: six 1s splits, 6s loop, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
