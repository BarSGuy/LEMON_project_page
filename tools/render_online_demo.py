"""Render a schematic of LEMON's incremental attention graph.

Run from any directory: python project_page/tools/render_online_demo.py
Outputs an animated GIF and a static PNG for reduced-motion preferences.
Attention weights are illustrative, not measurements from an LLM. Each new
query row n contributes edges m -> n for m < n; the diagonal represents the
new node's self feature. Existing rows, graph features, and scores are retained.
"""

from functools import lru_cache
import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parents[1] / "static" / "images"
WIDTH, HEIGHT = 1800, 900
TOKENS = ("The", "capital", "of", "France", "is", "Paris")
N = len(TOKENS)
STEPS = 18
FRAME_MS = 50
FINAL_HOLD_MS = 1000

INK = "#1f2937"
MUTED = "#667085"
RULE = "#e2e7ef"
BLUE = "#3273dc"
GREEN = "#208547"
PURPLE = "#7555ba"
PAST_EDGE = "#c5cee0"
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@lru_cache(maxsize=None)
def font(size, bold=False):
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
    )


def rgb(color):
    return ImageColor.getrgb(color)


def blend(first, second, amount):
    a = rgb(first) if isinstance(first, str) else first
    b = rgb(second) if isinstance(second, str) else second
    return tuple(round(x + (y - x) * amount) for x, y in zip(a, b))


def ease(value):
    value = max(0, min(1, value))
    return value * value * (3 - 2 * value)


def label(draw, xy, text, size=26, fill=INK, bold=False, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def arrow(draw, points, color, width=3, head=11):
    draw.line(points, fill=color, width=width, joint="curve")
    if len(points) < 2:
        return
    tip, previous = points[-1], points[-2]
    dx, dy = tip[0] - previous[0], tip[1] - previous[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    draw.polygon(
        [tip,
         (tip[0] - head * ux - head * 0.5 * uy,
          tip[1] - head * uy + head * 0.5 * ux),
         (tip[0] - head * ux + head * 0.5 * uy,
          tip[1] - head * uy - head * 0.5 * ux)],
        fill=color,
    )


def edge(draw, source, target, progress=1, color=PAST_EDGE, width=3):
    if progress <= 0:
        return
    start = (source + 14, 475)
    end = (target - 14, 475)
    control = ((source + target) / 2, 415 - (target - source) * 0.33)
    points = []
    for step in range(51):
        t = progress * step / 50
        points.append((
            (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0],
            (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1],
        ))
    arrow(draw, points, color, width=width)


def attention_weights(row):
    values = [2 + ((7 * col + 3 * row) % 11) for col in range(row + 1)]
    return [value / sum(values) for value in values]


def draw_attention(draw, count, progress):
    x0, y0, cell = 168, 325, 65
    row_progress = ease(progress / 0.35)
    label(draw, (72, 209), "Attention", 33, bold=True)
    label(draw, (72, 257), "Append the new query row", 25, MUTED)
    label(draw, (x0 + N * cell / 2, 304), "key token m", 23, MUTED, anchor="ms")
    for row in range(N):
        label(draw, (139, y0 + row * cell + cell / 2), str(row + 1), 24,
              BLUE if row == count - 1 else (INK if row < count else "#c6ccd5"),
              anchor="mm")
        weights = attention_weights(row)
        for col in range(N):
            left, top = x0 + col * cell, y0 + row * cell
            box = (left, top, left + cell - 2, top + cell - 2)
            draw.rectangle(box, fill="#fafbfc", outline="#edf0f5", width=1)
            if row < count and col < count and col > row:
                draw.rectangle(box, fill="#eef0f4")
            if row < count - 1 and col <= row:
                shade = blend("#f1edff", PURPLE, math.sqrt(weights[col]))
                draw.rectangle(box, fill=shade, outline="white", width=2)
            if row == count - 1 and col <= row:
                right = min(left + cell - 2, x0 + count * cell * row_progress)
                if right > left:
                    shade = blend("#f1edff", PURPLE, math.sqrt(weights[col]))
                    draw.rectangle((left, top, right, top + cell - 2),
                                   fill=shade, outline="white", width=2)
    for col in range(N):
        label(draw, (x0 + col * cell + cell / 2, y0 - 3), str(col + 1), 22,
              INK if col < count else "#c6ccd5", anchor="mb")
    if count > 1:
        draw.rectangle((x0 - 3, y0 - 3, x0 + (count - 1) * cell,
                        y0 + (count - 1) * cell), outline="#aeb8ca", width=2)
    if row_progress > 0:
        draw.rounded_rectangle(
            (x0 - 5, y0 + (count - 1) * cell - 5,
             x0 + count * cell * row_progress + 3, y0 + count * cell + 3),
            radius=5, outline=BLUE, width=4,
        )
    label(draw, (360, 757), f"new row n = {count}", 25, BLUE, bold=True, anchor="mm")


def draw_graph(draw, count, progress):
    xs = [810 + 170 * index for index in range(N)]
    graph_progress = ease((progress - 0.25) / 0.45)
    score_progress = ease((progress - 0.72) / 0.28)
    label(draw, (752, 209), "Incremental graph", 33, bold=True)
    label(draw, (752, 257), "Earlier tokens send messages to the new token", 25, MUTED)
    for target in range(count - 1):
        for source in range(target):
            edge(draw, xs[source], xs[target])
    if graph_progress > 0:
        weights = attention_weights(count - 1)
        for source in range(count - 1):
            edge(draw, xs[source], xs[count - 1], graph_progress,
                 BLUE, width=3 + round(5 * weights[source]))
    for index in range(count):
        current = index == count - 1
        visibility = graph_progress if current else 1
        if visibility <= 0:
            continue
        x, y, radius = xs[index], 500, 28
        fill = blend("white", "#eaf0ff", visibility)
        outline = blend("white", BLUE if current else "#8da4ce", visibility)
        if current:
            draw.ellipse((x - 38, y - 38, x + 38, y + 38),
                         outline=blend("white", "#c5d6ff", visibility), width=4)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=fill, outline=outline, width=4 if current else 3)
        label(draw, (x, y), str(index + 1), 26,
              blend("white", INK, visibility), bold=current, anchor="mm")
        label(draw, (x, 560), TOKENS[index], 25,
              blend("white", BLUE if current else INK, visibility), anchor="mm")
        score_visibility = score_progress if current else 1
        if score_visibility > 0:
            score_color = blend("white", GREEN, score_visibility)
            arrow(draw, [(x, 591), (x, 616)], score_color, width=3, head=8)
            draw.rounded_rectangle(
                (x - 49, 629, x + 49, 680), radius=15,
                fill=blend("white", "#edf8ef", score_visibility), outline=score_color, width=2,
            )
            label(draw, (x, 654), "ŷ" + str(index + 1).translate(SUBSCRIPTS),
                  29, score_color, anchor="mm")
    if score_progress > 0:
        label(draw, (1240, 711), f"emit ŷ{str(count).translate(SUBSCRIPTS)} now", 25,
              blend("white", GREEN, score_progress), bold=True, anchor="mm")
    label(draw, (1240, 757), "Earlier nodes and scores are retained", 24,
          GREEN if count > 1 else MUTED, anchor="mm")


def render(count, progress):
    frame = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(frame)
    label(draw, (44, 34), "LEMON is online", 47, bold=True)
    label(draw, (46, 102),
          "One new token → one attention row → one graph update → one score", 27, MUTED)
    draw.rounded_rectangle((1498, 38, 1758, 96), radius=20,
                           fill="#eef4ff", outline="#cfddf7", width=2)
    label(draw, (1628, 67), f"token {count} / {N}", 26, BLUE, bold=True, anchor="mm")
    draw.line((44, 151, 1758, 151), fill=RULE, width=2)
    draw.rounded_rectangle((40, 181, 680, 800), radius=21,
                           fill="white", outline=RULE, width=2)
    draw.rounded_rectangle((720, 181, 1760, 800), radius=21,
                           fill="white", outline=RULE, width=2)
    draw_attention(draw, count, progress)
    draw_graph(draw, count, progress)
    arrow(draw, [(691, 500), (709, 500)], BLUE, width=3, head=8)
    label(draw, (46, 833), "A new attention entry Aₙ,ₘ adds a forward edge m → n (m < n).", 25)
    label(draw, (1755, 850), "Schematic · one attention layer / head shown", 21,
          MUTED, anchor="rs")
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poster = render(N, 1)
    poster.save(OUT_DIR / "lemon_online_incremental.png")
    palette = poster.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames, durations = [], []
    for count in range(1, N + 1):
        for step in range(STEPS):
            progress = step / (STEPS - 1)
            frame = render(count, progress)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(200 if step == STEPS - 1 else FRAME_MS)
    durations[-1] = FINAL_HOLD_MS
    output = OUT_DIR / "lemon_online_incremental.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1, optimize=False)
    with Image.open(output) as gif:
        actual_durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            actual_durations.append(gif.info["duration"])
        assert gif.size == (WIDTH, HEIGHT)
        assert gif.info["loop"] == 0
        assert actual_durations[-1] == FINAL_HOLD_MS
        assert sum(actual_durations) == sum(durations)
        print(f"{output.name}: {gif.n_frames} frames, "
              f"{sum(actual_durations) / 1000:g}s loop, "
              f"{FINAL_HOLD_MS / 1000:g}s final hold, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
