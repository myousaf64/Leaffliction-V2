#!/usr/bin/env python3
"""Local web app for the leaf classifier.

Tab 1 takes one photo and shows the disease, the confidence, the heatmap, and
the score of every class.
Tab 2 takes many photos and scans them in one forward.

Run it:  python app.py
"""

import gradio as gr

import leaf

leaf.load()

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
      rel="stylesheet">
"""

# Gradio follows the browser dark mode preference. On a laptop set to dark the
# page renders dark whatever theme is passed. This forces light once.
FORCE_LIGHT = """
function forceLight() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""

CSS = """
:root {
    --ink: #1a2e1a;
    --muted: #6b7f6b;
    --line: #e6ebe6;
    --page: #f7f9f7;
    --good: #15803d;
    --good-bg: #f0fdf4;
    --warn: #b45309;
    --warn-bg: #fffbeb;
}

.gradio-container, body {
    background: var(--page) !important;
    font-family: Inter, -apple-system, "Segoe UI", system-ui, sans-serif !important;
    color: var(--ink);
}
.gradio-container { max-width: 1120px !important; }
footer { display: none !important; }

/* header */
#leaf-head { padding: 28px 4px 12px; }
#leaf-head h1 {
    font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
    margin: 0; color: var(--ink);
}
#leaf-head p { margin: 6px 0 0; color: var(--muted); font-size: 1.05rem; }

/* cards */
.leaf-card {
    background: #fff; border: 1px solid var(--line); border-radius: 16px;
    padding: 20px; box-shadow: 0 1px 2px rgba(20,40,20,.04),
                              0 8px 24px rgba(20,40,20,.05);
}

/* verdict */
.leaf-verdict {
    border-radius: 16px; padding: 24px 26px; border: 1px solid var(--line);
    background: #fff;
}
.leaf-verdict.good { background: var(--good-bg); border-color: #bbf7d0; }
.leaf-verdict.warn { background: var(--warn-bg); border-color: #fde68a; }
.leaf-verdict .name {
    font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em;
    line-height: 1.1; margin: 0;
}
.leaf-verdict.good .name { color: var(--good); }
.leaf-verdict.warn .name { color: var(--warn); }
.leaf-verdict .conf {
    font-size: 1.15rem; font-weight: 500; color: var(--ink);
    margin: 10px 0 0; opacity: .85;
}
.leaf-verdict .chip {
    display: inline-block; margin-top: 16px; padding: 5px 12px;
    border-radius: 999px; background: rgba(0,0,0,.05); color: var(--muted);
    font-size: .85rem; font-weight: 500;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
}
.leaf-verdict .idle {
    font-size: 1.15rem; color: var(--muted); margin: 0; font-weight: 500;
}

/* stat tiles */
.leaf-stat {
    background: #fff; border: 1px solid var(--line); border-radius: 16px;
    padding: 18px 20px; text-align: left;
}
.leaf-stat .big {
    font-size: 2rem; font-weight: 700; color: var(--ink);
    letter-spacing: -0.02em; line-height: 1.1;
}
.leaf-stat .lab {
    font-size: .82rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: .06em; font-weight: 600; margin-top: 4px;
}
.leaf-stat.good .big { color: var(--good); }
.leaf-stat.warn .big { color: var(--warn); }

/* gradio widgets */
.gradio-container .block { border-radius: 14px !important; }
.gradio-container label span { color: var(--muted) !important; font-weight: 600 !important; }
.gradio-container .tab-nav button {
    font-weight: 600 !important; font-size: 1rem !important;
}
"""

IDLE_ONE = '<div class="leaf-verdict"><p class="idle">Upload a leaf photo.</p></div>'
IDLE_MANY = (
    '<div class="leaf-stat"><div class="lab">Drop several photos to scan them together</div></div>'
)


def verdict_card(result):
    """Return the verdict card. Healthy is green, diseased is amber."""
    tone = "good" if leaf.is_healthy(result["name"]) else "warn"
    return (
        f'<div class="leaf-verdict {tone}">'
        f'<p class="name">{result["pretty"]}</p>'
        f'<p class="conf">{result["confidence"]:.1%} confident</p>'
        f'<span class="chip">answered in {result["ms"]:.0f} ms</span>'
        f"</div>"
    )


def stat_tiles(count, seconds, diseased):
    """Return the three batch stat tiles."""
    healthy = count - diseased
    return (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">'
        f'<div class="leaf-stat"><div class="big">{count}</div>'
        f'<div class="lab">leaves scanned</div></div>'
        f'<div class="leaf-stat"><div class="big">{seconds:.2f}s</div>'
        f'<div class="lab">total time</div></div>'
        f'<div class="leaf-stat {"warn" if diseased else "good"}">'
        f'<div class="big">{diseased}</div>'
        f'<div class="lab">diseased, {healthy} healthy</div></div>'
        "</div>"
    )


def one_leaf(image):
    """Classify one photo. Return the verdict, the image pair, and the scores."""
    if image is None:
        return IDLE_ONE, None, {}
    result = leaf.answer(image)
    return verdict_card(result), leaf.side_by_side(result), result["scores"]


def batch_scan(files):
    """Classify many photos in one forward. Return the stats and the grid."""
    if not files:
        return IDLE_MANY, []
    paths = [f if isinstance(f, str) else f.name for f in files]
    rows, elapsed = leaf.batch_answer(paths)
    diseased = sum(1 for row in rows if not row["healthy"])
    grid = [
        (path, f"{row['pretty']} - {row['confidence']:.0%}")
        for path, row in zip(paths, rows)
    ]
    return stat_tiles(len(rows), elapsed / 1000, diseased), grid


with gr.Blocks(title="Leaf check") as demo:
    gr.HTML(
        '<div id="leaf-head"><h1>Leaf check</h1>'
        "<p>Upload a photo of an apple or grape leaf. "
        "See what is wrong with it, and what the model looked at.</p></div>"
    )

    with gr.Tabs():
        with gr.Tab("One leaf"):
            with gr.Row():
                with gr.Column(scale=4):
                    photo = gr.Image(type="numpy", label="Your photo", height=330)
                with gr.Column(scale=6):
                    verdict = gr.HTML(IDLE_ONE)
            pair = gr.Image(
                label="What the model saw, and where it looked", height=300
            )
            scores = gr.Label(num_top_classes=8, label="Every class")
            photo.change(one_leaf, photo, [verdict, pair, scores])

        with gr.Tab("Many leaves"):
            drop = gr.File(
                file_count="multiple", type="filepath", label="Drop several photos"
            )
            count = gr.HTML(IDLE_MANY)
            grid = gr.Gallery(label="Every leaf", columns=4, height=480)
            drop.change(batch_scan, drop, [count, grid])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS, head=HEAD, js=FORCE_LIGHT)
