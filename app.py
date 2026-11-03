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

OPENING = "Upload a leaf photo."
BATCH_OPENING = "Drop several photos. They are scanned together."


def one_leaf(image):
    """Classify one photo. Return the headline, the image pair, and the scores."""
    if image is None:
        return OPENING, None, {}
    result = leaf.answer(image)
    headline = (
        f"# {result['pretty']}\n"
        f"## {result['confidence']:.1%} confident\n"
        f"answered in {result['ms']:.0f} ms"
    )
    return headline, leaf.side_by_side(result), result["scores"]


def batch_scan(files):
    """Classify many photos in one forward. Return the summary and the grid."""
    if not files:
        return BATCH_OPENING, []
    paths = [f if isinstance(f, str) else f.name for f in files]
    rows, elapsed = leaf.batch_answer(paths)
    diseased = sum(1 for row in rows if not row["healthy"])
    summary = (
        f"# scanned {len(rows)} leaves in {elapsed / 1000:.2f} s\n"
        f"## {diseased} diseased, {len(rows) - diseased} healthy\n"
        f"{elapsed / len(rows):.0f} ms per leaf"
    )
    grid = [
        (path, f"{row['pretty']} - {row['confidence']:.0%}")
        for path, row in zip(paths, rows)
    ]
    return summary, grid


with gr.Blocks(theme=gr.themes.Citrus(), title="Leaf check") as demo:
    gr.Markdown("# Leaf check\nUpload a photo of an apple or grape leaf.")

    with gr.Tabs():
        with gr.Tab("One leaf"):
            with gr.Row():
                photo = gr.Image(type="numpy", label="Your photo", height=320)
                pair = gr.Image(
                    label="What the model saw, and where it looked",
                    height=320,
                )
            verdict = gr.Markdown(OPENING)
            scores = gr.Label(num_top_classes=8, label="Every class")
            photo.change(one_leaf, photo, [verdict, pair, scores])

        with gr.Tab("Many leaves"):
            drop = gr.File(
                file_count="multiple",
                type="filepath",
                label="Drop several photos",
            )
            count = gr.Markdown(BATCH_OPENING)
            grid = gr.Gallery(label="Every leaf", columns=4, height=460)
            drop.change(batch_scan, drop, [count, grid])


if __name__ == "__main__":
    demo.launch()
