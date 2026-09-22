"""
Post-process a raw Open3D capture (from
opencood.visualization.vis_utils.save_o3d_visualization_cropped) into a
report-ready figure: crop tight to the actual content, thicken the
red/green box outlines (Open3D's line_width render option is capped to
1px by most GL drivers, including the Mesa software rasterizer used for
headless Kaggle rendering -- verified: no visible change locally either),
and burn in a title (model name + frame id) and a legend explaining what
red/green mean.

Standalone image-processing utility (PIL + numpy only) -- does not touch
OpenCOOD/model code, deliberately kept out of vis_utils.py since this is
report-figure formatting, not a general visualization capability.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Same colors vis_utils.py draws boxes in: bbx2oabb(pred, color=(1,0,0))
# is RED, bbx2oabb(gt, color=(0,1,0)) is GREEN -- confirmed by reading
# that code, not assumed.
PRED_COLOR = (220, 30, 30)
GT_COLOR = (30, 150, 40)


def _find_font(size):
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _dilate_mask(mask, iterations=2):
    """4-connectivity binary dilation via array shifts -- no scipy needed."""
    out = mask.copy()
    for _ in range(iterations):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def thicken_box_lines(arr, iterations=2):
    """
    arr: (H, W, 3) uint8 RGB array, background white/near-white, box
    lines drawn in near-pure red / near-pure green. Dilates each
    color's pixels by `iterations` and repaints them solid -- guarantees
    visible thick outlines regardless of what the renderer's own
    line_width did.
    """
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    red_mask = (r > 140) & (r - g > 60) & (r - b > 60)
    green_mask = (g > 100) & (g - r > 40) & (g - b > 40)

    red_mask = _dilate_mask(red_mask, iterations)
    green_mask = _dilate_mask(green_mask, iterations)
    # red wins where both would overlap after dilation (rare, edge pixels)
    green_mask = green_mask & ~red_mask

    out = arr.copy()
    out[red_mask] = PRED_COLOR
    out[green_mask] = GT_COLOR
    return out


def autocrop_to_content(img, pad_frac=0.08, min_pad_px=20):
    """
    Crop a white-background PIL image to its non-white content bounding
    box, plus padding. Falls back to the full image if nothing is found
    (e.g. an empty frame).
    """
    arr = np.asarray(img.convert("RGB"))
    non_white = np.any(arr < 245, axis=-1)
    ys, xs = np.where(non_white)
    if len(xs) == 0:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = max(int((x1 - x0) * pad_frac), min_pad_px)
    pad_y = max(int((y1 - y0) * pad_frac), min_pad_px)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(arr.shape[1], x1 + pad_x)
    y1 = min(arr.shape[0], y1 + pad_y)
    return img.crop((x0, y0, x1, y1))


def annotate_frame(raw_png_path, out_png_path, title, line_thicken_iters=2,
                   upscale_to_width=1400):
    """
    Full pipeline: thicken box lines -> crop to content -> burn in a
    title bar + legend -> save.

    Parameters
    ----------
    raw_png_path : str
        Path to the raw Open3D capture.
    out_png_path : str
        Where to save the annotated figure.
    title : str
        Text for the title bar, e.g. "KAN-ViT -- frame 0800".
    """
    img = Image.open(raw_png_path).convert("RGB")
    arr = np.asarray(img)
    arr = thicken_box_lines(arr, iterations=line_thicken_iters)
    img = Image.fromarray(arr)

    img = autocrop_to_content(img)

    if upscale_to_width and img.width < upscale_to_width:
        ratio = upscale_to_width / img.width
        img = img.resize((upscale_to_width, int(img.height * ratio)),
                         Image.LANCZOS)

    title_h = max(56, img.width // 22)
    legend_h = max(44, img.width // 28)
    canvas = Image.new("RGB", (img.width, img.height + title_h + legend_h),
                       (255, 255, 255))
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    title_font = _find_font(int(title_h * 0.5))
    legend_font = _find_font(int(legend_h * 0.42))

    draw.text((16, title_h // 2), title, fill=(20, 20, 20), font=title_font,
             anchor="lm")

    legend_y = title_h + img.height + legend_h // 2
    swatch = int(legend_h * 0.4)
    x = 16
    draw.rectangle([x, legend_y - swatch // 2, x + swatch, legend_y + swatch // 2],
                   fill=PRED_COLOR)
    x += swatch + 10
    draw.text((x, legend_y), "Prediction", fill=(20, 20, 20), font=legend_font,
              anchor="lm")
    x += draw.textlength("Prediction", font=legend_font) + 40

    draw.rectangle([x, legend_y - swatch // 2, x + swatch, legend_y + swatch // 2],
                   fill=GT_COLOR)
    x += swatch + 10
    draw.text((x, legend_y), "Ground truth", fill=(20, 20, 20), font=legend_font,
              anchor="lm")

    canvas.save(out_png_path)
    return out_png_path


def side_by_side(png_paths, out_path, gap=24, bg=(255, 255, 255)):
    """Horizontally concatenate already-annotated PNGs (matched height)."""
    imgs = [Image.open(p).convert("RGB") for p in png_paths]
    h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != h:
            ratio = h / im.height
            im = im.resize((int(im.width * ratio), h), Image.LANCZOS)
        resized.append(im)
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, h), bg)
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    canvas.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    annotate_frame(sys.argv[1], sys.argv[2],
                   sys.argv[3] if len(sys.argv) > 3 else "")
