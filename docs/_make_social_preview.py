"""
_make_social_preview.py — Build the GitHub social-preview PNG
==============================================================

Generates a 1280x640 PNG suitable for upload to:
   GitHub repo > Settings > Social preview

Run:
    cd ai-related-work
    pip install Pillow
    python docs/_make_social_preview.py

Output:
    docs/social-preview.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "social-preview.png"

W, H = 1280, 640

# Brand palette (matches docs/index.html)
BG_TOP = (13, 17, 23)        # GitHub dark
BG_BOT = (28, 34, 56)        # subtle blue tint
ACCENT = (110, 168, 254)     # #6ea8fe
ACCENT_DARK = (79, 140, 255)
TEXT_PRIMARY = (245, 248, 255)
TEXT_SECONDARY = (180, 192, 215)
TEXT_MUTED = (140, 152, 176)
PILL_BG = (30, 38, 60)
PILL_BORDER = (74, 86, 120)

# Windows system fonts (also work on most Linux/macOS via Segoe substitution)
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_SEMI = "C:/Windows/Fonts/segoeuisl.ttf"


# ──────────────────────────────────────────────────────────────────────
# Module 1: Vertical-gradient background
# ──────────────────────────────────────────────────────────────────────
def make_gradient_bg() -> Image.Image:
    """Vertical gradient from BG_TOP to BG_BOT, drawn row-by-row."""
    img = Image.new("RGB", (W, H), BG_TOP)
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img


# ──────────────────────────────────────────────────────────────────────
# Module 2: Soft accent glow at top-right
# ──────────────────────────────────────────────────────────────────────
def add_glow(img: Image.Image) -> Image.Image:
    """Soft radial accent at upper-right; layered alpha-composited circles."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    cx, cy = int(W * 0.85), int(H * 0.05)
    for r in range(700, 0, -25):
        # falloff: brighter near the centre, transparent at the edge
        alpha = max(0, int(46 - (700 - r) / 16))
        od.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=(*ACCENT, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# Module 3: Text helpers
# ──────────────────────────────────────────────────────────────────────
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def text_w(draw: ImageDraw.ImageDraw, s: str, f) -> int:
    bbox = draw.textbbox((0, 0), s, font=f)
    return bbox[2] - bbox[0]


def draw_pill(draw: ImageDraw.ImageDraw, x: int, y: int, label: str,
              f: ImageFont.FreeTypeFont) -> int:
    """Draw a rounded pill containing `label`; return the pill's right edge."""
    pad_x, pad_y = 18, 10
    tw = text_w(draw, label, f)
    th = f.size + 2
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    draw.rounded_rectangle([x, y, x + w, y + h],
                           radius=h // 2,
                           fill=PILL_BG,
                           outline=PILL_BORDER,
                           width=2)
    draw.text((x + pad_x, y + pad_y - 2), label, font=f, fill=TEXT_SECONDARY)
    return x + w


# ──────────────────────────────────────────────────────────────────────
# Module 4: Compose
# ──────────────────────────────────────────────────────────────────────
def compose() -> Image.Image:
    img = make_gradient_bg()
    img = add_glow(img)
    draw = ImageDraw.Draw(img)

    f_huge = font(FONT_BOLD, 96)
    f_tag = font(FONT_REG, 32)
    f_pill = font(FONT_SEMI, 22)
    f_url = font(FONT_REG, 22)

    # --- Eyebrow ("AI / ML ENGINEER · HYDERABAD") ---
    eyebrow = "AI / ML ENGINEER  ·  HYDERABAD, INDIA"
    f_eyebrow = font(FONT_SEMI, 22)
    draw.text((80, 110), eyebrow, font=f_eyebrow, fill=ACCENT)

    # --- Big name ---
    draw.text((80, 150), "Deepak Chaudhary", font=f_huge, fill=TEXT_PRIMARY)

    # --- Tagline ---
    tagline = "Production AI · LLMs · RAG · Multi-Agent · Computer Vision"
    draw.text((80, 270), tagline, font=f_tag, fill=TEXT_SECONDARY)

    # --- Pills (signal stack) ---
    pills = [
        "4+ yrs production AI",
        "NVIDIA GTC 2026",
        "Granted Indian Patent",
        "ISI Kolkata",
    ]
    x, y = 80, 360
    gap = 12
    for p in pills:
        x = draw_pill(draw, x, y, p, f_pill) + gap

    # --- Footer URL ---
    url = "github.com/deepak1212194/ai-related-work"
    draw.text((80, H - 70), url, font=f_url, fill=TEXT_MUTED)

    # --- Right-side accent dot grid (decorative) ---
    for row in range(8):
        for col in range(6):
            cx = W - 200 + col * 22
            cy = 200 + row * 22
            r = 3
            alpha_t = 1.0 - (row + col) / 14
            color = (
                int(ACCENT[0] * alpha_t + BG_BOT[0] * (1 - alpha_t)),
                int(ACCENT[1] * alpha_t + BG_BOT[1] * (1 - alpha_t)),
                int(ACCENT[2] * alpha_t + BG_BOT[2] * (1 - alpha_t)),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    return img


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    img = compose()
    img.save(OUT_PATH, "PNG", optimize=True)
    print(f"[SAVE] {OUT_PATH}  ({W}x{H})")
    print("[NEXT] Upload at: https://github.com/deepak1212194/ai-related-work/settings"
          "  →  Social preview")


if __name__ == "__main__":
    main()
