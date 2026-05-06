"""
_make_linkedin_banner.py — Build the LinkedIn cover banner
============================================================

Generates a 1584x396 PNG suitable for upload as a LinkedIn cover banner.

LinkedIn safe area: profile photo sits at roughly (110, H-100) and
overlaps the bottom-left ~280x180 region, so all important text lives
in the right two-thirds of the canvas.

Run:
    cd ai-related-work
    pip install Pillow
    python docs/_make_linkedin_banner.py

Output:
    docs/linkedin-banner.png
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "linkedin-banner.png"

W, H = 1584, 396                # LinkedIn cover-banner spec

# Brand palette (must match docs/index.html so the visual identity is
# consistent across portfolio site, social preview, and LinkedIn cover).
BG_TOP = (13, 17, 23)
BG_BOT = (28, 34, 56)
ACCENT = (110, 168, 254)        # primary accent (#6ea8fe)
TEXT_PRIMARY = (245, 248, 255)
TEXT_SECONDARY = (180, 192, 215)
TEXT_MUTED = (140, 152, 176)
PILL_BG = (30, 38, 60)
PILL_BORDER = (74, 86, 120)

# Windows system fonts
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_SEMI = "C:/Windows/Fonts/segoeuisl.ttf"


# ──────────────────────────────────────────────────────────────────────
# Module 1: Background gradient
# ──────────────────────────────────────────────────────────────────────
def make_gradient_bg() -> Image.Image:
    """Vertical gradient from BG_TOP (top) to BG_BOT (bottom)."""
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
# Module 2: Soft accent glow on the right
# ──────────────────────────────────────────────────────────────────────
def add_glow(img: Image.Image) -> Image.Image:
    """Layered radial accent at upper-right; gives the banner depth."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    cx, cy = int(W * 0.80), int(H * 0.30)
    for r in range(560, 0, -22):
        alpha = max(0, int(34 - (560 - r) / 18))
        od.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=(*ACCENT, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# Module 3: Helpers
# ──────────────────────────────────────────────────────────────────────
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def text_w(draw: ImageDraw.ImageDraw, s: str, f) -> int:
    bbox = draw.textbbox((0, 0), s, font=f)
    return bbox[2] - bbox[0]


def draw_pill(draw: ImageDraw.ImageDraw, x: int, y: int, label: str,
              f: ImageFont.FreeTypeFont) -> int:
    """Rounded pill containing `label`; returns the right edge x-coord."""
    pad_x, pad_y = 14, 8
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

    f_eyebrow = font(FONT_SEMI, 20)
    f_title = font(FONT_BOLD, 68)
    f_tagline = font(FONT_REG, 26)
    f_pill = font(FONT_SEMI, 18)
    f_url = font(FONT_REG, 20)

    # Profile photo on LinkedIn overlaps the bottom-left ~280x180 px,
    # so all critical text starts at x=520 to clear that area.
    main_x = 520

    # ── Eyebrow ──
    eyebrow = "PRODUCTION AI  ·  HYDERABAD, INDIA"
    draw.text((main_x, 70), eyebrow, font=f_eyebrow, fill=ACCENT)

    # ── Main title (no name, just the role) ──
    draw.text((main_x, 105), "AI / ML Engineer", font=f_title, fill=TEXT_PRIMARY)

    # ── Tagline ──
    tagline = "LLMs   ·   RAG   ·   Multi-Agent Systems   ·   Computer Vision"
    draw.text((main_x, 200), tagline, font=f_tagline, fill=TEXT_SECONDARY)

    # ── Tech-stack pills ──
    pills = ["Azure ML", "PyTorch", "FAISS", "CrewAI", "NVIDIA NIM", "Hugging Face"]
    y_pills = 250
    x = main_x
    gap = 10
    for p in pills:
        x = draw_pill(draw, x, y_pills, p, f_pill) + gap

    # ── URL footer (bottom-right) ──
    url = "github.com/deepak1212194"
    url_w_px = text_w(draw, url, f_url)
    draw.text((W - url_w_px - 60, H - 50), url, font=f_url, fill=TEXT_MUTED)

    # ── Decorative dot grid (right side, top) ──
    for row in range(5):
        for col in range(7):
            cx = W - 200 + col * 22
            cy = 60 + row * 22
            r = 3
            alpha_t = 1.0 - (row + col) / 12
            alpha_t = max(0.15, alpha_t)
            color = (
                int(ACCENT[0] * alpha_t + BG_BOT[0] * (1 - alpha_t)),
                int(ACCENT[1] * alpha_t + BG_BOT[1] * (1 - alpha_t)),
                int(ACCENT[2] * alpha_t + BG_BOT[2] * (1 - alpha_t)),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # ── Subtle accent stripe at bottom-left to fill the otherwise-empty
    #    area where the profile photo will sit. Stays well below where
    #    the photo actually overlaps so it doesn't fight visually.
    for i in range(3):
        x0 = 60 + i * 7
        y0 = 100 + i * 6
        draw.rounded_rectangle(
            [x0, y0, x0 + 320, y0 + 4],
            radius=2,
            fill=(ACCENT[0], ACCENT[1], ACCENT[2], 200),
        )

    return img


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    img = compose()
    img.save(OUT_PATH, "PNG", optimize=True)
    print(f"[SAVE] {OUT_PATH}  ({W}x{H})")


if __name__ == "__main__":
    main()
