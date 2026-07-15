"""
generate_faces.py — AttendVision AI
=====================================
Generates geometric avatar face images for the 8 demo students.
These are NOT real people — they are programmatically drawn avatars.

Usage:
    pip install Pillow
    python generate_faces.py

Output: known_faces/STU001.jpg … STU008.jpg
"""

from PIL import Image, ImageDraw, ImageFilter
import os

OUTPUT_DIR = "known_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (name, filename, skin_rgb, hair_rgb)
STUDENTS = [
    ("Ahmed Khan",       "STU001.jpg", (210,160,120), (40, 25, 10)),
    ("Sara Malik",       "STU002.jpg", (240,195,155), (80, 50, 30)),
    ("Bilal Hassan",     "STU003.jpg", (175,130, 95), (30, 20, 10)),
    ("Ayesha Siddiqui",  "STU004.jpg", (245,205,170), (60, 35, 15)),
    ("Omar Farooq",      "STU005.jpg", (190,145,105), (20, 12,  5)),
    ("Zara Ahmed",       "STU006.jpg", (225,175,135), (90, 55, 25)),
    ("Usman Ali",        "STU007.jpg", (165,120, 85), (35, 22,  8)),
    ("Fatima Noor",      "STU008.jpg", (250,215,180), (50, 30, 10)),
]

SHIRT_COLORS = [
    (30,60,140),(140,30,30),(30,120,60),
    (100,30,120),(140,100,20),(20,100,120),
    (120,80,20),(60,60,60),
]


def draw_avatar_face(draw, cx, cy, skin, hair):
    """Draw a simple geometric face centred at (cx, cy)."""
    dark  = tuple(max(0, c - 30) for c in skin)
    brow  = tuple(max(0, c - 50) for c in skin)
    shade = tuple(max(0, c - 20) for c in skin)

    # Hair
    draw.ellipse([cx-90, cy-100, cx+90, cy+70], fill=hair)

    # Face oval
    draw.ellipse([cx-70, cy-80, cx+70, cy+90], fill=skin)

    # Ears (both sides, avoid zero-width ellipse)
    draw.ellipse([cx-82, cy-8, cx-60, cy+18], fill=skin)
    draw.ellipse([cx+60, cy-8, cx+82, cy+18], fill=skin)

    # Eyes
    for ex in [cx-28, cx+28]:
        draw.ellipse([ex-13, cy-18, ex+13, cy+2],  fill=(250,250,255))
        draw.ellipse([ex-8,  cy-14, ex+8,  cy-1],  fill=(55, 80,130))
        draw.ellipse([ex-5,  cy-12, ex+5,  cy-3],  fill=(10, 10, 10))
        draw.ellipse([ex+1,  cy-11, ex+4,  cy-8],  fill=(255,255,255))
        draw.arc([ex-13, cy-18, ex+13, cy+2], 195, 345, fill=dark, width=2)

    # Eyebrows
    for bx in [cx-28, cx+28]:
        draw.arc([bx-14, cy-35, bx+14, cy-16], 200, 340, fill=brow, width=4)

    # Nose
    draw.line([(cx, cy-5), (cx, cy+18)], fill=dark, width=2)
    draw.ellipse([cx-9, cy+12, cx+9, cy+26], fill=shade)

    # Mouth
    draw.arc([cx-20, cy+30, cx+20, cy+50], 15, 165, fill=(170,70,70), width=3)

    # Neck
    draw.rectangle([cx-18, cy+88, cx+18, cy+130], fill=skin)


def make_photo(name, filename, skin, hair, shirt):
    W, H = 400, 500
    img  = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(H):
        t = y / H
        r = int(200 + 45*t)
        g = int(210 + 40*t)
        b = int(225 + 25*t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Shirt / shoulders
    draw.ellipse([W//2-130, 390, W//2+130, 580], fill=shirt)
    draw.ellipse([W//2-90,  360, W//2+90,  480], fill=shirt)

    # Face
    draw_avatar_face(draw, W//2, 220, skin, hair)

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, quality=92)
    return path


if __name__ == "__main__":
    print("Generating demo face avatars …\n")
    for i, (name, fname, skin, hair) in enumerate(STUDENTS):
        shirt = SHIRT_COLORS[i % len(SHIRT_COLORS)]
        path  = make_photo(name, fname, skin, hair, shirt)
        print(f"  ✓  {fname}  ({name})")
    print(f"\n✅  {len(STUDENTS)} avatar images saved to ./{OUTPUT_DIR}/")
    print("    Run  python seed_database.py  next.\n")
