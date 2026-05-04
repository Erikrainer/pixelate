"""
=============================================================================
  LEGO Mosaic Building Instruction Generator (VERSÃO DEFINITIVA E LIMPA)
=============================================================================
  Converts any image into a printable LEGO-style step-by-step PDF guide.
=============================================================================
"""

import sys
import os
import argparse
import textwrap
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFilter

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ──────────────────────────────────────────────────────────────────────────────
#  LEGO COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
LEGO_PALETTE = [
    ("White",           (255, 255, 255)),
    ("Light Bluish Gray",(171, 173, 172)),
    ("Dark Bluish Gray", ( 89,  93,  96)),
    ("Black",           (  0,   0,   0)),
    ("Bright Red",      (196,  40,  28)),
    ("Dark Red",        (123,  46,  47)),
    ("Bright Orange",   (254, 138,  24)),
    ("Bright Yellow",   (255, 205,   0)),
    ("Bright Green",    ( 75, 151,  74)),
    ("Dark Green",      ( 35, 120,  65)),
    ("Bright Blue",     (  0, 114, 188)),
    ("Dark Blue",       ( 20,  48, 100)),
    ("Medium Blue",     ( 97, 175, 217)),
    ("Sky Blue",        (146, 207, 224)),
    ("Sand Blue",       (105, 138, 167)),
    ("Bright Purple",   (107,  50, 124)),
    ("Medium Lavender", (160, 110, 185)),
    ("Bright Pink",     (255, 102, 147)),
    ("Light Pink",      (252, 204, 210)),
    ("Tan",             (222, 198, 156)),
    ("Dark Tan",        (150, 130,  96)),
    ("Brown",           (105,  64,  40)),
    ("Reddish Brown",   (128,  70,  50)),
    ("Olive Green",     ( 95, 116,  35)),
    ("Lime Green",      (167, 202,  26)),
    ("Light Nougat",    (255, 201, 149)),
    ("Nougat",          (204, 142, 105)),
    ("Reddish Orange",  (220,  60,  30)),
]

GRID_W  = 80
GRID_H  = 128
PLATE_W = 16
PLATE_H = 16
COLS    = 5
ROWS    = 8

# ──────────────────────────────────────────────────────────────────────────────
#  MATEMÁTICA DE COR
# ──────────────────────────────────────────────────────────────────────────────
def _rgb_to_linear(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_xyz(r: float, g: float, b: float):
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return m @ np.array([r, g, b])

def _xyz_to_lab(xyz):
    ref = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / ref
    def f(v):
        return v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116
    fx, fy, fz = f(t[0]), f(t[1]), f(t[2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])

def rgb_to_lab(rgb_tuple):
    return _xyz_to_lab(_linear_to_xyz(*[_rgb_to_linear(c) for c in rgb_tuple]))

_PALETTE_LAB = np.array([rgb_to_lab(c[1]) for c in LEGO_PALETTE], dtype=np.float32)

def _vectorized_rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ M.T
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    t = xyz / ref
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + (16.0 / 116.0))
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)

def quantize_to_lego(img_arr: np.ndarray) -> np.ndarray:
    H, W = img_arr.shape[:2]
    flat = img_arr.reshape(-1, 3)
    pix_lab = _vectorized_rgb_to_lab(flat)
    diff = pix_lab[:, np.newaxis, :] - _PALETTE_LAB[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    idx_flat = np.argmin(dist_sq, axis=1).astype(np.uint8)
    return idx_flat.reshape(H, W)

# ──────────────────────────────────────────────────────────────────────────────
#  PROCESSAMENTO DA IMAGEM CORRIGIDO
# ──────────────────────────────────────────────────────────────────────────────
def process_image(path: str):
    print("      [1/4] Ajustando fidelidade de cor e nitidez fina...")
    img_orig = Image.open(path).convert("RGB")
    
    # 1. Proporção Total (Mantendo o preenchimento 80x128)
    img_res = img_orig.resize((GRID_W, GRID_H), Image.LANCZOS)
    
    # 2. Correção de Cor Profissional (Neutralizando o Vermelho)
    # Primeiro, aumentamos um pouco a saturação para o marrom/tan não virar cinza
    img_res = ImageEnhance.Color(img_res).enhance(1.4) 
    
    # 3. Nitidez de Alta Frequência (Para as linhas do cabelo)
    # O filtro 'SHARPEN' ajuda a separar as mechas antes do processo de blocos
    img_res = img_res.filter(ImageFilter.SHARPEN)
    
    # 4. Ajuste de Contraste e Brilho (Para o fundo ficar limpo)
    # Contraste em 1.2 ajuda a separar o cabelo do fundo sem queimar as cores
    img_res = ImageEnhance.Contrast(img_res).enhance(1.2)
    img_res = ImageEnhance.Brightness(img_res).enhance(1.0) # Mantemos em 1.0 para não estourar

    # 5. Quantização com Paleta LEGO e Dithering
    # Criamos a paleta técnica para o algoritmo escolher as melhores peças
    lego_rgb_flat = []
    for c in LEGO_PALETTE:
        lego_rgb_flat.extend(c[1])
    lego_rgb_flat.extend([0] * (768 - len(lego_rgb_flat)))
    
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(lego_rgb_flat)
    
    # O Dithering é essencial aqui para evitar manchas sólidas de cores erradas
    img_lego = img_res.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG)
    
    index_map = np.array(img_lego).reshape(GRID_H, GRID_W)
    
    return index_map, np.zeros((GRID_H, GRID_W), dtype=bool)
# ──────────────────────────────────────────────────────────────────────────────
#  UTILITÁRIOS PDF E RENDER
# ──────────────────────────────────────────────────────────────────────────────
def split_into_plates(index_map: np.ndarray):
    plates = []
    for row in range(ROWS):
        for col in range(COLS):
            y0, y1 = row * PLATE_H, (row + 1) * PLATE_H
            x0, x1 = col * PLATE_W, (col + 1) * PLATE_W
            plates.append(index_map[y0:y1, x0:x1].copy())
    return plates

def render_full_mosaic(index_map: np.ndarray, scale: int = 1) -> Image.Image:
    palette_arr = np.array([c[1] for c in LEGO_PALETTE], dtype=np.uint8)
    rgb = palette_arr[index_map]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return Image.fromarray(rgb, "RGB")

def render_plate(plate: np.ndarray, cell_px: int = 30) -> Image.Image:
    margin = 28
    grid_size = PLATE_W * cell_px
    img = Image.new("RGB", (margin + grid_size + 4, margin + grid_size + 4), (225, 228, 232))
    draw = ImageDraw.Draw(img)
    try: font_sm = ImageFont.truetype("arial.ttf", 9)
    except: font_sm = ImageFont.load_default()
    
    for row in range(PLATE_H):
        for col in range(PLATE_W):
            colour = LEGO_PALETTE[plate[row, col]][1]
            x0, y0 = margin + col * cell_px, margin + row * cell_px
            draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=colour, outline=(80,80,80))
            sr, sg, sb = colour
            stud_c = (min(255, sr+60), min(255, sg+60), min(255, sb+60)) if (0.299*sr + 0.587*sg + 0.114*sb) < 128 else (max(0, sr-50), max(0, sg-50), max(0, sb-50))
            cx, cy = x0 + cell_px // 2, y0 + cell_px // 2
            r = max(2, cell_px // 5)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=stud_c)

    col_let = "ABCDEFGHIJKLMNOP"
    for col in range(PLATE_W): draw.text((margin + col*cell_px + cell_px//2, 6), col_let[col], fill=(80,80,80), font=font_sm, anchor="mt")
    for row in range(PLATE_H): draw.text((8, margin + row*cell_px + cell_px//2), str(row+1), fill=(80,80,80), font=font_sm, anchor="lm")
    return img

def pil_to_reportlab(pil_img: Image.Image) -> ImageReader:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)

# Funções simplificadas de PDF
def draw_cover(c, orig_path, mosaic_img, project_name):
    c.setFillColorRGB(0.95, 0.95, 0.95); c.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    c.setFillColorRGB(0.77, 0.16, 0.11); c.rect(0, A4[1] - 45*mm, A4[0], 45*mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(A4[0]/2, A4[1] - 22*mm, "LEGO MOSAIC - CORRIGIDO")
    
    try:
        orig = ImageReader(orig_path)
        c.drawImage(orig, 20*mm, A4[1]*0.3, width=200, height=280, preserveAspectRatio=True)
    except: pass
    c.drawImage(pil_to_reportlab(mosaic_img.resize((320, 512), Image.NEAREST)), A4[0]/2 + 20*mm, A4[1]*0.3, width=200, height=280, preserveAspectRatio=True)
    c.showPage()

def generate_lego_guide(image_path: str, output_path: str = "guia_lego_novo.pdf"):
    if not os.path.isfile(image_path): sys.exit(f"[ERRO] Arquivo não encontrado: {image_path}")
    project_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print("=" * 60)
    print("  Gerador LEGO (Motor Corrigido e Limpo)")
    print("=" * 60)
    
    index_map, _ = process_image(image_path)
    plates = split_into_plates(index_map)
    mosaic_preview = render_full_mosaic(index_map)
    
    c = canvas.Canvas(output_path, pagesize=A4)
    draw_cover(c, image_path, mosaic_preview, project_name)
    
    # Gera apenas as 40 placas para simplificar o PDF gerado no teste
    for i, p in enumerate(plates):
        c.drawImage(pil_to_reportlab(render_plate(p)), 20*mm, 80*mm, width=450, height=450, preserveAspectRatio=True)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, 270*mm, f"Placa {i+1} de 40")
        c.showPage()
        
    c.save()
    print(f"\n  ✓  PDF salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Caminho da imagem")
    parser.add_argument("-o", "--output", default="guia_lego_novo.pdf")
    args = parser.parse_args()
    generate_lego_guide(args.image, args.output)