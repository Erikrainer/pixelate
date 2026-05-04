# Pixelate – Pixel Art to Construction Guide

## 📌 Project Overview

This project is a Python-based image processing pipeline that converts any input image into a LEGO-style building instruction guide. The system transforms images into a structured pixel grid, maps colors to a limited LEGO-like palette, and generates a printable step-by-step PDF guide for reconstruction using 1x1 LEGO pieces.

The final output is designed to be practical for real-world assembly using standardized LEGO baseplates.

---

## 🎯 Core Features

- Accepts any input image
- Preserves full image content (no cropping)
- Resizes with aspect ratio preservation and padding
- Converts image into a **80 × 128 pixel grid**
- Applies LEGO-style color quantization (LAB/HSV-based matching)
- Splits final model into **40 baseplates (5 × 8 layout)**
- Generates a structured PDF instruction manual
- Includes color legend and full assembly map

---

## 🧱 Grid System

The build is based on standardized LEGO baseplates:

- Total plates: **40**
- Layout: **5 (width) × 8 (height)**
- Each plate: **16 × 16 studs**
- Final resolution:
  - Width: 5 × 16 = **80 pixels**
  - Height: 8 × 16 = **128 pixels**

Each pixel corresponds to a single 1x1 LEGO piece.

---

## 🖼️ Image Processing Pipeline

### 1. Resize (No Cropping)
- The input image is resized to fit within **80 × 128**
- Aspect ratio is preserved
- No distortion is allowed

### 2. Padding (Letterboxing)
- Remaining space is filled using a neutral background color (configurable)
- Ensures final output is exactly **80 × 128**

### 3. Pixelation
- Image is converted into a pixel grid
- Each pixel represents a LEGO 1x1 brick

---

## 🎨 Color Quantization

- Uses a predefined LEGO-like palette (15–25 colors)
- Conversion is done using perceptual color matching (LAB preferred, HSV acceptable)
- Each pixel is mapped to the nearest available LEGO color

### Color Legend Includes:
- Color name
- HEX / RGB value
- Optional LEGO color ID
- Visual swatch in PDF

---

## 🧩 Plate Division Logic

- Final grid is split into **40 segments**
- Each segment = **16 × 16 pixels**
- Plates are indexed by:
  - Row (0–7)
  - Column (0–4)

Each plate is rendered separately in the PDF for assembly guidance.

---

## 📄 PDF Output Structure

### 1. Cover Page
- Original image
- Pixelated LEGO version
- Project title

### 2. Color Legend
- All used colors
- Swatches and labels

### 3. Plate Instructions (Main Section)
For each of the 40 plates:
- Plate coordinates (Row, Column)
- 16 × 16 grid layout
- Color-coded pixel map

### 4. Full Assembly Map
- Overview showing how all plates connect
- Global layout reference (5 × 8 grid)

---

## 🛠️ Technical Requirements

### Dependencies
- Python 3.9+
- OpenCV or Pillow
- NumPy
- ReportLab or FPDF

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Erikrainer/pixelate
cd pixelate

# Install dependencies
pip install -r requirements.txt