import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import io
from fpdf import FPDF
import tempfile
import os
import time

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Mosaico Studio Pro v6.0", layout="wide")

COLUNAS_PLACAS, LINHAS_PLACAS = 5, 8
TAMANHO_PLACA = 16
GRID_W, GRID_H = 80, 128

if 'matriz' not in st.session_state:
    st.session_state.matriz = None
if 'paleta' not in st.session_state:
    st.session_state.paleta = None

st.title("🧱 Mosaico Studio v6.0 - Alta Definição de Linhas")

# --- SIDEBAR ---
st.sidebar.header("1. Configurações de Imagem")
uploaded_file = st.sidebar.file_uploader("Imagem do Luffy", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.sidebar.divider()
    st.sidebar.header("2. Motor de Nitidez")
    
    num_cores = st.sidebar.slider("Cores", 2, 48, 16)
    
    # NOVO: Filtro de Mediana para limpar pixels isolados
    limpeza = st.sidebar.slider("Limpeza de Ruído (Unificar Cores)", 0, 5, 2, 
                                help="Remove pixels 'perdidos' nas bordas. Valores altos deixam o desenho mais limpo.")
    
    # NOVO: Realce de Contorno para letras e detalhes
    contorno = st.sidebar.slider("Definição de Contorno (Linhas Pretas)", 0.0, 2.0, 1.0, 0.1,
                                 help="Aumenta as linhas pretas do cabelo e letras.")
    
    contraste_extra = st.sidebar.slider("Contraste da Paleta", 1.0, 3.0, 1.3)

    if st.sidebar.button("⚙️ Gerar Mosaico de Alta Definição"):
        img = Image.open(uploaded_file).convert("RGB")
        
        # --- PASSO 1: LIMPEZA DE RUÍDO ---
        if limpeza > 0:
            # O Filtro de Mediana é perfeito para o que você quer: ele unifica cores próximas
            for _ in range(limpeza):
                img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # --- PASSO 2: REALCE DE BORDAS ---
        if contorno > 0:
            # Encontra as bordas e as intensifica
            bordas = img.filter(ImageFilter.FIND_EDGES).convert("L")
            bordas = ImageEnhance.Contrast(bordas).enhance(contorno * 5)
            # "Queima" as bordas pretas de volta na imagem original
            img_bordas = Image.composite(Image.new("RGB", img.size, (0,0,0)), img, bordas)
            img = ImageBlend = Image.blend(img, img_bordas, alpha=contorno/2)

        # --- PASSO 3: AJUSTE DE CONTRASTE ---
        img = ImageEnhance.Contrast(img).enhance(contraste_extra)
        img = ImageEnhance.Sharpness(img).enhance(2.0)

        # --- PASSO 4: REDUÇÃO (SUPER-SAMPLING) ---
        # Reduz primeiro para 2x o tamanho com filtro de alta qualidade
        temp_res = img.resize((GRID_W * 2, GRID_H * 2), Image.LANCZOS)
        # Redução final para o tamanho LEGO usando NEAREST para não criar cores novas
        img_final = temp_res.resize((GRID_W, GRID_H), Image.NEAREST)

        # --- PASSO 5: QUANTIZAÇÃO K-MEANS ---
        img_quant = img_final.quantize(colors=num_cores, method=2, kmeans=num_cores, dither=Image.NONE)
        
        st.session_state.paleta = [img_quant.getpalette()[i*3:i*3+3] for i in range(num_cores)]
        st.session_state.matriz = np.array(img_quant)
        st.rerun()

# --- INTERFACE PRINCIPAL (EDITOR E VISUALIZAÇÃO) ---
if st.session_state.matriz is not None:
    matriz = st.session_state.matriz
    paleta = st.session_state.paleta
    
    col_v, col_e = st.columns([2, 1])
    
    with col_v:
        st.subheader("🖼️ Resultado com Filtros de Nitidez")
        p_flat = []
        for c in paleta: p_flat.extend(c)
        while len(p_flat) < 768: p_flat.append(0)
        
        img_p = Image.new("P", (GRID_W, GRID_H))
        img_p.putpalette(p_flat)
        img_p.putdata(matriz.flatten())
        
        # Zoom para visualização
        escala = 8
        v_rgb = img_p.convert("RGB").resize((GRID_W * escala, GRID_H * escala), Image.NEAREST)
        draw = ImageDraw.Draw(v_rgb)
        
        # Grade de placas vermelha (sutil)
        for x in range(0, GRID_W * escala, 16 * escala):
            draw.line([(x, 0), (x, GRID_H * escala)], fill=(255, 0, 0, 80), width=1)
        for y in range(0, GRID_H * escala, 16 * escala):
            draw.line([(0, y), (GRID_H * escala, y)], fill=(255, 0, 0, 80), width=1)
            
        st.image(v_rgb, use_container_width=True)

    with col_e:
        st.subheader("🖌️ Ajuste de Pixel")
        px = st.number_input("X (0-79)", 0, 79, 0)
        py = st.number_input("Y (0-127)", 0, 127, 0)
        c_sel = st.selectbox("Cor", range(len(paleta)), format_func=lambda x: f"Cor {x+1} {paleta[x]}")
        if st.button("Aplicar"):
            st.session_state.matriz[py, px] = c_sel
            st.rerun()

    # --- PDF BRIKO-STYLE ---
    st.divider()
    if st.button("🚀 Gerar PDF Linha-por-Linha"):
        try:
            with st.spinner("Desenhando manual..."):
                pdf = FPDF(orientation="P", unit="mm", format="A4")
                
                # Capa
                pdf.add_page()
                pdf.set_font("Arial", 'B', 24)
                pdf.cell(0, 20, "Luffy Mosaic Build Guide", ln=True, align="C")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    v_rgb.save(tf.name)
                    pdf.image(tf.name, x=35, y=40, w=140)
                    capa_path = tf.name

                # Loop das 128 Linhas
                cel = 10
                img_step = Image.new("RGB", (GRID_W * cel, GRID_H * cel), (250, 250, 250))
                draw_s = ImageDraw.Draw(img_step)

                for step in range(GRID_H):
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, f"Step {step+1} / 128", ln=True)
                    
                    # Cores da linha
                    linha = matriz[step, :]
                    unique_c = np.unique(linha)
                    
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 8, "Cores para esta linha:", ln=True)
                    
                    pdf.set_font("Arial", '', 9)
                    curr_x = 10
                    for c_idx in unique_c:
                        rgb = paleta[c_idx]
                        qtd = np.count_nonzero(linha == c_idx)
                        pdf.set_fill_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                        pdf.rect(curr_x, pdf.get_y(), 4, 4, 'F')
                        pdf.set_xy(curr_x + 5, pdf.get_y())
                        pdf.cell(35, 4, f"Cor {c_idx+1} ({qtd})")
                        curr_x += 40
                    
                    # Atualiza desenho do passo
                    for col in range(GRID_W):
                        idx = matriz[step, col]
                        draw_s.rectangle([col*cel, step*cel, (col+1)*cel, (step+1)*cel], fill=tuple(paleta[idx]), outline=(200,200,200))

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        img_step.save(tf.name)
                        pdf.image(tf.name, x=33, y=50, w=144)
                        step_path = tf.name
                    os.remove(step_path)

                os.remove(capa_path)
                pdf_out = pdf.output(dest='S').encode('latin-1')
                st.download_button("📥 Baixar PDF Briko", pdf_out, "manual_luffy.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Erro: {e}")