import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
from fpdf import FPDF
import tempfile
import os

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Mosaico Studio Ultra Control v8.0", layout="wide")

GRID_W, GRID_H = 80, 128

if 'matriz' not in st.session_state: st.session_state.matriz = None
if 'paleta' not in st.session_state: st.session_state.paleta = None

st.title("🧱 Mosaico Studio v8.0 - Laboratório de Pixels")

# --- SIDEBAR: O LABORATÓRIO ---
st.sidebar.header("🔬 Painel de Controle Total")
uploaded_file = st.sidebar.file_uploader("Carregar Imagem", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. AJUSTES DE FOTO BÁSICOS
    with st.sidebar.expander("1. Ajustes de Cor e Luz", expanded=True):
        brilho = st.slider("Brilho", 0.0, 3.0, 1.0)
        contraste = st.slider("Contraste", 0.0, 5.0, 1.5)
        saturacao = st.slider("Saturação", 0.0, 3.0, 1.0)
        nitidez_pre = st.slider("Nitidez Pré-Processo", 1.0, 10.0, 3.0)

    # 2. MOTOR DE REDUÇÃO (DOWNSCALING)
    with st.sidebar.expander("2. Algoritmo de Redução", expanded=True):
        metodo = st.selectbox("Método de Interpolação", 
                             ["NEAREST (Melhor para Letras)", "LANCZOS (Melhor para Rosto)", "BILINEAR", "BICUBIC", "BOX"])
        
        metodos_map = {
            "NEAREST (Melhor para Letras)": Image.NEAREST,
            "LANCZOS (Melhor para Rosto)": Image.LANCZOS,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "BOX": Image.BOX
        }

    # 3. DETECÇÃO E REALCE DE LINHAS (CONTORNOS)
    with st.sidebar.expander("3. Realce de Linhas e Letras", expanded=False):
        edge_boost = st.slider("Forçar Bordas Pretas", 0.0, 5.0, 0.0)
        threshold = st.slider("Threshold (Ponto de Corte)", 0, 255, 128, 
                             help="Define o que vira preto absoluto. Use para limpar o fundo das letras.")
        usar_bw_overlay = st.checkbox("Sobrepor Máscara Binária (Traço Único)")

    # 4. LIMPEZA E FINALIZAÇÃO
    with st.sidebar.expander("4. Cores e Limpeza", expanded=False):
        num_cores = st.slider("Quantidade de Cores", 2, 64, 16)
        filtro_mediana = st.slider("Limpeza de Ruído (Mediana)", 0, 7, 0, 
                                  help="Remove pixels 'perdidos' e unifica áreas de mesma cor.")
        dither = st.checkbox("Dithering (Misturar cores com pontos)")

    if st.sidebar.button("⚡ APLICAR TODOS OS FILTROS"):
        img = Image.open(uploaded_file).convert("RGB")
        
        # ETAPA 1: Ajustes de Imagem
        img = ImageEnhance.Brightness(img).enhance(brilho)
        img = ImageEnhance.Contrast(img).enhance(contraste)
        img = ImageEnhance.Color(img).enhance(saturacao)
        img = ImageEnhance.Sharpness(img).enhance(nitidez_pre)
        
        # ETAPA 2: Realce de Bordas (opcional)
        if edge_boost > 0:
            bordas = img.filter(ImageFilter.FIND_EDGES).convert("L")
            bordas = ImageEnhance.Contrast(bordas).enhance(edge_boost * 2)
            img = Image.composite(Image.new("RGB", img.size, (0,0,0)), img, bordas)

        # ETAPA 3: Redução de Tamanho
        img_res = img.resize((GRID_W, GRID_H), metodos_map[metodo])

        # ETAPA 4: Máscara de Traço Único (para letras)
        if usar_bw_overlay:
            gray = img_res.convert("L")
            bw = gray.point(lambda x: 0 if x < threshold else 255, '1')
            img_res = Image.composite(Image.new("RGB", (GRID_W, GRID_H), (0,0,0)), img_res, bw)

        # ETAPA 5: Limpeza de Ruído
        if filtro_mediana > 0:
            for _ in range(filtro_mediana):
                img_res = img_res.filter(ImageFilter.MedianFilter(size=3))

        # ETAPA 6: Quantização de Cores
        dither_type = Image.FLOYDSTEINBERG if dither else Image.NONE
        img_quant = img_res.quantize(colors=num_cores, method=2, kmeans=num_cores, dither=dither_type)
        
        st.session_state.paleta = [img_quant.getpalette()[i*3:i*3+3] for i in range(num_cores)]
        st.session_state.matriz = np.array(img_quant)
        st.rerun()

# --- ÁREA DE VISUALIZAÇÃO ---
if st.session_state.matriz is not None:
    matriz, paleta = st.session_state.matriz, st.session_state.paleta
    
    col_v, col_e = st.columns([2, 1])
    
    with col_v:
        st.subheader("🖼️ Visualização Microscópica")
        p_flat = []
        for c in paleta: p_flat.extend(c)
        while len(p_flat) < 768: p_flat.append(0)
        
        img_p = Image.new("P", (GRID_W, GRID_H))
        img_p.putpalette(p_flat)
        img_p.putdata(matriz.flatten())
        
        escala = 8
        v_rgb = img_p.convert("RGB").resize((GRID_W * escala, GRID_H * escala), Image.NEAREST)
        
        # Grade para ajudar na edição
        draw = ImageDraw.Draw(v_rgb)
        for x in range(0, GRID_W * escala, escala):
            draw.line([(x, 0), (x, GRID_H * escala)], fill=(200, 200, 200, 40), width=1)
        for y in range(0, GRID_H * escala, escala):
            draw.line([(0, y), (GRID_W * escala, y)], fill=(200, 200, 200, 40), width=1)
        
        # Marcação das Placas 16x16
        for x in range(0, GRID_W * escala, 16 * escala):
            draw.line([(x, 0), (x, GRID_H * escala)], fill=(255, 0, 0, 100), width=2)
        for y in range(0, GRID_H * escala, 16 * escala):
            draw.line([(0, y), (GRID_H * escala, y)], fill=(255, 0, 0, 100), width=2)
            
        st.image(v_rgb, use_container_width=True)

    with col_e:
        st.subheader("🖌️ Ajuste de Precisão")
        st.write("Se o algoritmo falhar, conserte aqui:")
        ex = st.number_input("X (0-79)", 0, 79, 0)
        ey = st.number_input("Y (0-127)", 0, 127, 0)
        nova_cor = st.selectbox("Cor", range(len(paleta)), format_func=lambda x: f"Cor {x+1} {paleta[x]}")
        if st.button("Pintar Pixel"):
            st.session_state.matriz[ey, ex] = nova_cor
            st.rerun()

    # --- PDF ---
    st.divider()
    if st.button("📥 Gerar Manual PDF"):
        # (Código do PDF Linha-por-Linha mantido)
        st.write("Gerando...") # Omitido aqui por brevidade, mas deve ser mantido do código anterior.