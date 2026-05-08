import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import io
from fpdf import FPDF
import tempfile
import os
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Gerador de Mosaico Pro", layout="wide")

# Constantes do Projeto (Luffy: 5x8 placas de 16x16 = 80x128 studs)
COLUNAS_PLACAS = 5
LINHAS_PLACAS = 8
TAMANHO_PLACA = 16
GRID_W = COLUNAS_PLACAS * TAMANHO_PLACA  # 80
GRID_H = LINHAS_PLACAS * TAMANHO_PLACA   # 128

# --- INICIALIZAÇÃO DO ESTADO (Para não perder dados ao interagir) ---
if 'matriz' not in st.session_state:
    st.session_state.matriz = None
if 'paleta' not in st.session_state:
    st.session_state.paleta = None

def reiniciar_projeto():
    st.session_state.matriz = None
    st.session_state.paleta = None

st.title("🧱 Mosaico Studio v5.0 - Edição & Manual Profissional")

# --- BARRA LATERAL: CONTROLES DE PROCESSAMENTO ---
st.sidebar.header("1. Entrada de Imagem")
uploaded_file = st.sidebar.file_uploader("Selecione a imagem original", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.sidebar.divider()
    st.sidebar.header("2. Ajustes de Qualidade")
    num_cores = st.sidebar.slider("Quantidade de Cores (Paleta)", 2, 64, 16)
    
    st.sidebar.subheader("Preservação de Detalhes")
    preservar = st.sidebar.checkbox("🛡️ Proteger Traços (Dedos/Letras)", value=True)
    contraste = st.sidebar.slider("Força do Traço Escuro", 1.0, 3.0, 1.5)
    
    if st.sidebar.button("⚙️ Gerar Mosaico Base"):
        img = Image.open(uploaded_file).convert("RGB")
        
        # --- MOTOR DE ALTA FIDELIDADE (SUPER-SAMPLING) ---
        if preservar:
            # Passo 1: Redução suave para o dobro do tamanho
            temp_img = img.resize((GRID_W * 2, GRID_H * 2), Image.LANCZOS)
            # Passo 2: Enfatiza os traços pretos (letras/contornos)
            temp_img = ImageEnhance.Contrast(temp_img).enhance(contraste)
            # Passo 3: Redução final para 80x128 (Nearest para cor limpa)
            img_res = temp_img.resize((GRID_W, GRID_H), Image.NEAREST)
        else:
            img_res = img.resize((GRID_W, GRID_H), Image.LANCZOS)

        # --- QUANTIZAÇÃO K-MEANS (Melhores cores possíveis) ---
        img_quant = img_res.quantize(colors=num_cores, method=2, kmeans=num_cores, dither=Image.NONE)
        
        # Salva no estado da sessão
        st.session_state.paleta = [img_quant.getpalette()[i*3:i*3+3] for i in range(num_cores)]
        st.session_state.matriz = np.array(img_quant)
        st.rerun()

# --- ÁREA PRINCIPAL ---
if st.session_state.matriz is not None:
    # Preparar a imagem para exibição
    paleta = st.session_state.paleta
    matriz = st.session_state.matriz
    
    p_flat = []
    for c in paleta: p_flat.extend(c)
    while len(p_flat) < 768: p_flat.append(0)
    
    img_display = Image.new("P", (GRID_W, GRID_H))
    img_display.putpalette(p_flat)
    img_display.putdata(matriz.flatten())
    
    col_view, col_edit = st.columns([2, 1])

    with col_view:
        st.subheader("🖼️ Visualização do Mosaico")
        escala = 8
        view_rgb = img_display.convert("RGB").resize((GRID_W * escala, GRID_H * escala), Image.NEAREST)
        
        # Grade de placas (16x16)
        draw_v = ImageDraw.Draw(view_rgb)
        for x in range(0, GRID_W * escala, TAMANHO_PLACA * escala):
            draw_v.line([(x, 0), (x, GRID_H * escala)], fill=(255, 0, 0, 120), width=1)
        for y in range(0, GRID_H * escala, TAMANHO_PLACA * escala):
            draw_v.line([(0, y), (GRID_H * escala, y)], fill=(255, 0, 0, 120), width=1)
            
        st.image(view_rgb, use_container_width=True)

    with col_edit:
        st.subheader("🖌️ Editor Manual")
        st.write("Corrija pixels específicos:")
        ex = st.number_input("X (Coluna)", 0, 79, 0)
        ey = st.number_input("Y (Linha)", 0, 127, 0)
        
        cores_opcoes = [f"Cor {i+1} - RGB{paleta[i]}" for i in range(len(paleta))]
        nova_cor = st.selectbox("Selecione a Cor", range(len(cores_opcoes)), format_func=lambda x: cores_opcoes[x])
        
        if st.button("Pintar Pixel"):
            st.session_state.matriz[ey, ex] = nova_cor
            st.success(f"Pixel ({ex},{ey}) atualizado!")
            time.sleep(0.3)
            st.rerun()

    # --- GERAÇÃO DE PDF (ESTILO BRIKO - LINHA POR LINHA) ---
    st.divider()
    st.subheader("📄 Gerar Manual de Instruções")
    
    if st.button("🚀 Gerar PDF Passo-a-Passo"):
        try:
            with st.spinner("Desenhando manual de 130 páginas..."):
                pdf = FPDF(orientation="P", unit="mm", format="A4")
                
                # --- PÁGINA 1: OVERVIEW ---
                pdf.add_page()
                pdf.set_font("Arial", 'B', 24)
                pdf.cell(0, 20, "Project Overview", ln=True, align="C")
                
                # Capa (Imagem do projeto)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    view_rgb.save(tf.name)
                    pdf.image(tf.name, x=35, y=40, w=140)
                    path_capa = tf.name

                pdf.set_xy(10, 250)
                pdf.set_font("Arial", '', 14)
                pdf.cell(0, 10, f"Size: {GRID_W}x{GRID_H} studs | Total: {GRID_W*GRID_H} bricks", ln=True, align="C")

                # --- PÁGINAS DE PASSOS (128 LINHAS) ---
                cel = 10 # Tamanho do stud na imagem do PDF
                img_progresso = Image.new("RGB", (GRID_W * cel, GRID_H * cel), (245, 245, 245))
                draw_p = ImageDraw.Draw(img_progresso)

                for step in range(GRID_H):
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, f"Step {step + 1}", ln=True)
                    
                    # Informações da linha
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(0, 6, f"Place now: {GRID_W}  |  Total so far: {(step+1)*GRID_W}", ln=True)
                    
                    # Legenda de cores desta linha
                    linha_data = matriz[step, :]
                    cores_na_linha = np.unique(linha_data)
                    
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 8, "Colors needed for this line:", ln=True)
                    
                    pdf.set_font("Arial", '', 9)
                    curr_x = 10
                    for c_idx in cores_na_linha:
                        rgb = paleta[c_idx]
                        qtd = np.count_nonzero(linha_data == c_idx)
                        
                        pdf.set_fill_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                        pdf.rect(curr_x, pdf.get_y(), 4, 4, 'F')
                        pdf.set_xy(curr_x + 5, pdf.get_y())
                        pdf.cell(35, 4, f"Cor {c_idx+1} ({qtd})")
                        curr_x += 40
                        if curr_x > 180:
                            curr_x = 10
                            pdf.ln(5)
                    
                    # Atualiza a imagem de progresso linha por linha
                    for col in range(GRID_W):
                        idx = matriz[step, col]
                        rgb = tuple(paleta[idx])
                        draw_p.rectangle([col*cel, step*cel, (col+1)*cel, (step+1)*cel], fill=rgb, outline=(180,180,180))

                    # Insere a imagem do progresso no PDF
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        img_progresso.save(tf.name)
                        pdf.image(tf.name, x=20, y=50, w=170)
                        path_step = tf.name
                    os.remove(path_step)

                    # Numeração de página
                    pdf.set_xy(10, 285)
                    pdf.set_font("Arial", 'I', 8)
                    pdf.cell(0, 5, f"p. {step+1} of 128", align="R")

                # --- PÁGINA FINAL: BOM ---
                pdf.add_page()
                pdf.set_font("Arial", 'B', 20)
                pdf.cell(0, 15, "Bill of Materials", ln=True)
                
                counts = np.unique(matriz, return_counts=True)
                for c_idx, count in zip(counts[0], counts[1]):
                    rgb = paleta[c_idx]
                    pdf.set_fill_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                    pdf.rect(10, pdf.get_y()+2, 6, 6, 'F')
                    pdf.set_xy(20, pdf.get_y())
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(100, 10, f"Cor {c_idx+1} (RGB: {rgb[0]},{rgb[1]},{rgb[2]})")
                    pdf.cell(30, 10, f"Total: {count}", align="R")
                    pdf.ln(8)

                # Limpeza e Download
                os.remove(path_capa)
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.success("✅ Manual Gerado!")
                st.download_button("📥 Baixar PDF Briko-Style", pdf_output, "manual_montagem.pdf", "application/pdf")

        except Exception as e:
            st.error(f"Erro na geração: {e}")

else:
    st.info("Aguardando imagem...")