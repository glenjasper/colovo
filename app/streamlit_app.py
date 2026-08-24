#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

# Garante que o Streamlit Cloud enxergue a raiz do projeto como caminho de busca do Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import streamlit as st
from colovo.utils.image import load_image
from colovo.segmentation.inference import load_segmentation_model
from colovo.calibration.dsm_random_forest import load_dsm_model
from colovo.inference.pipeline import ColovoPipeline

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolk_segmentation.pth")
DSM_MODEL_PATH = os.path.join(BASE_DIR, "models", "dsm_random_forest.pkl")

@st.cache_resource
def load_pipeline():
    seg_model = load_segmentation_model(SEG_MODEL_PATH)
    dsm_model = load_dsm_model(DSM_MODEL_PATH)
    return ColovoPipeline(seg_model, dsm_model)

# Inicialização da Página
st.set_page_config(page_title = "COLOVO", layout = "centered")
st.title("🥚 COLOVO")

pipeline = load_pipeline()

# --- Painel Lateral de Controle ---
with st.sidebar:
    st.header("⚙️ Painel de Operação")
    uploaded = st.file_uploader("Carregar Amostra da Gema", type = ["jpg", "jpeg", "png"])
    st.markdown("---")
    st.info("ℹ️ Certifique-se de que a iluminação da amostra esteja padronizada.")

if uploaded:
    image = load_image(uploaded)
    
    # Try-except encapsula a predição para capturar falhas de amostragem
    try:
        prediction = pipeline.predict(image)
        
        method = prediction["segmentation_method"]
        fallback_reason = prediction["fallback_reason"]

        # Status de Execução do Pipeline
        if method == "classical":
            st.success("✅ **Segmentação Concluída:** Algoritmo Clássico HSV")
        elif method == "unet":
            if fallback_reason == "forced":
                st.info("🤖 **Segmentação Executada:** Modelo de IA U-Net (Modo Forçado)")
            else:
                st.warning(f"⚠️ **Aviso de Fallback:** Modelo de IA U-Net acionado (Motivo: {fallback_reason})")

        mask = prediction["mask"]
        
        # Geração de Contorno de Auditoria sobreposto na Imagem Real
        contour_image = image.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3) # Desenha a borda em verde brilhante
        
        segmented = cv2.bitwise_and(image, image, mask = mask)

        # Seção de Imagens Lado a Lado
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(contour_image, caption = "Auditoria de Contorno", use_container_width = True)
        with col_img2:
            st.image(mask, caption = "Máscara Binária", use_container_width = True)
        with col_img3:
            st.image(segmented, caption = "Região Isolada", use_container_width = True)

        # Bloco de Destaque do DSM Estimado
        st.markdown("---")
        col_dsm1, col_dsm2 = st.columns(2)
        with col_dsm1:
            st.metric(label = "DSM Estimado (Contínuo)", value = f"{prediction['dsm']:.2f}")
        with col_dsm2:
            st.metric(label = "Escala DSM Comercial", value = f"{int(round(prediction['dsm']))}")

        # Painel de Características Colorimétricas
        st.markdown("### 📊 Perfil Cromático da Gema")
        features = prediction["features"]
        
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            st.metric("Hue Mediano", f"{features['hue_median']:.1f}")
        with col_f2:
            st.metric("Saturação Mediana", f"{features['sat_median']:.1f}")
        with col_f3:
            st.metric("Brilho (Value)", f"{features['val_median']:.1f}")
        with col_f4:
            st.metric("Pixels Válidos", f"{features['pixels']:,}")

        with st.expander("Ver todas as propriedades estatísticas"):
            st.write(features)
            
    except ValueError as e:
        # Captura o erro disparado pela falta de pixels válidos e exibe uma mensagem limpa
        st.error(f"❌ **Erro na Análise:** {str(e)}")
        st.warning("⚠️ **Recomendação:** Certifique-se de carregar uma imagem focada e centralizada contendo a gema do ovo quebrada sob um fundo neutro.")
else:
    st.info("💡 Por favor, carregue uma imagem na barra lateral para iniciar a análise colorimétrica.")
