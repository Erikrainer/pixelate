@echo off
title Abrindo Gerador de Mosaico...
echo Aguarde, o sistema esta iniciando os motores...

:: Verifica se o Streamlit ja esta instalado, se nao, instala
pip install -r requirements.txt

:: Comando para rodar o programa
python -m streamlit run lego_mosaic.py

:: Se houver erro, a janela nao fecha sozinha
pause