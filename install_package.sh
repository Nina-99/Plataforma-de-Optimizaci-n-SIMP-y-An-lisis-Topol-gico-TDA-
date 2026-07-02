#!/bin/bash

# Verificar si Python está instalado
if ! command -v python3 &>/dev/null; then
    echo "Python no está instalado. Instalando..."
    sudo apt-get update
    sudo apt-get install -y python3
fi

# Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar los paquetes desde requirements.txt
pip install -r requirements.txt

echo "======================================"
echo "Instalación completada exitosamente!."
echo "======================================"

streamlit run src/tda/app/app_master.py
