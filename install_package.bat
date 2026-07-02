@echo off

REM Verificar si Python está instalado
where python3 >nul 2>&1
if %errorlevel% neq 0 (
    echo Python no está instalado. Instalando...
    choco install python -y
)

REM Crear y activar el entorno virtual
python3 -m venv venv
call venv\Scripts\activate.bat

REM Instalar los paquetes desde requirements.txt
pip install -r requirements.txt

echo "======================================"
echo "Instalación completada exitosamente!."
echo "======================================"

streamlit run src/tda/app/app_master.py
