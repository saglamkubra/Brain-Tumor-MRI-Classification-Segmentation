@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "NEURO_PY=.venv\Scripts\python.exe"
if not exist "%NEURO_PY%" (
    python -m venv .venv
    if errorlevel 1 goto error
)
"%NEURO_PY%" -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo CPU destekli PyTorch kuruluyor...
        "%NEURO_PY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo CUDA destekli PyTorch kuruluyor...
        "%NEURO_PY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    )
    if errorlevel 1 goto error
)
"%NEURO_PY%" -c "import fastapi, uvicorn, timm, pytorch_grad_cam, segment_anything, cv2, numpy, PIL, reportlab" >nul 2>&1
if errorlevel 1 (
    "%NEURO_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto error
)
echo NEURO AI LAB baslatiliyor...
"%NEURO_PY%" -m backend.launch %*
if errorlevel 1 goto error
exit /b 0

:error
echo Uygulama baslatilamadi. Yukaridaki hata mesajini kontrol edin.
pause
exit /b 1
