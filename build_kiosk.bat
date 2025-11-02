@echo off
setlocal
set NAME=TaekwondoPoomsaeKiosk
set APPPY=app_gradio_poomsae14.py

pyinstaller --noconfirm --onedir --name %NAME% %APPPY% --collect-all mmpose --collect-all mmengine --collect-all mmcv --collect-submodules torch --collect-submodules torchvision --collect-submodules torchaudio --add-data "models;models"

echo Built dist\%NAME%\%NAME%.exe
