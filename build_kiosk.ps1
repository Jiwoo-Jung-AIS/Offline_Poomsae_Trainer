param(
  [string]$ProjectDir = ".",
  [string]$PyVersion = "3.11",
  [string]$Name = "TaekwondoPoomsaeKiosk",
  [string]$AppPy = "app_gradio_poomsae14.py"
)

Set-Location $ProjectDir

if (!(Test-Path ".\.venv")) {
  py -$PyVersion -m venv .venv
}
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install "mmcv-lite>=2.0.0"
pip install "mmpose>=1.0.0" gradio opencv-python numpy scipy pandas pyinstaller

if (!(Test-Path ".\models\rtmpose\rtmpose-s_8xb64-420e_coco-256x192.py") -or !(Test-Path ".\models\rtmpose\rtmpose-s_coco-256x192.pth")) {
  Write-Error "Missing model files under .\models\rtmpose\. Please add the .py config and .pth checkpoint."
  exit 1
}

pyinstaller --noconfirm --onedir --name $Name $AppPy --collect-all mmpose --collect-all mmengine --collect-all mmcv --collect-submodules torch --collect-submodules torchvision --collect-submodules torchaudio --add-data "models;models"

Write-Host "Build complete: dist\$Name\$Name.exe"
