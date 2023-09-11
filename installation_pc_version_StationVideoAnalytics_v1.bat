:: Step 1: Create virtual environment
:: python -m venv env

:: Step 2: Run command in virtual environment
:: cd env/Scripts && activate && cd ../../

:: Step 3: Install YOLO v5 //Replace with your own github instead. 
:: git clone https://github.com/ultralytics/yolov5.git

:: Step 4: Install pc_version_vehicle_counting code
:: git clone https://github.com/zhiyi-li-zdaly/pc_version_StationVideoAnalitycs

:: Step 5: Install nmap for IP address search for cameras. 
cd pc_version_StationVideoAnalitycs
nmap-7.94-setup.exe

:: Step 5: Install CUDA
:: Use already download exe file with Windows 11, X86-64, and local file.
:: Need inter-active operations
:: cuda_12.2.2_537.13_windows.exe

:: Step 5: Install PyTorch environment.
:: Goto website to download: Start Locally | PyTorch
:: Select version: Stable 2.0.1, Windows, Pip, Python, CPU
:: pip3 install torch torchvision torchaudio

:: Step 6: Install extra requirements:
:: cd yolov5
:: pip install -r requirements.txt

:: copy pc_version_StationVideoAnalitycs code main_v6.py into yolov5 window.
:: cd ../
:: copy pc_version_StationVideoAnalitycs\main_v6.py yolov5
:: cd yolov5

:: Step 7: Test yolo v5 program
:: python detect.py --source 0
:: PAUSE
