:: Step 0: Launch Anaconda environment from bat
call C:\Users\zhiyi\anaconda3\condabin\activate.bat

:: Step 1. Find IP of cameras within same local network of PC and record in config/cameras.yml file.
python findIPandCaptureVideoofCamera_v1.py

:: Step 2. 
PAUSE