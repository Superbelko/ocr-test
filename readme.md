
## Make virtual environment
- python -m venv venv## (where ## is your python version, such as 37 for python 3.7)
- venv##/Scripts/activate (Windows)
- source ./venv##/bin/activate (Linux)
- python -m pip install --upgrade pip

**You need to activate your environment every time you open new terminal, otherwise it'll use your system python and it will mess it up if you do pip install**

## Install dependencies using pip

- pip install -r requirements.txt

Windows extra steps:

- Get OpenCV DLL's, extract somewhere (will need to add to %PATH% later)
https://github.com/opencv/opencv/releases
- Get one of the .whl from https://github.com/simonflueckiger/tesserocr-windows_build/releases
- pip install tesserocr-{wutever-ver}.whl
- Get tesseract binaries for v4.x from https://github.com/UB-Mannheim/tesseract/wiki

Here is tesserocr info about Windows https://github.com/sirfz/tesserocr

## Dependencies

- Get libs - tesseract, opencv
- Get tesseract trained data for required languages and copy to your tesseract/tessdata folder https://github.com/tesseract-ocr/tesseract/wiki/Data-Files
https://github.com/tesseract-ocr/tessdata

- Get OpenCV trained model for EAST
check this file for where to download it
https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py

## Set Environment variables

> PowerShell set env vars (change to your paths)
> $env:PATH+=";E:\tesseract\"
> $env:PATH+=";E:\opencv\build\x64\vc15\bin"
> $env:TESSDATA_PREFIX+="E:\tesseract\tessdata"

- **$PATH**: OpenCV must be present, tesseract as well
- **$TESSDATA_PREFIX**: Should point to tesseract install dir tessdata/ subfolder, ensure it has language data


## Run the app

    python combined.py --input <path/to/file> [--width #] [--height #] [--frame #]

Use width and height to downscale images and improve performance
You don't need to set width and height, however you should keep aspect ratio if you do, currently it must be multiple of 32, we'll change this later

You can also pass --frame \<number\> to pick specific frame from video file

Use --debug to see visualization