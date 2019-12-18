# Simple OCR processing application(script)

This is a basic OCR system that is created to help picking up text from video stream (or still images), and output JSON data on where and what was detected, it produces rect coordinates with text recognized and confidences.


## Installation

In order to work this script requires that your machine has Python 3, OpenCV and Tesseract installed. 

On Linux (and Mac?) these can be obtained from your OS package manager. 

On Windows you have to manually download and install these libraries.

Before doing any setup grab OpenCV trained data

The link to the actual data may change, so open up this file in text editor and find the URL for 'EAST' model, download and save it.

https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py

MobileNet SSD v2 weights and config data
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

Now let's move on to the platform specific packages.

### Ubuntu packages 

Most systems should have everything required in their package management systems, on Ubuntu Linux for example this is all what you need

    apt install libtesseract-dev tesseract-ocr-eng libopencv-dev

But also install python3 if not present yet.

    apt install python3 pip3

### Windows prebuilt binaries

Download and extract those in relevant location, later you will need to update system's %PATH% variable to point to OpenCV and Tesseract location 
 
OpenCV
https://github.com/opencv/opencv/releases

Tesseract 4.x
https://github.com/UB-Mannheim/tesseract/wiki


You will need tesseract trained data for the required languages (copy this to tesseract/tessdata folder)

https://github.com/tesseract-ocr/tesseract/wiki/Data-Files

https://github.com/tesseract-ocr/tessdata

Additionally you will need prebuilt python package for tesseract (.whl file)

https://github.com/simonflueckiger/tesserocr-windows_build/releases



### Make virtual environment

Now when you have those libraries let's start with creating virtual environment

Python 2.x (Linux) / Windows

    python -m venv venv## 

Python 3 (Linux)

    python3 -m venv venv##

Where ## is your Python version, such as 37 for Python 3.7

Now we can activate it and install Python packages

Windows

    ./venv##/Scripts/activate 

Linux 

    source ./venv##/bin/activate

And also update the newly created environment's pip

    python -m pip install --upgrade pip


**On Linux this script will set temporary alias so if previously you were using ```python3``` explicitly now it will be just ```python```. It also might add deactivate() shell function to clear this off so you can continue using your system Python without restarting the terminal**

**Remember that you will need to activate your environment every time you open new terminal, otherwise it'll use your system python and it will mess it up if you do pip install**


## Install dependencies using pip

Now we are ready to install python packages, let's do it

    pip install -r requirements.txt

(Windows only) 
We have to install prebuilt .whl for tesserocr you've downloaded previously

    pip install tesserocr-{downloaded.version}.whl


## Set Environment variables

On some platforms we might have to update system's ```PATH``` environment variable, on Linux this may already handled when you did installation using package manager, on Windows however you have to do it manually

(PowerShell) set env vars (change to your paths)  

> $env:PATH+=";E:\tesseract\"  
> $env:PATH+=";E:\opencv\build\x64\vc15\bin"  
> $env:TESSDATA_PREFIX+="E:\tesseract\tessdata"  


And for reference here is what is expected

- **$PATH**: OpenCV .so/.dll, tesseract as well
- **$TESSDATA_PREFIX**: Should point to tesseract install dir tessdata/ subfolder, ensure it has language data


## Run the app

Everything should be ready to run the script. 

Remember that you must activate the environment if not yet.

Finally let's run it.

    python combined.py --input <path/to/file> [--width #] [--height #] [--frame #]

Use width and height to downscale images and improve performance
You don't need to set width and height, however you should keep aspect ratio if you do, currently it must be multiple of 32, we'll change this later

You can also pass --frame \<number\> to pick specific frame from video file

Use --debug to see visualization


## Minimalistic Web Server (optional)

For testing purposes there is basic web server available, it accepts file with width and height using POST request.

To run this server grab ```flask``` using pip.

    pip install flask

Follow flask instructions and set environment and flask script app(server.py without extension, in this case) to run, optionally using prod or development version

    export FLASK_APP=server
    export FLASK_ENV=development

Run the server.

    flask run

Open http://127.0.0.1:5000 in a browser.

_For more information refer to flask documentation._