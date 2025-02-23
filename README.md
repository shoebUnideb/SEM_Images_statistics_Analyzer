## Introduction
This github repository is to help in statistical analysis of the SEM images. It includes 
  1. Total number of particles
  2. Surface Coverage (%)
  3. Total grain area (µm²)
  4. Mean grain area (µm²)
  5. Particle density (particles/µm²)
  6. Standard deviation of area (µm²):
  7. Histogram plot of frequency of number of particles vs area in µm²
The repository contains the following files: input images, output measurements, main file and pipfile.

## To use :
1. Clone the repository
2. install pipenv using `pipenv install`
3. Activate pipenv `pipenv shell`
4. Install the required packages by running `pip install -r requirements.txt`
5. Move the images you need to analyze to the input_images directory.
6. Make changes to the code in the line 11 and 12 according to the calibration of your image type, for instance its set that 300 pixel = 1000 nanometer, which is also equal to the sample image calibration
7. Run the file using `python main.py `

## Libraries used
1. Python-Opencv
2. Scikit-learn
3. Scikit-image
4. Matplotlib
5. Numpy
6. Scipy
