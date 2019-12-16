import argparse
import json

import cv2 as cv


def laplacian_variance(image):
    return cv.Laplacian(image, cv.CV_64F).var()


def main():
    parser = argparse.ArgumentParser(description='Blur detection utility, calculates pixel variance for an image. (less variance = more blur)')
    parser.add_argument('--input', help='Path to input image file.', required=True)
    args = parser.parse_args()

    image = cv.imread(args.input)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    variance = laplacian_variance(gray)

    print('Variance: {:.2f}'.format(variance))

if __name__ == "__main__":
    main()
