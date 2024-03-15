import cv2
import requests
import numpy as np

# from: https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
def resize(image, width=None, height=None, inter=cv2.INTER_AREA) -> np.ndarray:
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def get_template(name: str, width=500) -> np.ndarray:
    # load from disk
    image = cv2.imread(name)

    # resize the image
    resized = resize(image, width=width)

    # convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray

def get_image(url: str, width=500) -> np.ndarray:
    # download the image, convert it to a NumPy array, and then read
    response = requests.get(url)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # resize the image
    resized = resize(image, width=width)

    # convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray


def measure_blur(image: np.ndarray, size=60) -> tuple[float, float]:
    # phase 1: Blur measure using Laplacian filter
    laplacian_blur = cv2.Laplacian(image, cv2.CV_64F).var()

    # phase 2: Blur measure using FFT
    (h, w) = image.shape
    (cX, cY) = (w // 2, h // 2)

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    fft_measure = np.mean(magnitude)

    return (laplacian_blur, fft_measure)
    
def measure_similarity(image: np.ndarray, template: np.ndarray) -> float:
    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m.distance < 64 for m in matches]

    return sum(good_matches) / len(matches)
