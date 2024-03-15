import cv2
import numpy as np

# from: https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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


image = cv2.imread("dataset/1101062028003_003_p1.jpg")
resized = resize(image, width=500)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# using laplacian filter
laplacian_blur = cv2.Laplacian(gray, cv2.CV_64F).var()
print("Laplacian =", laplacian_blur)

# using FFT
size = 60
(h, w) = gray.shape
(cX, cY) = (w // 2, h // 2)

# compute the FFT to find the frequency transform, then shift
# the zero frequency component (i.e., DC component located at
# the top-left corner) to the center where it will be more
# easy to analyze
fft = np.fft.fft2(gray)
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
mean = np.mean(magnitude)
print("FFT =", mean)
