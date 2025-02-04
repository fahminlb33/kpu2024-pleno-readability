import json
import argparse
import functools
from hashlib import sha3_256
from multiprocessing import Pool, cpu_count

import cv2
import requests
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity, mean_squared_error

# --------------------------------------------------------------------
#  HELPERS
# --------------------------------------------------------------------


def on_error_return_none(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            print("Error trap!", ex)

    return inner


# --------------------------------------------------------------------
#  IMAGE PROCESSING
# --------------------------------------------------------------------


def load_image(response) -> tuple[np.ndarray, int]:
    img = np.asarray(bytearray(response.content), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


# https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
def resize(
    image: np.ndarray, width=None, height=None, inter=cv2.INTER_AREA
) -> np.ndarray:
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


@on_error_return_none
def laplacian_blur_score(image: np.ndarray) -> float:
    return cv2.Laplacian(image, cv2.CV_64F).var()


@on_error_return_none
def fft_blur_score(image: np.ndarray, size=60, width=500) -> tuple[float, float]:
    resized = resize(image, width=width)
    (h, w) = resized.shape
    (cX, cY) = (w // 2, h // 2)

    fft = np.fft.fft2(resized)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    fft_measure = np.mean(magnitude)

    return fft_measure


@on_error_return_none
def sift_keypoints(image: np.ndarray, template: np.ndarray) -> float:
    # SIFT + Lowe ratio
    # https://stackoverflow.com/a/50161781/5561144
    lowe_ratio = 0.75
    sift = cv2.SIFT_create()

    _, des1 = sift.detectAndCompute(template, None)
    _, des2 = sift.detectAndCompute(image, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m.distance < lowe_ratio * n.distance for m, n in matches]

    return len(matches), sum(good_matches)


@on_error_return_none
def similarity_scores(image: np.ndarray, template: np.ndarray) -> float:
    # Structural similarity index
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
    image_resized = cv2.resize(
        image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA
    )

    mse = mean_squared_error(image_resized, template)
    ssim = structural_similarity(
        image_resized, template, data_range=template.max() - template.min()
    )

    return mse, ssim


@on_error_return_none
def detect_aruco(image: np.ndarray) -> bool:
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)
    (corners, ids, _) = detector.detectMarkers(image)

    if len(corners) > 0:
        return True, ids.ravel().tolist()

    return False, None


# --------------------------------------------------------------------
#  PROCESSING
# --------------------------------------------------------------------


def init_worker(template_arr):
    global global_template_arr
    global_template_arr = template_arr


@on_error_return_none
def process_single(s: str) -> list[str]:
    if s is None or len(s.strip()) == 0:
        return None

    # parse line
    row = json.loads(s)

    # get global template
    global global_template_arr
    current_template = global_template_arr[row["page"] - 1, :, :]

    # download image
    response = requests.get(row["url"])
    img = load_image(response)

    (h, w) = img.shape
    img_ext = row["url"].split(".")[-1]
    img_size = len(response.content)
    img_hash = sha3_256(response.content).hexdigest()

    # get similarity measures
    sift_total, sift_good = sift_keypoints(img, current_template)
    sim_mse, sim_ssim = similarity_scores(img, current_template)

    # aruco detection
    aruco_detected, aruco_ids = detect_aruco(img)

    return {
        "id": row["id"],
        "page": row["page"],
        # file meta
        "hash": img_hash,
        "extension": img_ext,
        "size_bytes": img_size,
        # image data
        "height": h,
        "width": w,
        # blur measures
        "laplacian_blur_score": laplacian_blur_score(img),
        "fft_blur_score": fft_blur_score(img),
        # similarity measures
        "sift_keypoints_total": sift_total,
        "sift_keypoints_good": sift_good,
        "mse_score": sim_mse,
        "ssim_score": sim_ssim,
        # aruco detection
        "aruco_detected": aruco_detected,
        "aruco_ids": aruco_ids,
    }


# --------------------------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------------------------


def main(args):
    # open input and output file
    with (
        open(args.input_file, "r") as infile,
        open(args.output_file, "w") as outfile,
    ):
        # load template file
        template = np.load(args.template_file)

        # create worker pool
        with Pool(
            processes=args.jobs, initializer=init_worker, initargs=(template,)
        ) as executor:
            # submit jobs
            jobs = executor.imap_unordered(
                process_single, infile, chunksize=args.chunk_size
            )

            # save results
            for result in (pbar := tqdm(jobs, total=args.total)):
                json.dump(result, outfile)
                outfile.write("\n")

                pbar.set_description(f"Processing {result['id']}")


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--jobs", type=int, default=cpu_count())
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--total", type=int, default=1999658)

    # parse CLI
    args = parser.parse_args()

    # start processing
    main(args)
