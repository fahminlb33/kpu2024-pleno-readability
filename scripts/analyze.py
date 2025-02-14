import os
import json
import zipfile
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
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

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
def sift_keypoints(sift, image: np.ndarray, template: np.ndarray, lowe_ratio = 0.75) -> float: 
    # SIFT + Lowe ratio
    # https://stackoverflow.com/a/50161781/5561144
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
def detect_aruco(arucoDetector: cv2.aruco.ArucoDetector, image: np.ndarray) -> bool:
    # crop image
    img_crop = image[: int(image.shape[0] / 4), int(image.shape[1] / 1.5) :].copy()
    center_x = img_crop.shape[1] / 3

    # detect aruco
    (corners, ids, _) = arucoDetector.detectMarkers(img_crop)

    if len(corners) > 0:
        return True, ids.ravel().tolist()

    # detect squares
    img_blur = cv2.GaussianBlur(img_crop, (5, 5), 1)
    img_edges = cv2.Canny(img_blur, 10, 50)
    contours = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, 10, True)
        area = w * h #cv2.contourArea(contour)

        if len(approx) < 4 or area < 900 or x < center_x:
            continue

        aspect_ratio = w / h
        if aspect_ratio > 0.8 and aspect_ratio < 1.2:
            return True, None

    # aruco not detected, square contour not detected
    return False, None


# --------------------------------------------------------------------
#  PROCESSING
# --------------------------------------------------------------------

def init_worker(template_file_path):
    global g_template_file_path
    g_template_file_path = template_file_path


@functools.lru_cache()
def get_worker_data():
    # load template file
    global g_template_file_path
    template: np.ndarray = np.load(g_template_file_path)

    # SIFT detector
    sift = cv2.SIFT_create()

    # ARUCO detector
    arucoParams = cv2.aruco.DetectorParameters()
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
    detector = cv2.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)

    print(f"INIT PID:{os.getpid()} PPID:{os.getppid()}")

    return template, sift, detector


def process_single(s: str) -> list[str]:
    if s is None or len(s.strip()) == 0:
        return None
    
    template, sift, detector = get_worker_data()

    try:
        # parse line
        row = json.loads(s)

        # get global template
        current_template = template[row["page"] - 1, :, :]

        # download image
        response = requests.get(row["url"])
        img = load_image(response)

        (h, w) = img.shape
        img_ext = row["url"].split(".")[-1]
        img_size = len(response.content)
        img_hash = sha3_256(response.content).hexdigest()

        # get similarity measures
        sift_total, sift_good = sift_keypoints(sift, img, current_template)
        sim_mse, sim_ssim = similarity_scores(img, current_template)

        # aruco detection
        aruco_likely_detected, aruco_ids = detect_aruco(detector, img)

        return {
            "success": True,
            # original data
            **row,
            # file meta
            "hash": img_hash,
            "extension": img_ext,
            "size_bytes": img_size,
            "bytes": response.content,
            # image data
            "width": w,
            "height": h,
            # blur measures
            "fft_blur_score": fft_blur_score(img),
            "laplacian_blur_score": laplacian_blur_score(img),
            # similarity measures
            "mse_score": sim_mse,
            "ssim_score": sim_ssim,
            "sift_keypoints_good": sift_good,
            "sift_keypoints_total": sift_total,
            # aruco detection
            "aruco_ids": aruco_ids,
            "aruco_likely_detected": aruco_likely_detected,
        }
    except Exception as ex:
            return {
                "success": False,
                # original data
                **row,
                # error details
                "error": repr(ex),
            }


# --------------------------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------------------------


def main(args):
    # create output directory
    archive_dir = os.path.join(args.output_dir, "archives")
    os.makedirs(archive_dir, exist_ok=True)

    # open input and output file
    with (
        open(args.input_file, "r") as infile,
        open(os.path.join(args.output_dir, "tps-analysis.jsonl"), "a+") as outfile,
    ):
        # create worker pool
        with Pool(
            processes=args.jobs, initializer=(init_worker), initargs=(args.template_file,)
        ) as executor:
            # submit jobs
            jobs = executor.imap_unordered(
                process_single, infile, chunksize=args.chunk_size
            )

            file_count = 0
            archive_count = 1

            # save results
            for result in (pbar := tqdm(jobs, total=args.total, mininterval=1)):

                # max number of files reached
                if file_count == args.files_chunks:
                    # close archive
                    archive_file.close()

                    # TODO: start upload

                    # update states
                    file_count = 0
                    archive_count += 1
                    pbar.set_description(f"Current archive: {archive_count}")

                # create new archive
                if file_count == 0:
                    archive_filename = os.path.join(archive_dir, f"./kpu2024-part_{archive_count}.zip")
                    archive_file = zipfile.ZipFile(archive_filename, "w", compression=zipfile.ZIP_DEFLATED)

                # write to ZIP
                # if this file has an error, skip
                if result["success"]:
                    file_name = sha3_256(result["url"].encode("utf8")).hexdigest()
                    file_name += "." + result["url"].split("/")[-1].split(".")[1]
                    archive_file.writestr(file_name, result["bytes"])

                    file_count += 1
                    del result["bytes"]

                # write output to JSON
                json.dump(result, outfile)
                outfile.write("\n")


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--jobs", type=int, default=cpu_count())
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--files_chunks", type=int, default=5000)
    parser.add_argument("--total", type=int, default=1999658)

    # parse CLI
    args = parser.parse_args()

    # start processing
    main(args)
