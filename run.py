import csv
import argparse

import numpy as np
from joblib import Memory, Parallel, delayed

from model import Row
from processors import measure_blur, measure_similarity, get_template, get_image


TEMPLATE_PLENO_P1 = "dataset/benchmark/1101102001002_002_p1.jpg"
TEMPLATE_PLENO_P2 = "dataset/benchmark/1101102001002_002_p2.jpg"
TEMPLATE_PLENO_P3 = "dataset/benchmark/1101102001002_002_p3.jpg"


class Processor:
    def __init__(self, input_pemilu_path: str, output_path: str, n_jobs: int) -> None:
        # set input and output path
        self.input_path = input_pemilu_path
        self.output_file = output_path
        self.n_jobs = n_jobs

        # create cache
        self.memory = Memory(location='cache', verbose=0)

        # create template cache
        self.load_template = self.memory.cache(get_template)


    def inner_measures(self, id: str, url: str, template: np.ndarray) -> str:
        # download image
        img, size_bytes, img_hash = get_image(url)
        (h, w) = img.shape

        # get file extension
        ext = url.split('.')[-1]

        # measure blur and similarity
        laplacian_blur, fft_blur = measure_blur(img)
        similarity = measure_similarity(img, template)

        return f"{id},{size_bytes},{ext},{h},{w},{laplacian_blur},{fft_blur},{similarity},{img_hash}\n"


    def process_single(self, line: str) -> list[str]:
        try:
            # parse line
            row = Row.from_line(line)
            if row is None or not row.is_ok:
                return None

            # measure blur and similarity
            return [
                self.inner_measures(f"{row.kode},{row.tps},1", row.chasil_hal_1, self.load_template(TEMPLATE_PLENO_P1)),
                self.inner_measures(f"{row.kode},{row.tps},2", row.chasil_hal_2, self.load_template(TEMPLATE_PLENO_P2)),
                self.inner_measures(f"{row.kode},{row.tps},3", row.chasil_hal_3, self.load_template(TEMPLATE_PLENO_P3)),
            ]
        except Exception as e:
            print(f"Error processing {line}: {e}")
            return None


    def run(self):              
        # open input and output file
        with open(self.input_path, "r") as infile, open(self.output_file, "w") as outfile:
            # create csv reader
            csvreader = csv.reader(infile, delimiter=',', quotechar='"')

            # skip first header
            for _ in range(1):
                next(csvreader)

            # create parallel
            parallel = Parallel(n_jobs=self.n_jobs, return_as="generator", verbose=10)
            output_gen = parallel(delayed(self.process_single)(line) for line in csvreader)

            # write header
            outfile.write("kode,tps,page,size_bytes,extension,height,width,laplacian_blur,fft_blur,similarity,hash\n")

            # write to file
            for out in output_gen:
                if out is None:
                    continue

                outfile.writelines(out)


if __name__ == '__main__':
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input pemilu csv file", default="dataset/pemilu.csv")
    parser.add_argument("--output", type=str, help="Path to output file", default="result.csv")
    parser.add_argument("--jobs", type=int, help="Number of jobs", default=-1)

    # parse CLI
    args = parser.parse_args()

    # start processing
    processor = Processor(args.input, args.output, args.jobs)
    processor.run()
