import csv
import numpy as np
from joblib import Memory, Parallel, delayed

from model import Row
from processors import measure_blur, measure_similarity, get_template, get_image

TEMPLATE_PLENO_P1 = "dataset/benchmark/1101102001002_002_p1.jpg"
TEMPLATE_PLENO_P2 = "dataset/benchmark/1101102001002_002_p2.jpg"
TEMPLATE_PLENO_P3 = "dataset/benchmark/1101102001002_002_p3.jpg"

def load_template(name: str):
    return get_template(name)

def perform_measures(id: str, url: str, template: np.ndarray) -> str:
    # download image
    img = get_image(url)

    # measure blur and similarity
    laplacian_blur, fft_blur = measure_blur(img)
    similarity = measure_similarity(img, template)

    return f"{id},{laplacian_blur},{fft_blur},{similarity}\n"

def main():
    # create cache
    memory = Memory(location='cache', verbose=0)

    # create cached functions
    load_template_cached = memory.cache(load_template)

    # image processor
    def process(line: str) -> list[str]:
        try:
            # parse line
            row = Row.from_line(line)
            if row is None or not row.is_ok:
                return None

            # measure blur and similarity
            return [
                perform_measures(f"{row.kode},{row.tps},p1", row.chasil_hal_1, load_template_cached(TEMPLATE_PLENO_P1)),
                perform_measures(f"{row.kode},{row.tps},p2", row.chasil_hal_2, load_template_cached(TEMPLATE_PLENO_P2)),
                perform_measures(f"{row.kode},{row.tps},p3", row.chasil_hal_3, load_template_cached(TEMPLATE_PLENO_P3)),
            ]
        except Exception as e:
            print(f"Error processing {line}: {e}")
            return None
    
    # open input and output file
    with open("dataset/pemilu.csv", "r") as infile, open("result_c1.csv", "w") as outfile:
        # create csv reader
        csvreader = csv.reader(infile, delimiter=',', quotechar='"')

        # create parallel
        parallel = Parallel(n_jobs=-1, return_as="generator", verbose=10)
        output_gen = parallel(delayed(process)(line) for line in csvreader)

        # write header
        outfile.write("kode,tps,page,laplacian_blur,fft_blur,similarity\n")

        # write to file
        for out in output_gen:
            if out is None:
                continue

            outfile.writelines(out)


if __name__ == '__main__':
    main()
