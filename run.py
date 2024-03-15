import csv
from joblib import Memory, Parallel, delayed

from model import Row
from processors import measure_blur, measure_similarity, get_template, get_image


def load_template(name: str):
    return get_template(name)


def main():
    # create cache
    memory = Memory(location='cache', verbose=0)

    # create cached functions
    load_template_cached = memory.cache(load_template)

    # image processor
    def process(line: str):
        try:
            # parse line
            row = Row.from_line(line)
            if row is None or not row.is_ok:
                return None

            # get pleno c1
            img = get_image(row.chasil_hal_1)

            # measure blur and similarity
            laplacian_blur, fft_blur = measure_blur(img)
            similarity = measure_similarity(img, load_template_cached("dataset/1101102001002_002_p1.jpg"))

            # print(f"Processed {row.kode} TPS {row.kode}")
            return f"{row.kode},{row.tps},{laplacian_blur},{fft_blur},{similarity}\n"
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

        # write to file
        for line in output_gen:
            if line is None:
                continue

            outfile.write(line)



if __name__ == '__main__':
    main()
