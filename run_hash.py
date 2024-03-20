import csv
import smtplib
from email.message import EmailMessage

import numpy as np
from joblib import Memory, Parallel, delayed

from model import Row
from processors import get_image_hash


def perform_measures(id: str, url: str) -> str:
    # download image
    hash = get_image_hash(url)

    return f"{id},{hash}\n"

def main():
    # image processor
    def process(line: str) -> list[str]:
        try:
            # parse line
            row = Row.from_line(line)
            if row is None or not row.is_ok:
                return None

            # measure blur and similarity
            return [
                perform_measures(f"{row.kode},{row.tps},1", row.chasil_hal_1),
                perform_measures(f"{row.kode},{row.tps},2", row.chasil_hal_2),
                perform_measures(f"{row.kode},{row.tps},3", row.chasil_hal_3),
            ]
        except Exception as e:
            print(f"Error processing {line}: {e}")
            return None
    
    # open input and output file
    with open("dataset/pemilu.csv", "r") as infile, open("result_hash.csv", "w") as outfile:
        # create csv reader
        csvreader = csv.reader(infile, delimiter=',', quotechar='"')

        # optional skip
        for _ in range(1):
            next(csvreader)

        # create parallel
        parallel = Parallel(n_jobs=30, return_as="generator", verbose=10)
        output_gen = parallel(delayed(process)(line) for line in csvreader)

        # write header
        outfile.write("kode,tps,page,hash\n")

        # write to file
        for out in output_gen:
            if out is None:
                continue

            outfile.writelines(out)

    # build email
    mail = EmailMessage()
    mail.set_content("Scraping selesai!")
    mail["Subject"] = "Scraping Selesai!"
    mail["From"] = "fahminlb33@gmail.com"
    mail["To"] = "fahmi@kodesiana.com"

    # send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    print(server.ehlo())
    print(server.starttls())
    print(server.login('', 'solh rgja rcas gjew'))
    print(server.send_message(mail))
    server.close()


if __name__ == '__main__':
    main()
