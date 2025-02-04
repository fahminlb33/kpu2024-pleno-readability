import json

from tqdm import tqdm

TOTAL_ROWS = 1830652

if __name__ == "__main__":
    with (
        open("./data/scraped/kpu2024.jsonl", "r") as input_file,
        open("./data/scraped/wilayah.jsonl", "w") as wilayah_file,
        open("./data/scraped/summaries.jsonl", "w") as summary_file,
        open("./data/scraped/tps.jsonl", "w") as tps_file,
    ):
        for line in tqdm(input_file, total=TOTAL_ROWS):
            parsed = json.loads(line)

            if "hhcw" not in parsed["url"]:
                wilayah_file.write(line)
            elif "images" in parsed:
                tps_file.write(line)
            else:
                summary_file.write(line)
