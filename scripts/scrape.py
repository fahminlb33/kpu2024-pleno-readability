import scrapy

PARTS_SEGMENTS = [2, 4, 6, 10, 13]


def create_urls(kode, tingkat):
    parts = [kode[: PARTS_SEGMENTS[i]] for i in range(tingkat)]
    filename = "/".join(parts) + ".json"

    if tingkat < 5:
        yield f"https://sirekap-obj-data.kpu.go.id/wilayah/pemilu/ppwp/{filename}"

    yield f"https://sirekap-obj-data.kpu.go.id/pemilu/hhcw/ppwp/{filename}"


class SirekapSpider(scrapy.Spider):
    name = "sirekap"
    start_urls = ["https://sirekap-obj-data.kpu.go.id/wilayah/pemilu/ppwp/0.json"]
    custom_settings = {
        "LOG_LEVEL": "INFO",
        "JOBDIR": "data/crawls",
        "FEEDS": {
            "data/scraped/kpu2024.jsonl": {
                "format": "jsonlines",
                "encoding": "utf8",
                "store_empty": False,
            },
        },
    }

    def parse(self, response):
        body = response.json()

        # wilayah file has array in its root
        if isinstance(body, list):
            for wilayah in body:
                yield {**wilayah, "url": response.url}

                for next_url in create_urls(wilayah["kode"], wilayah["tingkat"]):
                    yield response.follow(next_url, self.parse)
        else:
            yield {**body, "url": response.url}
