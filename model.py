from dataclasses import dataclass


@dataclass
class Row:
    kode: str
    provinsi_kode: str
    kabupaten_kota_kode: str
    kecamatan_kode: str
    kelurahan_desa_kode: str
    tps: str
    suara_paslon_1: str
    suara_paslon_2: str
    suara_paslon_3: str
    chasil_hal_1: str
    chasil_hal_2: str
    chasil_hal_3: str
    suara_sah: str
    suara_total: str
    pemilih_dpt_j: str
    pemilih_dpt_l: str
    pemilih_dpt_p: str
    pengguna_dpt_j: str
    pengguna_dpt_l: str
    pengguna_dpt_p: str
    pengguna_dptb_j: str
    pengguna_dptb_l: str
    pengguna_dptb_p: str
    suara_tidak_sah: str
    pengguna_total_j: str
    pengguna_total_l: str
    pengguna_total_p: str
    pengguna_non_dpt_j: str
    pengguna_non_dpt_l: str
    pengguna_non_dpt_p: str
    psu: str
    ts: str
    status_suara: str
    status_adm: str
    updated_at: str
    created_at: str
    url_page: str
    provinsi_nama: str
    kabupaten_kota_nama: str
    kecamatan_nama: str
    kelurahan_desa_nama: str
    url_api: str
    fetch_count: str

    @property
    def is_ok(self):
        return self.kode != 'kode' and self.chasil_hal_1 and self.chasil_hal_2 and self.chasil_hal_3

    @staticmethod
    def from_line(line: list[str]):
        try:
            return Row(*line)
        except Exception as e:
            print("ERROR parsing line:", line)
            print(e)
            return None
            
