#!/usr/bin/env python3
"""Perbarui data/forecast.json dari prakiraan musiman Copernicus C3S.

- Anomali Niño3.4: ensemble mean model dikurangi klimatologi hindcast model itu sendiri.
- Peluang tercile hujan: porsi anggota ensemble yang jatuh di bawah/di atas tercile hindcast
  untuk kotak lokal (mentah, BELUM dikalibrasi ke pos hujan).
Skrip membaca CDSAPI_URL dan CDSAPI_KEY dari environment.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"      # klimatologi hindcast, di-commit supaya tidak diunduh ulang
TMP = ROOT / "tmp"
OUT = DATA / "forecast.json"

DATASET = "seasonal-monthly-single-levels"
NINO34 = [5, -170, -5, -120]                  # utara, barat, selatan, timur
LOCAL = json.loads(os.environ.get("LOCAL_AREA", "[-2, 118, -8, 123]"))  # Sulawesi Selatan
LOCATION = os.environ.get("LOCATION", "Sulawesi Selatan")
LEADS = [2, 3, 4]                             # lead 1 = bulan rilis; 3 bulan berikutnya = 2-4
HINDCAST_YEARS = range(1993, 2017)

# Label: (originating_centre, system). CEK nomor system di form CDS sebelum dipakai;
# tambahkan pusat lain (meteo_france, dwd, cmcc, ncep) setelah nomornya dipastikan.
CENTRES = {
    "ECMWF": ("ecmwf", "51"),
    "UK Met Office": ("ukmo", "610"),
}

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def fetch(client, centre, system, var, area, years, month, out):
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve(DATASET, {
        "originating_centre": centre,
        "system": system,
        "variable": [var],
        "product_type": ["monthly_mean"],
        "year": [str(y) for y in years],
        "month": [f"{month:02d}"],
        "leadtime_month": [str(l) for l in LEADS],
        "area": area,
        "data_format": "grib",
    }, str(out))
    return out


def pool(path):
    """Rata-rata area (berbobot cos lintang) -> array (lead, sampel); sampel = anggota x tahun."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    lat = next(d for d in da.dims if d.startswith("lat"))
    lon = next(d for d in da.dims if d.startswith("lon"))
    da = da.weighted(np.cos(np.deg2rad(da[lat]))).mean([lat, lon])
    lead = "forecastMonth" if "forecastMonth" in da.dims else "step"
    da = da.transpose(lead, ...)
    return da.values.reshape(da.sizes[lead], -1)


def run_model(client, centre, system, start):
    m, y = start.month, start.year
    key = f"{centre}_{system}_{m:02d}"

    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        clim = json.loads(cache_file.read_text())
    else:
        hs = pool(fetch(client, centre, system, "sea_surface_temperature", NINO34,
                        HINDCAST_YEARS, m, TMP / f"{key}_hc_sst.grib"))
        ht = pool(fetch(client, centre, system, "total_precipitation", LOCAL,
                        HINDCAST_YEARS, m, TMP / f"{key}_hc_tp.grib"))
        clim = {
            "sst": np.nanmean(hs, axis=1).tolist(),
            "t33": np.nanpercentile(ht, 100 / 3, axis=1).tolist(),
            "t67": np.nanpercentile(ht, 200 / 3, axis=1).tolist(),
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(clim))

    fs = pool(fetch(client, centre, system, "sea_surface_temperature", NINO34,
                    [y], m, TMP / f"{key}_{y}_fc_sst.grib"))
    ft = pool(fetch(client, centre, system, "total_precipitation", LOCAL,
                    [y], m, TMP / f"{key}_{y}_fc_tp.grib"))

    anom = np.round(np.nanmean(fs, axis=1) - np.array(clim["sst"]), 2).tolist()
    t33 = np.array(clim["t33"])[:, None]
    t67 = np.array(clim["t67"])[:, None]
    below = (ft < t33).mean(axis=1)
    above = (ft > t67).mean(axis=1)
    return anom, np.stack([below, 1 - below - above, above])


def main():
    now = dt.datetime.now(dt.timezone.utc)
    ym = os.environ.get("START", now.strftime("%Y-%m"))
    start = dt.date(int(ym[:4]), int(ym[5:7]), 1)

    if OUT.exists() and not os.environ.get("FORCE"):
        if json.loads(OUT.read_text()).get("start") == ym:
            print(f"Data {ym} sudah terbaru; dilewati (set FORCE=1 untuk memaksa).")
            return

    # Bawaan cdsapi: 500 percobaan x 120 detik, bisa menggantung berjam-jam saat server error.
    client = cdsapi.Client(retry_max=3, sleep_max=60)
    models, probs = {}, []
    for label, (centre, system) in CENTRES.items():
        try:
            anom, p = run_model(client, centre, system, start)
            models[label] = anom
            probs.append(p)
            print(f"OK   {label}: {anom}")
        except Exception as e:  # satu model gagal tidak menggagalkan semuanya
            print(f"GAGAL {label}: {e}", file=sys.stderr)

    if not models:
        sys.exit("Tidak ada model yang berhasil diunduh; data lama dibiarkan.")

    P = np.mean(probs, axis=0)
    below = np.rint(P[0] * 100).astype(int)
    above = np.rint(P[2] * 100).astype(int)
    normal = 100 - below - above

    months = []
    for L in LEADS:
        idx = start.month - 1 + (L - 1)
        months.append(f"{BULAN[idx % 12]} {start.year + idx // 12}")

    DATA.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "start": ym,
        "issued": now.strftime("%Y-%m-%d"),
        "location": LOCATION,
        "months": months,
        "nino34": {"models": models},
        "tercile": {"below": below.tolist(), "normal": normal.tolist(), "above": above.tolist()},
        "note": ("Sumber: Copernicus C3S. Anomali terhadap klimatologi hindcast 1993–2016 tiap model. "
                 "Peluang hujan dihitung langsung dari anggota ensemble dan belum dikalibrasi ke pos hujan. "
                 "Bukan prakiraan resmi BMKG."),
    }, ensure_ascii=False, indent=2))
    print("Menulis", OUT)


if __name__ == "__main__":
    main()
