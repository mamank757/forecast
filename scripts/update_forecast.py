#!/usr/bin/env python3
"""Perbarui data/forecast.json dari Open-Meteo Seasonal API (ECMWF SEAS5). Tanpa akun/token.

- Anomali Niño3.4: rata-rata anomali SST bulanan di 9 titik dalam kotak 5°LU-5°LS, 170°BB-120°BB.
- Peluang hujan: porsi anggota ensemble (51, dari data 6-jam) yang anomali relatifnya jatuh
  di bawah/di atas tercile anomali relatif hujan bulanan ERA5 1991-2020 di titik lokal.
Catatan: nama parameter/kolom Open-Meteo belum diuji dari sini. Jalankan sekali dan baca lognya.
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "forecast.json"

SEASONAL = "https://seasonal-api.open-meteo.com/v1/seasonal"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Titik lokal [lintang, bujur]; ganti lewat env POINTS, mis. '[[-5.15,119.4],[-4.0,119.6]]'
POINTS = json.loads(os.environ.get("POINTS", "[[-5.15, 119.40]]"))
LOCATION = os.environ.get("LOCATION", "Makassar, Sulawesi Selatan")
NINO_POINTS = [[lat, lon] for lat in (-4, 0, 4) for lon in (-165, -145, -125)]

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def get(url, params):
    """GET dengan retry untuk error jaringan/5xx; error 4xx (permintaan salah) langsung berhenti."""
    err = ""
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(err)
        except requests.RequestException as e:
            err = str(e)
        print(f"Percobaan {i + 1} gagal: {err}", file=sys.stderr)
        time.sleep(20 * (i + 1))
    raise RuntimeError(err)


def as_list(d):
    return d if isinstance(d, list) else [d]


def pick(monthly, var, kind):
    """Kolom bulanan yang namanya memuat `var` dan `kind` (mean/anomaly), terurut per anggota."""
    cols = sorted(k for k in monthly if k != "time" and var in k and kind in k)
    if not cols:
        raise RuntimeError(f"Kolom {var}/{kind} tidak ada. Kunci respons: {list(monthly)[:20]}")
    return cols


def target_index(times, start):
    """Indeks 3 bulan setelah bulan rilis (lead 2-4) di daftar waktu respons."""
    idx, labels, nums = [], [], []
    for i in (1, 2, 3):
        k = start.month - 1 + i
        key = f"{start.year + k // 12}-{k % 12 + 1:02d}"
        j = next((n for n, t in enumerate(times) if str(t).startswith(key)), None)
        if j is None:
            raise RuntimeError(f"Bulan {key} tidak ada di respons. Waktu: {times}")
        idx.append(j)
        labels.append(f"{BULAN[k % 12]} {start.year + k // 12}")
        nums.append(k % 12 + 1)
    return idx, labels, nums


def matrix(monthly, cols, idx):
    return np.array([[monthly[c][j] for j in idx] for c in cols], dtype=float)


def nino34(start):
    d = as_list(get(SEASONAL, {
        "latitude": ",".join(str(p[0]) for p in NINO_POINTS),
        "longitude": ",".join(str(p[1]) for p in NINO_POINTS),
        "models": "ecmwf_seas5_ensemble_mean",
        "monthly": "sea_surface_temperature_anomaly",
    }))
    per_point = []
    for loc in d:
        m = loc["monthly"]
        idx, _, _ = target_index(m["time"], start)
        cols = pick(m, "sea_surface_temperature", "anomaly")
        per_point.append(np.nanmean(matrix(m, cols, idx), axis=0))
    return np.round(np.nanmean(per_point, axis=0), 2).tolist()


def era5_rel_terciles(lat, lon):
    """Tercile (33%, 67%) anomali relatif hujan bulanan ERA5 1991-2020, per nomor bulan."""
    d = get(ARCHIVE, {
        "latitude": lat, "longitude": lon,
        "start_date": "1991-01-01", "end_date": "2020-12-31",
        "daily": "precipitation_sum", "timezone": "UTC",
    })["daily"]
    totals = {}
    for t, v in zip(d["time"], d["precipitation_sum"]):
        if v is not None:
            key = (int(t[:4]), int(t[5:7]))
            totals[key] = totals.get(key, 0.0) + v
    res = {}
    for mo in range(1, 13):
        x = np.array([totals[(y, mo)] for y in range(1991, 2021) if (y, mo) in totals])
        if len(x) >= 20 and x.mean() > 0:
            res[mo] = np.percentile(x / x.mean() - 1, [100 / 3, 200 / 3])
    return res


def precip_probs(start):
    """Rata-rata antar titik dari peluang (di bawah, normal, di atas) per bulan target.

    Data bulanan Open-Meteo hanya berisi rata-rata ensemble, jadi total hujan per anggota
    dihitung dari data 6-jam. Klimatologi model = (mean - anomali) dalam satuan data bulanan,
    lalu diskalakan ke mm/bulan dengan membandingkan rata-rata anggota terhadap nilai mean.
    """
    all_p, labels = [], None
    for lat, lon in POINTS:
        loc = as_list(get(SEASONAL, {
            "latitude": lat, "longitude": lon,
            "models": "ecmwf_seas5",
            "six_hourly": "precipitation",
            "monthly": "precipitation_mean,precipitation_anomaly",
        }))[0]
        sh, mo = loc["six_hourly"], loc["monthly"]
        print("Kunci six_hourly:", list(sh)[:5], "... total", len(sh), file=sys.stderr)
        cols = sorted(k for k in sh if k != "time" and k.startswith("precipitation"))
        if len(cols) < 10:
            raise RuntimeError(f"Butuh anggota ensemble; hanya {len(cols)} kolom: {list(sh)[:10]}")
        X = np.array([sh[c] for c in cols], dtype=float)      # (anggota, langkah 6 jam)
        stamps = [str(t) for t in sh["time"]]
        idx, labels, nums = target_index(mo["time"], start)
        terc = era5_rel_terciles(lat, lon)
        p = []
        for i, (j, num) in enumerate(zip(idx, nums)):
            kk = start.month - 1 + (i + 1)
            key = f"{start.year + kk // 12}-{kk % 12 + 1:02d}"
            sel = np.array([s.startswith(key) for s in stamps])
            T = X[:, sel].sum(axis=1)                          # total bulanan per anggota (mm)
            mean_v = float(mo["precipitation_mean"][j])
            anom_v = float(mo["precipitation_anomaly"][j])
            if mean_v <= 0 or sel.sum() == 0:
                raise RuntimeError(f"Data hujan {key} tidak valid (mean={mean_v}, langkah={sel.sum()})")
            scale = np.nanmean(T) / mean_v                     # satuan bulanan -> mm/bulan
            M = (mean_v - anom_v) * scale                      # klimatologi model (mm/bulan)
            if M <= 0:
                raise RuntimeError(f"Klimatologi model {key} tidak positif: {M}")
            R = T / M - 1                                      # anomali relatif per anggota
            print(f"{key}: langkah={sel.sum()}, skala={scale:.1f}, klimatologi={M:.0f} mm, "
                  f"anomali rata-rata={np.nanmean(R) * 100:.0f}%", file=sys.stderr)
            R = R[~np.isnan(R)]
            t33, t67 = terc[num]
            below, above = np.mean(R < t33), np.mean(R > t67)
            p.append([below, 1 - below - above, above])
        all_p.append(np.array(p).T)                            # (3 kategori, 3 bulan)
    return np.mean(all_p, axis=0), labels


def main():
    now = dt.datetime.now(dt.timezone.utc)
    ym = os.environ.get("START", now.strftime("%Y-%m"))
    start = dt.date(int(ym[:4]), int(ym[5:7]), 1)

    if OUT.exists() and not os.environ.get("FORCE"):
        if json.loads(OUT.read_text()).get("start") == ym:
            print(f"Data {ym} sudah terbaru; dilewati (set FORCE=1 untuk memaksa).")
            return

    anom = nino34(start)
    print("Niño3.4 anomali:", anom)
    P, months = precip_probs(start)
    below = np.rint(P[0] * 100).astype(int)
    above = np.rint(P[2] * 100).astype(int)
    normal = 100 - below - above

    DATA.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "start": ym,
        "issued": now.strftime("%Y-%m-%d"),
        "location": LOCATION,
        "months": months,
        "nino34": {"models": {"ECMWF SEAS5": anom}},
        "tercile": {"below": below.tolist(), "normal": normal.tolist(), "above": above.tolist()},
        "note": ("Sumber: ECMWF SEAS5 melalui Open-Meteo.com (CC BY 4.0, non-komersial). "
                 "Anomali terhadap klimatologi model. Peluang hujan dihitung dari anggota ensemble "
                 "terhadap tercile anomali relatif ERA5 1991–2020 dan belum dikalibrasi ke pos hujan. "
                 "Bukan prakiraan resmi BMKG."),
    }, ensure_ascii=False, indent=2))
    print("Menulis", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"GAGAL: {e}")
