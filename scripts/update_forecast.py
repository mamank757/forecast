#!/usr/bin/env python3
"""Perbarui data/forecast.json dari Open-Meteo Seasonal API (ECMWF SEAS5) dan NOAA CPC. Tanpa akun/token.

- Jumlah bulan prakiraan: FORECAST_MONTHS (default 5), dihitung mulai bulan setelah bulan rilis.
- Anomali Niño3.4 (prakiraan): rata-rata anomali SST bulanan di 9 titik dalam kotak 5°LU-5°LS, 170°BB-120°BB.
- Anomali Niño3.4 TRADISIONAL (analisis lampau): nilai teramati bulanan NOAA CPC (OISST v2.1, klimatologi 1991-2020)
  untuk PAST_MONTHS bulan terakhir (default 6 = 5 bulan sebelumnya + bulan terkini yang sudah dirilis).
  Sebagai pembanding dengan BoM/NOAA resmi, ditambahkan Niño3.4 RELATIF bulanan (CPC Rnino34, ERSSTv6)
  untuk rentang bulan yang sama. Prakiraan SEAS5 tetap tradisional (belum dikurangi rata-rata tropis).
- Peluang hujan: porsi anggota ensemble (51, dari data 6-jam) yang anomali relatifnya jatuh
  di bawah/di atas tercile anomali relatif hujan bulanan ERA5 1991-2020 di titik lokal.
Catatan: nama parameter/kolom Open-Meteo dan format berkas CPC belum diuji dari sini. Jalankan sekali dan baca lognya.
"""
import calendar
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
CPC_NINO = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
CPC_REL = "https://www.cpc.ncep.noaa.gov/data/indices/Rnino34.ascii.txt"

# Titik lokal [lintang, bujur]; ganti lewat env POINTS, mis. '[[-5.15,119.4],[-4.0,119.6]]'
POINTS = json.loads(os.environ.get("POINTS", "[[-5.15, 119.40]]"))
LOCATION = os.environ.get("LOCATION", "Makassar, Sulawesi Selatan")
NINO_POINTS = [[lat, lon] for lat in (-4, 0, 4) for lon in (-165, -145, -125)]
PAST_MONTHS = int(os.environ.get("PAST_MONTHS", "6"))  # 5 bulan lalu + bulan terkini
FORECAST_MONTHS = int(os.environ.get("FORECAST_MONTHS", "5"))  # bulan ke depan (maks. ~6 untuk SEAS5)

BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def get(url, params=None, text=False):
    """GET dengan retry untuk error jaringan/5xx; error 4xx (permintaan salah) langsung berhenti."""
    err = ""
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=90,
                             headers={"User-Agent": "forecast-updater (GitHub Actions)"})
            if r.status_code == 200:
                return r.text if text else r.json()
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
    """Indeks FORECAST_MONTHS bulan setelah bulan rilis di daftar waktu respons."""
    idx, labels, nums = [], [], []
    for i in range(1, FORECAST_MONTHS + 1):
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


def parse_cpc_nino34(text, n):
    """Ambil n baris bulanan terakhir dari sstoi.indices NOAA CPC: (tahun, bulan, anomali Niño3.4).

    Kolom dibaca lewat baris judul (YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM);
    jika judul tidak dikenali, dipakai kolom ke-10 (indeks 9).
    """
    col = 9
    rows = []
    for line in text.splitlines():
        p = line.split()
        if "NINO3.4" in p:
            col = p.index("NINO3.4") + 1  # kolom ANOM tepat setelah NINO3.4
            continue
        if len(p) > col and p[0].isdigit() and p[1].isdigit():
            try:
                y, mo, v = int(p[0]), int(p[1]), float(p[col])
            except ValueError:
                continue
            if 1 <= mo <= 12 and abs(v) < 5:  # saring baris rusak/nilai hilang (-99.99)
                rows.append((y, mo, v))
    if not rows:
        raise RuntimeError(f"Tidak ada baris Niño3.4 yang terbaca dari CPC. Awal berkas: {text[:200]!r}")
    return rows[-n:]


def parse_rnino34(text):
    """Baris (tahun, bulan, anomali) dari Rnino34.ascii.txt CPC; format `YR MTH ANOM`."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
            try:
                y, mo, v = int(p[0]), int(p[1]), float(p[2])
            except ValueError:
                continue
            if 1 <= mo <= 12 and abs(v) < 9:
                rows.append((y, mo, v))
    if not rows:
        raise RuntimeError(f"Tidak ada baris Niño3.4 relatif yang terbaca. Awal berkas: {text[:200]!r}")
    return rows


def nino34_relative(first_key):
    """Niño3.4 relatif bulanan (CPC, ERSSTv6) mulai dari bulan `first_key` (YYYY-MM)."""
    rows = [r for r in parse_rnino34(get(CPC_REL, text=True)) if f"{r[0]}-{r[1]:02d}" >= first_key]
    if not rows:
        raise RuntimeError(f"Niño3.4 relatif belum memuat bulan sejak {first_key}")
    return {
        "source": "NOAA CPC, Niño3.4 relatif (ERSSTv6), anomali terhadap 1991–2020",
        "months": [f"{BULAN[mo - 1]} {y}" for y, mo, _ in rows],
        "keys": [f"{y}-{mo:02d}" for y, mo, _ in rows],
        "nino34": [round(v, 2) for _, _, v in rows],
    }


def nino34_observed(n=PAST_MONTHS):
    """Analisis Niño3.4 teramati (NOAA CPC) untuk n bulan terakhir yang sudah dirilis."""
    rows = parse_cpc_nino34(get(CPC_NINO, text=True), n)
    return {
        "source": "NOAA CPC, Niño3.4 tradisional (OISST v2.1), anomali terhadap 1991–2020",
        "months": [f"{BULAN[mo - 1]} {y}" for y, mo, _ in rows],
        "keys": [f"{y}-{mo:02d}" for y, mo, _ in rows],
        "nino34": [round(v, 2) for _, _, v in rows],
    }


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


def forecast_days(start):
    """Hari prakiraan yang diminta agar data 6-jam mencakup akhir bulan target terakhir.

    Bawaan Open-Meteo hanya ~92 hari, tidak cukup untuk 4-5 bulan ke depan.
    Dipilih nilai terkecil dari 92/183/274 yang cukup.
    """
    k = start.month - 1 + FORECAST_MONTHS + 1  # indeks (basis 0) bulan setelah bulan target terakhir
    end = dt.date(start.year + k // 12, k % 12 + 1, 1) - dt.timedelta(days=1)
    need = (end - dt.datetime.now(dt.timezone.utc).date()).days + 3
    return next((d for d in (92, 183, 274) if d >= need), 274)


def fetch_members(lat, lon, days):
    """Ambil data 6-jam per anggota; nama parameter dan bagian respons dideteksi otomatis."""
    last = None
    for param in ("six_hourly", "hourly"):
        loc = as_list(get(SEASONAL, {
            "latitude": lat, "longitude": lon,
            "models": "ecmwf_seas5",
            "forecast_days": days,
            param: "precipitation",
            "monthly": "precipitation_mean,precipitation_anomaly",
        }))[0]
        last = list(loc)
        for k, v in loc.items():
            if (k != "monthly" and not k.endswith("_units") and isinstance(v, dict)
                    and "time" in v and any(c.startswith("precipitation") for c in v)):
                print(f"Data 6-jam: parameter '{param}' -> kunci respons '{k}'", file=sys.stderr)
                return loc, v
        print(f"Parameter '{param}' tidak menghasilkan data 6-jam. Kunci respons: {last}", file=sys.stderr)
    raise RuntimeError(f"Respons tidak memuat data 6-jam per anggota. Kunci respons: {last}")


def precip_probs(start):
    """Rata-rata antar titik dari peluang (di bawah, normal, di atas) per bulan target.

    Data bulanan Open-Meteo hanya berisi rata-rata ensemble, jadi total hujan per anggota
    dihitung dari data 6-jam. Klimatologi model = (mean - anomali) dalam satuan data bulanan,
    lalu diskalakan ke mm/bulan dengan membandingkan rata-rata anggota terhadap nilai mean.
    """
    all_p, labels = [], None
    days = forecast_days(start)
    print(f"forecast_days={days} untuk {FORECAST_MONTHS} bulan prakiraan", file=sys.stderr)
    for lat, lon in POINTS:
        loc, sh = fetch_members(lat, lon, days)
        mo = loc["monthly"]
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
            last_day = calendar.monthrange(int(key[:4]), int(key[5:]))[1]
            if stamps[-1][:10] < f"{key}-{last_day:02d}":
                print(f"Peringatan: data 6-jam berakhir {stamps[-1][:10]}, sebelum akhir {key}; "
                      "bulan ini hanya terhitung sebagian.", file=sys.stderr)
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


def latest_expected_key(start):
    """Kunci bulan (YYYY-MM) terakhir yang seharusnya sudah ada di analisis lampau: bulan sebelum rilis."""
    k = start.month - 2
    return f"{start.year + k // 12}-{k % 12 + 1:02d}"


def main():
    now = dt.datetime.now(dt.timezone.utc)
    ym = os.environ.get("START", now.strftime("%Y-%m"))
    start = dt.date(int(ym[:4]), int(ym[5:7]), 1)

    if OUT.exists() and not os.environ.get("FORCE"):
        old = json.loads(OUT.read_text())
        keys = (old.get("past") or {}).get("keys") or [""]
        has_rel = bool((old.get("past") or {}).get("relative"))
        # Lewati hanya jika prakiraan bulan ini ada DAN analisis lampau sudah sampai bulan sebelumnya
        # (CPC kadang terlambat; run hari berikutnya akan mencoba lagi) dan jumlah bulannya sama
        # dengan FORECAST_MONTHS.
        if (old.get("start") == ym and len(old.get("months", [])) == FORECAST_MONTHS
                and keys[-1] >= latest_expected_key(start) and has_rel):
            print(f"Data {ym} sudah terbaru; dilewati (set FORCE=1 untuk memaksa).")
            return

    anom = nino34(start)
    print("Niño3.4 anomali:", anom)
    try:
        past = nino34_observed()
        print("Niño3.4 teramati (tradisional):", dict(zip(past["months"], past["nino34"])))
        try:
            past["relative"] = nino34_relative(past["keys"][0])
            print("Niño3.4 teramati (relatif):", dict(zip(past["relative"]["months"], past["relative"]["nino34"])))
        except Exception as e:  # pembanding saja
            print(f"Peringatan: Niño3.4 relatif gagal ({e}); garis relatif dilewati.", file=sys.stderr)
    except Exception as e:  # analisis lampau hanya pelengkap; jangan gagalkan prakiraan
        print(f"Peringatan: analisis Niño3.4 teramati gagal ({e}); grafik tanpa data lampau.", file=sys.stderr)
        past = None
    P, months = precip_probs(start)
    below = np.rint(P[0] * 100).astype(int)
    above = np.rint(P[2] * 100).astype(int)
    normal = 100 - below - above

    DATA.mkdir(exist_ok=True)
    out = {
        "start": ym,
        "issued": now.strftime("%Y-%m-%d"),
        "location": LOCATION,
        "months": months,
        "nino34": {"models": {"ECMWF SEAS5": anom}},
        "tercile": {"below": below.tolist(), "normal": normal.tolist(), "above": above.tolist()},
        "note": ("Sumber: ECMWF SEAS5 melalui Open-Meteo.com (CC BY 4.0, non-komersial); analisis Niño3.4 "
                 "teramati dari NOAA CPC. Prakiraan dan garis hitam memakai Niño3.4 tradisional; BoM dan NOAA "
                 "kini memakai indeks relatif (dikurangi rata-rata tropis), sehingga angkanya lebih rendah. "
                 "Anomali prakiraan terhadap klimatologi model, anomali analisis terhadap klimatologi 1991–2020. Peluang hujan dihitung dari anggota ensemble "
                 "terhadap tercile anomali relatif ERA5 1991–2020 dan belum dikalibrasi ke pos hujan. "
                 "Keandalan menurun untuk bulan yang lebih jauh. Bukan prakiraan resmi BMKG."),
    }
    if past:
        out["past"] = past
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("Menulis", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"GAGAL: {e}")
