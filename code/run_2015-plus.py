"""
Robust pipeline for automatically fetching and extracting FOMC SEP dot plot data.
Handles format differences across years (2012-present) automatically.

Usage:
cd /Users/knimm/Documents/FOMC_Project_Clean/notebooks
python3 run_all_seps.py
"""

import os
from datetime import date
import cv2
import fitz
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

# ── CONFIG ────────────────────────────────────────────────────────────
BASE_URL = "https://www.federalreserve.gov/monetarypolicy/files/"
SAVE_DIR = "sep_pdfs"
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents/FOMC_Project_Clean/outputs")
CSV_PATH = os.path.join(OUTPUT_DIR, "projections.csv")

# Y-axis maximum per meeting — update when new meetings are added
Y_AXIS_MAX = {
    (2012, 6): 6.0, (2012, 9): 6.0, (2012, 12): 6.0,
    (2013, 3): 5.0, (2013, 6): 5.0, (2013, 9): 5.0, (2013, 12): 5.0,
    (2014, 3): 5.0, (2014, 6): 5.0, (2014, 9): 5.0, (2014, 12): 5.0,
    (2015, 3): 5.0, (2015, 6): 5.0, (2015, 9): 5.0, (2015, 12): 5.0,
    (2016, 3): 5.0, (2016, 6): 5.0, (2016, 9): 5.0, (2016, 12): 5.0,
    (2017, 3): 5.0, (2017, 6): 5.0, (2017, 9): 5.0, (2017, 12): 5.0,
    (2018, 3): 5.0, (2018, 6): 5.0, (2018, 9): 5.0, (2018, 12): 5.0,
    (2019, 3): 5.0, (2019, 6): 5.0, (2019, 9): 5.0, (2019, 12): 5.0,
    (2020, 6): 4.0, (2020, 9): 5.0, (2020, 12): 4.0,
    (2021, 3): 4.0, (2021, 6): 4.0, (2021, 9): 4.0, (2021, 12): 4.0,
    (2022, 3): 4.0, (2022, 6): 5.0, (2022, 9): 6.0, (2022, 12): 6.0,
    (2023, 3): 6.0, (2023, 6): 7.0, (2023, 9): 7.0, (2023, 12): 7.0,
    (2024, 3): 7.0, (2024, 6): 7.0, (2024, 9): 7.0, (2024, 12): 7.0,
    (2025, 3): 6.0, (2025, 6): 6.0, (2025, 9): 6.0, (2025, 12): 6.0,
    (2026, 3): 6.0, (2026, 6): 6.0, (2026, 9): 6.0, (2026, 12): 6.0,
}

ALL_DATES = [
    # 2015
    date(2015, 3, 18), date(2015, 6, 17), date(2015, 9, 17), date(2015, 12, 16),
    # 2016
    date(2016, 3, 16), date(2016, 6, 15), date(2016, 9, 21), date(2016, 12, 14),
    # 2017
    date(2017, 3, 15), date(2017, 6, 14), date(2017, 9, 20), date(2017, 12, 13),
    # 2018
    date(2018, 3, 21), date(2018, 6, 13), date(2018, 9, 26), date(2018, 12, 19),
    # 2019
    date(2019, 3, 20), date(2019, 6, 19), date(2019, 9, 18), date(2019, 12, 11),
    # 2020 (no March - COVID)
    date(2020, 6, 10), date(2020, 9, 16), date(2020, 12, 16),
    # 2021
    date(2021, 3, 17), date(2021, 6, 16), date(2021, 9, 22), date(2021, 12, 15),
    # 2022
    date(2022, 3, 16), date(2022, 6, 15), date(2022, 9, 21), date(2022, 12, 14),
    # 2023
    date(2023, 3, 22), date(2023, 6, 14), date(2023, 9, 20), date(2023, 12, 13),
    # 2024
    date(2024, 3, 20), date(2024, 6, 12), date(2024, 9, 18), date(2024, 12, 18),
    # 2025
    date(2025, 3, 19), date(2025, 6, 18), date(2025, 9, 17), date(2025, 12, 10),
    # 2026
    date(2026, 3, 18), date(2026, 6, 17),
]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_page_index(sep_date):
    """2021+ and Dec 2020: page 4 (index 3). Everything else: page 3 (index 2)."""
    if sep_date.year >= 2021:
        return 3
    if sep_date.year == 2020 and sep_date.month == 12:
        return 3
    return 2


def get_crop(img, sep_date):
    """
    2012-2014: dot plot is in bottom half of page, under a bar chart.
    2015+: dot plot is standalone, use fixed top crop.
    """
    h, w = img.shape[:2]
    left = 200
    right = min(2200, w)
    if sep_date.year <= 2014:
        top = int(h * 0.52)
        bottom = h - 50
    else:
        top = 410
        bottom = min(2200, h)
    return img[top:bottom, left:right]


def get_hsv_range(sep_date):
    """Pre-2021 PDFs use a lighter shade of blue — widen the range."""
    if sep_date.year >= 2021:
        return np.array([85, 60, 60]), np.array([130, 255, 255])
    else:
        return np.array([80, 30, 30]), np.array([135, 255, 255])


def get_erosion_params(sep_date):
    """2025+ have smaller dots; older have larger dots that merge."""
    if sep_date.year >= 2025:
        return np.ones((3, 3), np.uint8), 1
    else:
        return np.ones((5, 5), np.uint8), 2


def download_sep(sep_date):
    local_path = os.path.join(SAVE_DIR, f"SEP_{sep_date}.pdf")
    if os.path.exists(local_path):
        print(f"  [CACHE]  {local_path}")
        return local_path
    url = BASE_URL + f"fomcprojtabl{sep_date.strftime('%Y%m%d')}.pdf"
    print(f"  [FETCH]  {url}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            print(f"  [OK]     Saved -> {local_path}")
            return local_path
        elif r.status_code == 404:
            print(f"  [SKIP]   Not published yet (404)")
            return None
        else:
            print(f"  [ERROR]  HTTP {r.status_code}")
            return None
    except requests.RequestException as e:
        print(f"  [ERROR]  Network error: {e}")
        return None


def detect_dots(mask, sep_date):
    kernel, iterations = get_erosion_params(sep_date)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8
    )
    centers = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10:
            centers.append((centroids[i][0], centroids[i][1]))
    return centers


def get_calibration(unique_y, sep_date):
    if len(unique_y) < 2:
        return None, None, None
    gaps = [unique_y[i + 1] - unique_y[i] for i in range(len(unique_y) - 1)]
    median_gap = np.median(gaps)
    top_grid_y = None
    for i in range(len(unique_y) - 1):
        if abs((unique_y[i + 1] - unique_y[i]) - median_gap) < 10:
            top_grid_y = unique_y[i]
            break
    if top_grid_y is None:
        return None, None, None
    pixels_per_0125 = median_gap / 2
    top_grid_value = Y_AXIS_MAX.get((sep_date.year, sep_date.month))
    if top_grid_value is None:
        print(f"  [WARN]  No Y_AXIS_MAX entry for {sep_date}")
        return None, None, None
    return top_grid_y, pixels_per_0125, top_grid_value


def y_to_rate(y, top_grid_y, pixels_per_0125, top_grid_value):
    steps = (y - top_grid_y) / pixels_per_0125
    rate = top_grid_value - (steps * 0.125)
    return round(rate / 0.125) * 0.125


def get_n_clusters(sep_date):
    return 5 if sep_date.month in (9, 12) else 4


def process_sep(sep_date, pdf_path):
    print(f"  Processing...")
    n_clusters = get_n_clusters(sep_date)
    starting_year = sep_date.year
    page_index = get_page_index(sep_date)
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            print(
                f"  [WARN]  PDF only has {len(doc)} pages, expected index {page_index}"
            )
            doc.close()
            return None
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))
        pix.save("page.png")
        doc.close()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    img = cv2.imread("page.png")
    if img is None:
        print(f"  [ERROR] Failed to load page.png")
        return None
    crop = get_crop(img, sep_date)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=800,
        maxLineGap=20,
    )
    y_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                y_lines.append(y1)
    unique_y = []
    for y in sorted(y_lines):
        if not unique_y or abs(y - unique_y[-1]) > 10:
            unique_y.append(y)

    top_grid_y, pixels_per_0125, top_grid_value = get_calibration(
        unique_y, sep_date
    )
    if top_grid_y is None:
        print(f"  [WARN]  Calibration failed for {sep_date}")
        return None

    print(
        f"  top_grid_y={top_grid_y}, px_per_0125={pixels_per_0125:.1f}, top_val={top_grid_value}"
    )

    lower_hsv, upper_hsv = get_hsv_range(sep_date)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    centers = detect_dots(mask_clean, sep_date)

    if not centers:
        print(f"  [WARN]  No dots detected for {sep_date}")
        return None

    print(f"  dots detected: {len(centers)}")

    x_values = np.array([d[0] for d in centers]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(x_values)
    x_cluster_centers = list(sorted(kmeans.cluster_centers_.flatten()))

    rows = []
    for cx, cy in centers:
        nearest = min(x_cluster_centers, key=lambda c: abs(cx - c))
        cluster_index = x_cluster_centers.index(nearest)
        year = (
            str(starting_year + cluster_index)
            if cluster_index < n_clusters - 1
            else "Longer Run"
        )
        rows.append(
            {
                "Year": year,
                "Projection": y_to_rate(
                    cy, top_grid_y, pixels_per_0125, top_grid_value
                ),
            }
        )

    df_sorted = pd.DataFrame(rows).sort_values(["Year", "Projection"])
    df_sorted["SEP_Date"] = str(sep_date)
    return df_sorted


def save_to_csv(df_sorted):
    if os.path.exists(CSV_PATH):
        df_sorted.to_csv(CSV_PATH, mode="a", header=False, index=False)
        print(f"  [CSV]    Appended -> {CSV_PATH}")
    else:
        df_sorted.to_csv(CSV_PATH, index=False)
        print(f"  [CSV]    Created  -> {CSV_PATH}")


if __name__ == "__main__":
    today = date.today()
    for sep_date in ALL_DATES:
        print(f"\n-- {sep_date} ------------------------------------------")
        if sep_date > today:
            print(f"  [FUTURE] Not released yet, skipping.")
            continue
        pdf_path = download_sep(sep_date)
        if not pdf_path:
            continue
        df = process_sep(sep_date, pdf_path)
        if df is not None:
            print(df.to_string())
            save_to_csv(df)
    print("\nDone. CSV saved to:", CSV_PATH)