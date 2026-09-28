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
    (2012, 1): 6.0, (2012, 4): 6.0, (2012, 6): 6.0, (2012, 9): 6.0, (2012, 12): 6.0,
    (2013, 3): 5.0, (2013, 6): 5.0, (2013, 9): 5.0, (2013, 12): 5.0,
    (2014, 3): 5.0, (2014, 6): 5.0, (2014, 9): 5.0, (2014, 12): 5.0,
}

ALL_DATES = [
    # 2012
    date(2012, 1, 25), date(2012, 4, 25), date(2012, 6, 20), date(2012, 9, 13), date(2012, 12, 12),
    # 2013
    date(2013, 3, 20), date(2013, 6, 19), date(2013, 9, 18), date(2013, 12, 18),
    # 2014
    date(2014, 3, 19), date(2014, 6, 18), date(2014, 9, 17), date(2014, 12, 17),
]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_page_index(sep_date):
    """Different page indices for different periods."""
    if sep_date.year >= 2021:
        return 3  # Page 4 (index 3)
    if sep_date.year == 2020 and sep_date.month == 12:
        return 3  # Page 4 (index 3)
    if sep_date == date(2012, 1, 25):
        return 2  # Page 3 (index 2)
    if sep_date == date(2012, 4, 25):
        return 2  # Page 3 (index 2)
    return 2  # Default: Page 3 (index 2)


def get_crop(img, sep_date):
    """
    2012-2014: dot plot is in bottom half of page, under a bar chart.
    2015+: dot plot is standalone, use fixed top crop.
    """
    h, w = img.shape[:2]
    left = 200
    right = min(2200, w)
    if sep_date.year <= 2014:
        top = int(h * 0.45)
        bottom = int(h * 0.82)  # ← Stop before the text underneath
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
    if sep_date.year >= 2025:
        return np.ones((3, 3), np.uint8), 1
    elif sep_date.year <= 2014:
        return np.ones((3, 3), np.uint8), 1  # ← Gentler for packed dots
    else:
        return np.ones((5, 5), np.uint8), 2

def get_hough_length(sep_date):
    if sep_date.year <= 2014 and sep_date < date(2014, 9, 1):
        return 500
    elif date(2014, 9, 1) <= sep_date <= date(2014, 12, 31):
        return 300  # Lower for dotted gridlines in late 2014
    else:
        return 800
        
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
    # For 2012-2014, use distance transform to find individual dots even when touching
    if sep_date.year <= 2014:
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Find local maxima (peaks = individual dots)
        _, maxima = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        maxima = np.uint8(maxima)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            maxima, connectivity=8
        )
    else:
        # 2015+ use regular erosion
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
    
    # 2012-2014 (before Sept) have 1% gridline spacing; late 2014+ have 0.25% spacing
    if sep_date.year <= 2014 and sep_date < date(2014, 9, 1):
        # Gridlines are 1% apart = 8 x 0.125%
        pixels_per_0125 = median_gap / 8
    else:
        # Gridlines are 0.25% apart = 2 x 0.125%
        pixels_per_0125 = median_gap / 2
    
    # For early 2012-2014 (before Sept 2014), assume topmost line is missing and extrapolate upward
    # For late 2014+ (Sept onwards), use detected top line like 2015+
    if sep_date.year <= 2014 and sep_date < date(2014, 9, 1):
        top_grid_y = unique_y[0] - median_gap
        print(f"  [CALIB] Extrapolated top line (missing): {top_grid_y}")
    else:
        # For 2014-09+ and 2015+, use detected top line
        top_grid_y = None
        for i in range(len(unique_y) - 1):
            if abs((unique_y[i + 1] - unique_y[i]) - median_gap) < 10:
                top_grid_y = unique_y[i]
                break
        if top_grid_y is None:
            return None, None, None
    
    top_grid_value = Y_AXIS_MAX.get((sep_date.year, sep_date.month))
    if top_grid_value is None:
        print(f"  [WARN]  No Y_AXIS_MAX entry for {sep_date}")
        return None, None, None
    return top_grid_y, pixels_per_0125, top_grid_value

def y_to_rate(y, top_grid_y, pixels_per_0125, top_grid_value, sep_date=None):
    if sep_date and sep_date.year <= 2014 and sep_date < date(2014, 9, 1):
        # Early 2012-2014 (before Sept) use 0.25 increments
        increment = 0.25
        pixels_per_increment = pixels_per_0125 * 2  # pixels_per_0125 was for 0.125, double it for 0.25
    else:
        # Late 2014+ and 2015+ use 0.125 increments
        increment = 0.125
        pixels_per_increment = pixels_per_0125
    
    steps = (y - top_grid_y) / pixels_per_increment
    rate = top_grid_value - (steps * increment)
    return round(rate / increment) * increment

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
    print("Original image:", img.shape)
    print("Crop:", crop.shape)
    cv2.imwrite(f"debug_crop_{sep_date}.png", crop)

    # Only use the graph area for Hough line detection
    graph_top = int(crop.shape[0] * 0.22)
    graph_bottom = int(crop.shape[0] * 0.78)
    
    graph = crop[graph_top:graph_bottom, :]
    
    gray = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)  # Lower thresholds to catch fainter dotted lines

    # Lower threshold for late 2014+ to catch fainter dotted gridlines
    hough_threshold = 50 if date(2014, 9, 1) <= sep_date <= date(2014, 12, 31) else 100
    hough_gap = 50 if date(2014, 9, 1) <= sep_date <= date(2014, 12, 31) else 20
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=get_hough_length(sep_date),
        maxLineGap=hough_gap,
    )
    # Debug: save edge detection
    cv2.imwrite(f"debug_edges_{sep_date}.png", edges)
    
    # Debug: print all detected lines
    if lines is not None:
        print(f"  [HOUGH] Found {len(lines)} lines total")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            print(f"    Line {i}: y1={y1}, y2={y2}, length={abs(x2-x1)}, dy={abs(y2-y1)}")
    y_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
        
            if abs(y2 - y1) < 5:
                # Late 2014+ dotted lines are shorter, lower threshold
                min_span = 0.3 if date(2014, 9, 1) <= sep_date <= date(2014, 12, 31) else 0.5
                if abs(x2 - x1) > (graph.shape[1] * min_span):
                    # Convert from graph coordinates back to crop coordinates
                    y_lines.append(y1 + graph_top)
    unique_y = []
    
    for y in sorted(y_lines):
    
        if not unique_y or abs(y - unique_y[-1]) > 15:
            unique_y.append(y)
    
    
    # Ignore everything near the top of the crop, but not for late 2014+ which have the top line visible
    if sep_date.year <= 2014 and sep_date < date(2014, 9, 1):
        unique_y = [y for y in unique_y if y > crop.shape[0] * 0.15]
    print(unique_y)

    top_grid_y, pixels_per_0125, top_grid_value = get_calibration(
        unique_y, sep_date
    )
    
    # Special case: for late 2014+ (0.25% gridlines), calculate how far below Y_AXIS_MAX the first detected line is
    if date(2014, 9, 1) <= sep_date <= date(2014, 12, 17) and len(unique_y) >= 1:
        gaps = [unique_y[i + 1] - unique_y[i] for i in range(len(unique_y) - 1)]
        median_gap = np.median(gaps)
        pixels_per_0125 = median_gap / 2
        # First detected line is at 3.5%, actual top is at 5.0% = 6 gridlines above
        # Extrapolate up to the true top
        top_grid_y = unique_y[0] - (median_gap * 6)
        print(f"  [SPECIAL] Extrapolating from 3.5% to 5% for {sep_date}")
    # Early 2013-2014 (1% gridlines)
    elif date(2013, 3, 20) <= sep_date <= date(2014, 6, 18) and len(unique_y) >= 1:
        top_grid_y = unique_y[0]
        gaps = [unique_y[i + 1] - unique_y[i] for i in range(len(unique_y) - 1)]
        median_gap = np.median(gaps)
        pixels_per_0125 = median_gap / 8
        print(f"  [SPECIAL] Using detected top line for {sep_date}")
    
    # Fallback: if we couldn't detect the top gridline, estimate from remaining lines
    if top_grid_y is None and len(unique_y) >= 2:
        # Use spacing from detected lines to estimate calibration
        gaps = [unique_y[i + 1] - unique_y[i] for i in range(len(unique_y) - 1)]
        median_gap = np.median(gaps)
        pixels_per_0125 = median_gap / 2
        
        # Estimate where top line should be (one gap above highest detected line)
        top_grid_y = unique_y[0] - median_gap
        top_grid_value = Y_AXIS_MAX.get((sep_date.year, sep_date.month))
        
        print(f"  [FALLBACK] Estimated top_grid_y={top_grid_y}, using line spacing")
    
    if top_grid_y is None:
        print(f"  [WARN]  Calibration failed for {sep_date}")
        return None

    print(
        f"  top_grid_y={top_grid_y}, px_per_0125={pixels_per_0125:.1f}, top_val={top_grid_value}"
    )

    lower_hsv, upper_hsv = get_hsv_range(sep_date)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # DEBUG: Save raw mask
    cv2.imwrite(f"debug_mask_raw_{sep_date}.png", mask)
    
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    centers = detect_dots(mask_clean, sep_date)

    if not centers:
        print(f"  [WARN]  No dots detected for {sep_date}")
        return None

    print(f"  dots detected: {len(centers)}")

    # ── DEBUG VISUALIZATION ───────────────────────────────
    debug = crop.copy()
    
    # Draw horizontal grid lines
    for y in unique_y:
        cv2.line(
            debug,
            (0, y),
            (debug.shape[1], y),
            (0, 0, 255),
            2
        )

    # Draw grid line info on image
    for i, y in enumerate(unique_y):
        cv2.putText(debug, str(i), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw detected dots
    for x, y in centers:
        cv2.circle(
            debug,
            (int(x), int(y)),
            8,
            (0, 255, 0),
            2
        )
    
    # Save debug image
    debug_path = f"debug_detection_{sep_date}.png"
    cv2.imwrite(debug_path, debug)
    
    print(f"  Debug image saved: {debug_path}")

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
        projection = y_to_rate(
            cy, top_grid_y, pixels_per_0125, top_grid_value, sep_date
        )
        
        # Fix offset for early 2012
        if sep_date in (date(2012, 1, 25), date(2012, 4, 25)):
            projection -= 1.0
        
        # Filter only obviously wrong detections (outside realistic bounds)
        if projection < -1.0 or projection > 7.0:
            continue
        
        rows.append(
            {
                "Year": year,
                "Projection": projection,
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