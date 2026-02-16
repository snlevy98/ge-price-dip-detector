import requests
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# ------------------ CONFIG ------------------

MAPPING_API = "https://prices.runescape.wiki/api/v1/osrs/mapping"
TIMESERIES_API = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id={}"
AVG_API = "https://prices.runescape.wiki/api/v1/osrs/5m"
LIVE_API = "https://prices.runescape.wiki/api/v1/osrs/latest"

TOP_N_ITEMS = 50
RETRAIN_INTERVAL = 60 * 60  # 1 hour
POLL_INTERVAL = 300  # 5 minute

# --------------------------------------------


def fetch_json(url):
    try:
        headers = {'User-Agent': 'ge_price_dip_detector - @IrreguLars on Discord'}
        return requests.get(url, headers=headers, timeout=10).json()
    except Exception as e:
        print("FAILED:", url, e)
        return None


# ---------- MAPPING FILTER ----------

def mapping_filter(item):
    if not item.get("members", False):
        return False

    limit = item.get("limit")
    if limit is None:
        return False

    # Filter low limits
    if limit < 100:          # <-- realistic threshold
        return False

    value = item.get("value", 0)
    highalch = item.get("highalch", 0)

    if value > 0 and highalch / value > 0.9:
        return False

    return True


# ---------- CLEAN TIMESERIES DATA ----------

def clean_timeseries(ts):
    return [
        p for p in ts
        if p["avgHighPrice"] is not None
        and p["avgLowPrice"] is not None
        and p["highPriceVolume"] is not None
        and p["lowPriceVolume"] is not None
    ]


# ---------- TIMESERIES SCORING ----------

def compute_score(ts):
    highs = np.array([p["avgHighPrice"] for p in ts])
    lows  = np.array([p["avgLowPrice"] for p in ts])
    vols  = np.array([p["highPriceVolume"] for p in ts])

    if np.any(vols == 0):
        return None

    features = [
        (highs[-1] - highs.mean()) / highs.mean(),
        vols[-1] / vols.mean(),
        (highs[-1] - lows[-1]) / highs[-1]
    ]

    if any(np.isnan(features)) or any(np.isinf(features)):
        return None

    return features


# ---------- FEATURE ENGINEERING ----------

def build_features(live, avg, prev_high):
    ret = (live["high"] - prev_high) / prev_high
    spread = (live["high"] - live["low"]) / live["low"]
    volume = avg["highPriceVolume"] + avg["lowPriceVolume"]
    liquidity = np.log10(volume + 1)

    return [ret, spread, volume, liquidity]


# ---------- MAIN ----------

print("Fetching mapping...")
mapping = fetch_json(MAPPING_API)

candidates = [i for i in mapping if mapping_filter(i)]
print(f"Candidates before volume filter: {len(candidates)}")

print("Fetching 5m volume universe...")
avg_snapshot = fetch_json(AVG_API)["data"]

active_ids = {
    int(k)
    for k, v in avg_snapshot.items()
    if (v["highPriceVolume"] + v["lowPriceVolume"]) > 200
}

candidates = [
    i for i in candidates
    if i["id"] in active_ids
]

print(f"Candidates after volume filter: {len(candidates)}")

print(f"Scoring {len(candidates)} items...")
scores = {}
names = {}
debugCnt = 0

for i, item in enumerate(candidates):
    print(f"{i+1}/{len(candidates)} scoring {item['name']}")
    ts_raw = fetch_json(TIMESERIES_API.format(item["id"]))["data"]
    ts = clean_timeseries(ts_raw)

    if debugCnt < 1:
        debugCnt += 1
        print(f"{item['id']}: kept {len(ts)}/{len(ts_raw)} samples")

    if len(ts) < 30:
        continue

    avg_vol = np.mean([p["highPriceVolume"] for p in ts])
    if avg_vol < 50:
        continue

    score = compute_score(ts)            # FIX
    if score is not None:                # FIX
        scores[item["id"]] = score       # FIX
        names[item["id"]] = item["name"] # FIX

print("Valid items:", len(scores))       # FIX

# FIX: rank by first feature (high deviation)
top_items = sorted(
    scores,
    key=lambda k: scores[k][0]
)[:TOP_N_ITEMS]

print("Tracking:", top_items)
for id in names:
    if id in top_items:
        print(f"Tracking {names[id]} (ID: {id})")


# ---------- TRAIN MODELS ----------

models = {}
last_train = time.time()
high_means = {}
last_train_high = {}

print("Training models...")
for item_id in top_items:
    ts_raw = fetch_json(TIMESERIES_API.format(item_id))["data"]
    ts = clean_timeseries(ts_raw)
    X = []

    prev_high = ts[0]["avgHighPrice"]
    last_train_high[item_id] = ts[-1]["avgHighPrice"]

    for p in ts[1:]:
        ret = (p["avgHighPrice"] - prev_high) / prev_high
        prev_high = p["avgHighPrice"]
        spread = (p["avgHighPrice"] - p["avgLowPrice"]) / p["avgLowPrice"]
        volume = p["highPriceVolume"] + p["lowPriceVolume"]

        liquidity = np.log10(volume + 1)

        X.append([ret, spread, volume, liquidity])

    model = IsolationForest(contamination=0.02)
    model.fit(X)
    models[item_id] = model

print("System live.\n")


# ---------- LIVE LOOP ----------

while True:
    try:
        counter = 0
        live_data = fetch_json(LIVE_API)["data"]
        avg_data = fetch_json(AVG_API)["data"]

        for item_id in top_items:
            item_key = str(item_id)
            live_item = live_data.get(item_key)
            avg_item = avg_data.get(item_key)

            if not live_item or not avg_item:
                continue
            counter += 1

            prev_high = last_train_high[item_id]
            feats = build_features(live_item, avg_item, prev_high)
            last_train_high[item_id] = live_item["high"]
            pred = models[item_id].predict([feats])[0]

            if pred == -1:
                # Require micro-reversal
                if live_item["high"] < avg_item["avgHighPrice"] * 0.97:
                    continue

                if live_item["high"] > live_item["low"] * 1.02:
                    continue

                drop_pct = (avg_item["avgHighPrice"] - live_item["high"]) / avg_item["avgHighPrice"]

                if drop_pct < 0.03:
                    continue

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"DIP DETECTED on {names[item_id]}: {live_item}"
                )

        # FIX: retrain with same 4 features
        if time.time() - last_train > RETRAIN_INTERVAL:
            print("Retraining models...")
            for item_id in top_items:
                ts = clean_timeseries(fetch_json(TIMESERIES_API.format(item_id))["data"])
                X = []
                high_mean = high_means[item_id]

                for p in ts:
                    mid = (p["avgHighPrice"] + p["avgLowPrice"]) / 2
                    spread = (p["avgHighPrice"] - p["avgLowPrice"]) / p["avgLowPrice"]
                    volume = p["highPriceVolume"] + p["lowPriceVolume"]
                    high_dev = p["avgHighPrice"] / high_mean
                    high_dev_weighted = high_dev * 3
                    X.append([mid, spread, volume, high_dev_weighted])

                models[item_id].fit(X)

            last_train = time.time()

        print(f"Updated Items: {counter}. Waiting...\n")
        time.sleep(POLL_INTERVAL)

    except Exception as e:
        print("Error:", e)
        time.sleep(10)
