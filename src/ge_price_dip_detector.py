from xml.parsers.expat import model
import requests
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ------------------ CONFIG ------------------

MAPPING_API = "https://prices.runescape.wiki/api/v1/osrs/mapping"
TIMESERIES_API = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id={}"
AVG_API = "https://prices.runescape.wiki/api/v1/osrs/5m"
LIVE_API = "https://prices.runescape.wiki/api/v1/osrs/latest"

TOP_N_ITEMS = 50
VOLUME_THRESHOLD = 200
DROP_THRESHOLD = 0.01
RETRAIN_INTERVAL = 60 * 60  # 1 hour
POLL_INTERVAL = 300  # 5 minute
BUILD_UNIVERSE = False # Set to True to build universe from mapping, False to use hardcoded list

# Item names and IDs to track

ITEM_IDS = [
    566,
    21820,
    565,
    12934,
    2,
    28924,
    892,
    6332,
    31111,
    810,
    21880,
    25853,
    823,
    567,
    1777,
    66,
    32907,
    19580,
    3142,
    9242,
    383,
    8778,
    9075,
    811,
    19669,
    21316,
    24607,
    9144,
    21326,
    28991,
    239,
    28878,
    21622,
    19582,
    3144,
    9338,
    9340,
    31914,
    9143,
    9341,
    28157,
    385,
    4697,
    203,
    4822,
    9243,
    24595,
    21350,
    231,
    4694,
    395,
    4823,
    821,
    4740,
    9244,
    9381,
    8007,
    1775,
    7944,
    25849,
    11875,
    30900,
    9192,
    861,
    9191,
    3138,
    9339,
    31910,
    241,
    1625,
    868,
    10810,
    30843,
    7946,
    8013,
    8010,
    19484,
    575,
    8008,
    21177,
    10937,
    6333,
    199,
    869,
    9380,
    235,
    22593,
    253,
    10033,
    29140
]

ITEM_NAMES = {
    566: "Soul rune",
    21820: "Revenant ether",
    565: "Blood rune",
    12934: "Zulrah's scales",
    2: "Steel cannonball",
    28924: "Sunfire splinters",
    892: "Rune arrow",
    6332: "Mahogany logs",
    31111: "Demon tear",
    810: "Adamant dart",
    21880: "Wrath rune",
    25853: "Amethyst dart tip",
    823: "Adamant dart tip",
    567: "Unpowered orb",
    1777: "Bow string",
    66: "Yew longbow (u)",
    32907: "Ironwood logs",
    19580: "Rune javelin tips",
    3142: "Raw karambwan",
    9242: "Ruby bolts (e)",
    383: "Raw shark",
    8778: "Oak plank",
    9075: "Astral rune",
    811: "Rune dart",
    19669: "Redwood logs",
    21316: "Amethyst broad bolts",
    24607: "Blighted ancient ice sack",
    9144: "Runite bolts",
    21326: "Amethyst arrow",
    28991: "Atlatl dart",
    239: "White berries",
    28878: "Moonlight antler bolts",
    21622: "Volcanic ash",
    19582: "Dragon javelin tips",
    3144: "Cooked karambwan",
    9338: "Emerald bolts",
    9340: "Diamond bolts",
    31914: "Rune cannonball",
    9143: "Adamant bolts",
    9341: "Dragonstone bolts",
    28157: "Forester's ration",
    385: "Shark",
    4697: "Smoke rune",
    203: "Grimy tarromin",
    4822: "Mithril nails",
    9243: "Diamond bolts (e)",
    24595: "Blighted karambwan",
    21350: "Amethyst arrowtips",
    231: "Snape grass",
    4694: "Steam rune",
    395: "Raw sea turtle",
    4823: "Adamantite nails",
    821: "Steel dart tip",
    4740: "Bolt rack",
    9244: "Dragonstone bolts (e)",
    9381: "Runite bolts (unf)",
    8007: "Varrock teleport (tablet)",
    1775: "Molten glass",
    7944: "Raw monkfish",
    25849: "Amethyst dart",
    11875: "Broad bolts",
    30900: "Shark lure",
    9192: "Diamond bolt tips",
    861: "Magic shortbow",
    9191: "Ruby bolt tips",
    3138: "Potato cactus",
    9339: "Ruby bolts",
    31910: "Mithril cannonball",
    241: "Dragon scale dust",
    1625: "Uncut opal",
    868: "Rune knife",
    10810: "Arctic pine logs",
    30843: "Aether rune",
    7946: "Monkfish",
    8013: "Teleport to house (tablet)",
    8010: "Camelot teleport (tablet)",
    19484: "Dragon javelin",
    575: "Earth orb",
    8008: "Lumbridge teleport (tablet)",
    21177: "Expeditious bracelet",
    10937: "Nail beast nails",
    6333: "Teak logs",
    199: "Grimy guam leaf",
    869: "Black knife",
    9380: "Adamant bolts(unf)",
    235: "Unicorn horn dust",
    22593: "Te salt",
    253: "Tarromin",
    10033: "Chinchompa",
    29140: "Cooked sunlight antelope"
}

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
    if prev_high is None:
        return None

    if live["high"] is None or live["low"] is None:
        return None

    if avg["highPriceVolume"] is None or avg["lowPriceVolume"] is None:
        return None
    
    ret = (live["high"] - prev_high) / prev_high
    spread = (live["high"] - live["low"]) / live["low"]
    volume = avg["highPriceVolume"] + avg["lowPriceVolume"]
    liquidity = np.log10(volume + 1)

    return [ret, spread, volume, liquidity]


# ---------- MAIN ----------

def main():
    if BUILD_UNIVERSE:
        print("Fetching mapping...")
        mapping = fetch_json(MAPPING_API)

        candidates = [i for i in mapping if mapping_filter(i)]
        print(f"Candidates before volume filter: {len(candidates)}")

        print("Fetching 5m volume universe...")
        avg_snapshot = fetch_json(AVG_API)["data"]

        active_ids = {
            int(k)
            for k, v in avg_snapshot.items()
            if (v["highPriceVolume"] + v["lowPriceVolume"]) > VOLUME_THRESHOLD
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

        names = {
            item_id: names[item_id]
            for item_id in top_items
        }
    else:
        top_items = ITEM_IDS
        names = ITEM_NAMES

    print("Tracking:", top_items)
    print("Names:", names)
    for id in names:
        if id in top_items:
            print(f"Tracking {names[id]} (ID: {id})")


    # ---------- TRAIN MODELS ----------

    models = {}
    last_train = time.time()
    last_train_high = {}
    scalers = {}

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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=200
        )
        model.fit(X_scaled)
        models[item_id] = model
        scalers[item_id] = scaler

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

                if live_item["high"] is None or live_item["low"] is None:
                    continue

                if avg_item["avgHighPrice"] is None or avg_item["avgLowPrice"] is None:
                    continue

                counter += 1

                prev_high = last_train_high[item_id]
                feats = build_features(live_item, avg_item, prev_high)
                last_train_high[item_id] = live_item["high"]

                scaler = scalers[item_id]
                feats_scaled = scaler.transform([feats])
                pred = models[item_id].predict(feats_scaled)[0]

                if feats is None:
                    continue

                if pred == -1:
                    # Require micro-reversal
                    if live_item["high"] < avg_item["avgHighPrice"] * 0.97:
                        continue

                    if live_item["high"] > live_item["low"] * 1.02:
                        continue

                    drop_pct = (avg_item["avgHighPrice"] - live_item["high"]) / avg_item["avgHighPrice"]

                    if drop_pct < DROP_THRESHOLD:
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
                    prev_high = ts[0]["avgHighPrice"]
                    for p in ts[1:]:
                        ret = (p["avgHighPrice"] - prev_high) / prev_high
                        prev_high = p["avgHighPrice"]
                        spread = (p["avgHighPrice"] - p["avgLowPrice"]) / p["avgLowPrice"]
                        volume = p["highPriceVolume"] + p["lowPriceVolume"]
                        liquidity = np.log10(volume + 1)
                        X.append([ret, spread, volume, liquidity])

                    models[item_id].fit(X)

                last_train = time.time()

            print(f"Updated Items: {counter}. Waiting...\n")
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("Error:", e)
            time.sleep(10)

if __name__ == "__main__":
    main()