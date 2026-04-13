import requests
import time
import numpy as np
from datetime import datetime

# ---------------- CONFIG ----------------

ITEM_IDS = [5323, 5100, 6032, 6034, 31235, 3042, 401, 4798, 1823, 5305, 13391, 4696, 6289, 10327, 5504, 6018, 21490, 243, 4698, 7060, 4699, 45, 21483, 9241, 9144, 9339, 22593, 11230, 11940, 24592, 1941, 2505, 10810, 31710, 29143, 4822, 6333, 21555, 7944, 868, 22124, 6332, 1783, 1775, 3325, 29311, 22879, 203, 31906, 12409]
NAMES = {5323: 'Strawberry seed', 5100: 'Limpwurt seed', 6032: 'Compost', 6034: 'Supercompost', 31235: 'Gryphon feather', 3042: 'Magic potion(3)', 401: 'Seaweed', 4798: 'Adamant brutal', 1823: 'Waterskin(4)', 5305: 'Barley seed', 13391: 'Lizardman fang', 4696: 'Dust rune', 6289: 'Snakeskin', 10327: 'White firelighter', 5504: 'Strawberry', 6018: 'Poison ivy berries', 21490: 'Seaweed spore', 243: 'Blue dragon scale', 4698: 'Mud rune', 7060: 'Tuna potato', 4699: 'Lava rune', 45: 'Opal bolt tips', 21483: 'Ultracompost', 9241: 'Emerald bolts (e)', 9144: 'Runite bolts', 9339: 'Ruby bolts', 22593: 'Te salt', 11230: 'Dragon dart', 11940: 'Dark fishing bait', 24592: 'Blighted anglerfish', 1941: 'Swamp paste', 2505: 'Blue dragon leather', 10810: 'Arctic pine logs', 31710: 'Rainbow crab paste', 29143: 'Cooked moonlight antelope', 4822: 'Mithril nails', 6333: 'Teak logs', 21555: 'Numulite', 7944: 'Raw monkfish', 868: 'Rune knife', 22124: 'Superior dragon bones', 6332: 'Mahogany logs', 1783: 'Bucket of sand', 1775: 'Molten glass', 3325: 'Vampyre dust', 29311: 'Hunter spear tips', 22879: 'Snape grass seed', 203: 'Grimy tarromin', 31906: 'Bronze cannonball', 12409: 'Tai bwo wannai teleport'}

TIMESERIES_API = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id={}"
AVG_API = "https://prices.runescape.wiki/api/v1/osrs/5m"

POLL_INTERVAL = 300  # 5 minutes
Z_THRESHOLD = 2.0    # 2 std dev dip

# ----------------------------------------


def fetch_json(url):
    headers = {'User-Agent': 'baseline_zscore_detector'}
    return requests.get(url, headers=headers, timeout=10).json()


def clean_timeseries(ts):
    return [
        p for p in ts
        if p["avgHighPrice"] is not None
    ]


# ---------- TRAIN BASELINES ----------

print("Training baselines...\n")

baselines = {}

for item_id in ITEM_IDS:
    ts = fetch_json(TIMESERIES_API.format(item_id))["data"]
    ts = clean_timeseries(ts)

    highs = np.array([p["avgHighPrice"] for p in ts])

    mean = highs.mean()
    std = highs.std()

    baselines[item_id] = {
        "mean": mean,
        "std": std
    }

    print(f"{NAMES[item_id]} → mean={int(mean)} std={int(std)}")

print("\nBaseline system live.\n")


# ---------- LIVE LOOP ----------

while True:
    try:
        avg_data = fetch_json(AVG_API)["data"]

        for item_id in ITEM_IDS:
            item_key = str(item_id)
            item = avg_data.get(item_key)

            if not item or item["avgHighPrice"] is None:
                continue

            mean = baselines[item_id]["mean"]
            std = baselines[item_id]["std"]

            current = item["avgHighPrice"]
            z = (current - mean) / std

            if z < -Z_THRESHOLD:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"BASELINE DIP on {NAMES[item_id]}: "
                    f"price={current}, z={round(z,2)}"
                )

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        print("Error:", e)
        time.sleep(10)
