import requests
import time
import numpy as np
from collections import deque
from datetime import datetime

LIVE_API = "https://prices.runescape.wiki/api/v1/osrs/latest"

# -------- CONFIG --------

POLL_INTERVAL = 10
WINDOW_SIZE = 50
STD_THRESHOLD = 2.5        # how many sigmas below mean = dip
MIN_PROFIT = 20           # after tax
TAX = 0.02


# Use same IDs as Isolation Forest
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

# ---------------------------------------


def fetch_live():
    headers = {'User-Agent': 'osrs-std-baseline'}
    return requests.get(LIVE_API, headers=headers, timeout=10).json()["data"]


# rolling price buffers
buffers = {
    item_id: deque(maxlen=WINDOW_SIZE)
    for item_id in ITEM_IDS
}

print("Collecting initial data...\n")

# ---------- WARMUP ----------
while True:
    live = fetch_live()
    filled = 0

    for item_id in ITEM_IDS:
        key = str(item_id)
        if key not in live:
            continue

        price = live[key]["high"]
        if price is not None:
            buffers[item_id].append(price)

        if len(buffers[item_id]) == WINDOW_SIZE:
            filled += 1

    print(f"Warming: {filled}/{len(ITEM_IDS)} ready")

    if filled == len(ITEM_IDS):
        break

    time.sleep(POLL_INTERVAL)


print("\nSystem live.\n")

# ---------- LIVE LOOP ----------
while True:
    try:
        live = fetch_live()

        for item_id in ITEM_IDS:
            key = str(item_id)
            if key not in live:
                continue

            price = live[key]["high"]
            if price is None:
                continue

            buf = buffers[item_id]
            buf.append(price)

            mean = np.mean(buf)
            std = np.std(buf)

            if std == 0:
                continue

            z = (price - mean) / std

            if z < -STD_THRESHOLD:
                buy = price
                sell = mean + (std * 0.8)  # target a sell price slightly above the threshold

                profit = (sell * (1 - TAX)) - buy

                if profit < MIN_PROFIT:
                    continue

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"DIP {ITEM_NAMES[item_id]} | "
                    f"Buy: {buy} | "
                    f"Sell: {int(sell)} | "
                    f"Mean: {int(mean)} | "
                    f"Std: {int(std)} | "
                    f"Profit/item: {int(profit)}"
                )

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)
