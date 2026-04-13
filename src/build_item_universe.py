import requests
import numpy as np

MAPPING_API = "https://prices.runescape.wiki/api/v1/osrs/mapping"
AVG_API = "https://prices.runescape.wiki/api/v1/osrs/5m"

# ----------- FILTER CONFIG -----------

MIN_VOLUME = 1000        # per 5 minutes
MIN_PRICE = 30
MAX_PRICE = 1000
MIN_BUY_LIMIT = 5000
TOP_N = 100               # how many items to keep

# -----------------------------------


def fetch_json(url):
    headers = {'User-Agent': 'osrs-universe-builder'}
    return requests.get(url, headers=headers, timeout=10).json()


print("Fetching mapping...")
mapping = fetch_json(MAPPING_API)

print("Fetching 5m volume snapshot...")
avg_data = fetch_json(AVG_API)["data"]

candidates = []

for item in mapping:
    if not item.get("members", False):
        continue

    limit = item.get("limit")
    if not limit or limit < MIN_BUY_LIMIT:
        continue

    item_id = item["id"]
    key = str(item_id)

    if key not in avg_data:
        continue

    avg = avg_data[key]
    volume = avg["highPriceVolume"] + avg["lowPriceVolume"]

    if volume < MIN_VOLUME:
        continue

    price = avg["avgHighPrice"]
    if price is None or not (MIN_PRICE <= price <= MAX_PRICE):
        continue

    candidates.append({
        "id": item_id,
        "name": item["name"],
        "price": price,
        "volume": volume,
        "limit": limit
    })

# sort by volume descending
candidates.sort(key=lambda x: x["volume"], reverse=True)

if (len(candidates) < TOP_N):
    top = candidates
else:
    top = candidates[:TOP_N]

print("\n===== GENERATED UNIVERSE =====\n")
print(f"Total candidates: {len(candidates)}\n")
print("ITEM_IDS = [")
for i in top:
    print(f"    {i['id']},")
print("]\n")

print("ITEM_NAMES = {")
for i in top:
    print(f"    {i['id']}: \"{i['name']}\",")
print("}\n")

print("Top items summary:\n")
for i in top:
    print(
        f"{i['name']:<30} "
        f"Price: {i['price']:<6} "
        f"Vol/5m: {i['volume']:<6} "
        f"Limit: {i['limit']}"
    )
