# OSRS GE Price Dip Detector

A quantitative trading bot for Old School RuneScape that detects temporary price dips using market microstructure and anomaly detection.

## Features
- Filters tradable items by real liquidity
- Uses return-based features
- Mean reversion detection
- Isolation Forest anomaly model

## Setup

```bash
git clone https://github.com/snlevy/ge-price-dip-detector
cd ge-price-dip-detector
pip install -r requirements.txt
python src/detector.py
