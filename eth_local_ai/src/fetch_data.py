import ccxt
import pandas as pd
import os


def fetch_eth_data(symbol='ETH/USDT', timeframe='1h', limit=1000):
    exchange = ccxt.binance()

    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        bars,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    os.makedirs('data/raw', exist_ok=True)
    save_path = 'data/raw/eth_1h.csv'
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"資料抓取成功，共 {len(df)} 筆")
    print(f"檔案已儲存到：{save_path}")
    print(df.tail())


if __name__ == "__main__":
    fetch_eth_data()