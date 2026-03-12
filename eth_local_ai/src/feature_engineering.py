import pandas as pd
import ta
import os


def build_features(input_path='data/raw/eth_1h.csv', output_path='data/processed/eth_features.csv'):
    df = pd.read_csv(input_path)

    # 技術指標
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    df['ema_10'] = ta.trend.EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()

    macd_obj = ta.trend.MACD(close=df['close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()

    bb_obj = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb_obj.bollinger_hband()
    df['bb_low'] = bb_obj.bollinger_lband()
    df['bb_mid'] = bb_obj.bollinger_mavg()

    # 報酬率 / 變化率
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['vol_change'] = df['volume'].pct_change(1)

    # 下一根K棒是否上漲
    df['future_close'] = df['close'].shift(-1)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # 去除空值
    df = df.dropna().reset_index(drop=True)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"特徵資料建立完成，共 {len(df)} 筆")
    print(f"檔案已儲存到：{output_path}")
    print(df.tail())


if __name__ == "__main__":
    build_features()