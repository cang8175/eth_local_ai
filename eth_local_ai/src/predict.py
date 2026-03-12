import pandas as pd
import joblib


FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'ema_10', 'ema_20',
    'macd', 'macd_signal',
    'bb_high', 'bb_low', 'bb_mid',
    'return_1', 'return_3', 'vol_change'
]


def predict_latest(data_path='data/processed/eth_features.csv', model_path='models/eth_xgb_model.pkl'):
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    latest = df.iloc[-1:][FEATURE_COLS]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    up_prob = prob[1]
    down_prob = prob[0]

    print("=== 最新預測結果 ===")
    print(f"目前價格: {df.iloc[-1]['close']}")
    print(f"預測下一根方向: {'上漲' if pred == 1 else '下跌'}")
    print(f"上漲機率: {up_prob:.4f}")
    print(f"下跌機率: {down_prob:.4f}")


if __name__ == "__main__":
    predict_latest()