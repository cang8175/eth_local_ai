import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'rsi', 'ema_10', 'ema_20',
    'macd', 'macd_signal',
    'bb_high', 'bb_low', 'bb_mid',
    'return_1', 'return_3', 'vol_change'
]

st.set_page_config(page_title="ETH Local AI Analyst", layout="wide")
st.title("ETH Local AI Analyst")
st.write("本地端 ETH 漲跌分析系統")

try:
    df = pd.read_csv('data/processed/eth_features.csv')
    model = joblib.load('models/eth_xgb_model.pkl')

    latest = df.iloc[-1:][FEATURE_COLS]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    current_price = df.iloc[-1]['close']
    up_prob = prob[1]
    down_prob = prob[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("最新價格", f"{current_price:.2f}")
    col2.metric("預測方向", "上漲" if pred == 1 else "下跌")
    col3.metric("上漲機率", f"{up_prob:.2%}")

    st.subheader("最近 100 根收盤價")
    chart_df = df.tail(100)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(chart_df['close'])
    ax.set_title("ETH Close Price")
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.subheader("最新技術指標")
    latest_row = df.iloc[-1]

    st.write(f"RSI: {latest_row['rsi']:.2f}")
    st.write(f"MACD: {latest_row['macd']:.4f}")
    st.write(f"MACD Signal: {latest_row['macd_signal']:.4f}")
    st.write(f"EMA 10: {latest_row['ema_10']:.2f}")
    st.write(f"EMA 20: {latest_row['ema_20']:.2f}")

    st.subheader("最近資料表")
    st.dataframe(df.tail(20))

except Exception as e:
    st.error(f"發生錯誤：{e}")
    st.info("請先依序執行 fetch_data.py → feature_engineering.py → train_model.py")