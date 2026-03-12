# ETH Local AI Analyst

本專案是一個 **本地端運行的 ETH 價格分析 AI 系統**，  
透過抓取加密貨幣交易資料、計算技術指標，並使用機器學習模型預測下一根 K 線的市場方向。

此系統完全在本地端運行，不依賴雲端服務。

---

# 專案功能

目前版本包含以下功能：

    1️⃣ 自動抓取 ETH/USDT 歷史 K 線資料  
    2️⃣ 計算常見技術指標

    - RSI
    - EMA
    - MACD
    - Bollinger Bands
    - 報酬率變化
    - 成交量變化

    3️⃣ 使用 **XGBoost 機器學習模型**預測下一根 K 線漲跌  

    4️⃣ 本地端網頁分析介面 (Streamlit)

    顯示內容：

    - 最新 ETH 價格
    - AI 預測方向
    - 上漲機率
    - 最近 100 根 K 線圖
    - 技術指標數據

---

# 專案架構

eth_local_ai/
│
├─ data/
│ ├─ raw/ # 原始市場資料
│ └─ processed/ # AI 訓練資料
│
├─ models/
│ └─ eth_xgb_model.pkl # 訓練好的模型
│
├─ src/
│ ├─ fetch_data.py # 抓取 ETH 市場資料
│ ├─ feature_engineering.py # 計算技術指標
│ ├─ train_model.py # 訓練 AI 模型
│ ├─ predict.py # 預測最新市場方向
│ └─ app.py # Streamlit 分析介面
│
├─ requirements.txt
└─ README.md

---

# 安裝環境

安裝套件：

```bash
pip install -r requirements.txt

或手動安裝：

pip install ccxt pandas numpy matplotlib scikit-learn xgboost ta streamlit joblib

使用流程

請依照以下順序執行：

    1️⃣ 抓取市場資料
    python src/fetch_data.py

    會產生：

    data/raw/eth_1h.csv
    2️⃣ 建立特徵資料
    python src/feature_engineering.py

    會產生：

    data/processed/eth_features.csv
    3️⃣ 訓練 AI 模型
    python src/train_model.py

    會產生：

    models/eth_xgb_model.pkl
    4️⃣ 預測市場方向
    python src/predict.py

    會輸出：

    目前價格
    預測方向
    上漲機率
    5️⃣ 啟動本地分析頁面
    streamlit run src/app.py

    瀏覽器會開啟：

    http://localhost:8501
