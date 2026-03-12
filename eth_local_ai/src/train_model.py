import pandas as pd
import os
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(input_path='data/processed/eth_features.csv', model_path='models/eth_xgb_model.pkl'):
    df = pd.read_csv(input_path)

    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'ema_10', 'ema_20',
        'macd', 'macd_signal',
        'bb_high', 'bb_low', 'bb_mid',
        'return_1', 'return_3', 'vol_change'
    ]

    X = df[feature_cols]
    y = df['target']

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"模型準確率 Accuracy: {acc:.4f}")
    print("\n分類報告：")
    print(classification_report(y_test, preds))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)

    print(f"模型已儲存到：{model_path}")


if __name__ == "__main__":
    train_model()