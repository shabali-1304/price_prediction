"""Educational next-day closing-price regression example; not financial advice."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["Close", "Volume", "Return_1d", "MA_7", "MA_21", "Volatility_7"]

def args():
    p = argparse.ArgumentParser(description="Predict the next day's closing price.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path, help="CSV containing Date and Close")
    g.add_argument("--download", metavar="TICKER", help="Ticker for optional yfinance download")
    p.add_argument("--period", default="5y"); p.add_argument("--output-dir", type=Path, default=Path("outputs")); return p.parse_args()

def load(path):
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), None)
    if date_col is None or "Close" not in df.columns: raise ValueError("CSV must contain a Date (or Datetime) column and a Close column.")
    df = df.rename(columns={date_col: "Date"}); df["Date"] = pd.to_datetime(df["Date"], errors="coerce"); df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" not in df: df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)

def features(df):
    out = df.copy(); out["Return_1d"] = out.Close.pct_change(); out["MA_7"] = out.Close.rolling(7).mean(); out["MA_21"] = out.Close.rolling(21).mean(); out["Volatility_7"] = out.Return_1d.rolling(7).std(); out["Target_Next_Close"] = out.Close.shift(-1); return out

def score(y, pred):
    mse = mean_squared_error(y, pred); return {"MAE": float(mean_absolute_error(y, pred)), "MSE": float(mse), "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y, pred))}

def chart(df, pred, out):
    plt.figure(figsize=(11,5)); plt.plot(df.Date, df.Close, label="Close"); plt.plot(df.Date, df.MA_7, label="7-day MA"); plt.plot(df.Date, df.MA_21, label="21-day MA"); plt.title("Price trend and moving averages"); plt.legend(); plt.tight_layout(); plt.savefig(out/"price_trend.png", dpi=150); plt.close()
    plt.figure(figsize=(11,4)); plt.plot(df.Date, df.Return_1d*100); plt.axhline(0, color="black", lw=.8); plt.title("Daily returns (%)"); plt.tight_layout(); plt.savefig(out/"returns.png", dpi=150); plt.close()
    plt.figure(figsize=(11,5)); plt.plot(pred.Date, pred.Actual, label="Actual")
    for col in pred.columns:
        if col.startswith("Predicted_"): plt.plot(pred.Date, pred[col], label=col.replace("Predicted_", ""))
    plt.title("Actual vs predicted next-day close (test set)"); plt.legend(); plt.tight_layout(); plt.savefig(out/"actual_vs_predicted.png", dpi=150); plt.close()

def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.download:
        try: import yfinance as yf
        except ImportError as exc: raise SystemExit("Install download support: pip install yfinance") from exc
        data = yf.download(a.download, period=a.period, auto_adjust=False, progress=False)
        if data.empty: raise SystemExit("No data returned for that ticker.")
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        path = Path("data") / f"{a.download.replace('/', '_')}.csv"; path.parent.mkdir(exist_ok=True); data.reset_index().to_csv(path, index=False)
    else: path = a.data
    raw = load(path); info = {"rows_before_cleaning": len(raw), "missing_values": {k:int(v) for k,v in raw.isna().sum().items() if v}, "duplicate_rows": int(raw.duplicated().sum()), "duplicate_dates": int(raw.Date.duplicated().sum())}
    raw = raw.drop_duplicates().drop_duplicates(subset=["Date"]).dropna(subset=["Date", "Close"]).reset_index(drop=True)
    df = features(raw).dropna(subset=FEATURES+["Target_Next_Close"]).reset_index(drop=True); n=len(df); first=int(n*.6); second=int(n*.8)
    if first < 10 or n-second < 5: raise ValueError("Need at least 25 usable rows for a 60/20/20 split.")
    train, valid, test = df.iloc[:first], df.iloc[first:second], df.iloc[second:]
    models = {"LinearRegression": LinearRegression(), "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42), "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)}
    valid_scores = {"NaiveBaseline": score(valid.Target_Next_Close, valid.Close)}
    for name, model in models.items(): model.fit(train[FEATURES], train.Target_Next_Close); valid_scores[name] = score(valid.Target_Next_Close, model.predict(valid[FEATURES]))
    best = min(valid_scores, key=lambda k: valid_scores[k]["RMSE"]); combined = pd.concat([train, valid])
    if best != "NaiveBaseline":
        chosen = models[best].fit(combined[FEATURES], combined.Target_Next_Close); joblib.dump({"model":chosen, "features":FEATURES, "target":"Target_Next_Close"}, a.output_dir/"best_model.joblib")
    predictions = {"Date":test.Date, "Actual":test.Target_Next_Close, "Predicted_NaiveBaseline":test.Close}; test_scores={"NaiveBaseline":score(test.Target_Next_Close,test.Close)}
    for name, model in models.items(): predictions[f"Predicted_{name}"]=model.predict(test[FEATURES]); test_scores[name]=score(test.Target_Next_Close,predictions[f"Predicted_{name}"])
    pred=pd.DataFrame(predictions); pred.to_csv(a.output_dir/"test_predictions.csv", index=False); chart(df,pred,a.output_dir)
    info.update({"rows_after_features":len(df),"split_sizes":{"train":len(train),"validation":len(valid),"test":len(test)},"validation_metrics":valid_scores,"test_metrics":test_scores,"selected_model":best})
    (a.output_dir/"metrics.json").write_text(json.dumps(info,indent=2),encoding="utf-8"); print(json.dumps({"selected_model":best,"test_metrics":test_scores[best],"output_dir":str(a.output_dir)},indent=2))

if __name__ == "__main__": main()
