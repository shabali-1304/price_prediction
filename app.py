"""MCA-style Streamlit dashboard for market trend analysis and regression."""
from pathlib import Path
import io, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="MarketLens | Forecasting Lab", page_icon="📊", layout="wide")
ROOT = Path(__file__).parent
FEATURES = ["Close", "Volume", "Return_1d", "MA_7", "MA_21", "Volatility_7", "RSI_14", "MACD", "BB_Position"]

def metric_row(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse, "RMSE": np.sqrt(mse), "R²": r2_score(actual, predicted)}

def load_csv(data):
    df = pd.read_csv(data); date = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), None)
    if date is None or "Close" not in df: raise ValueError("Data must contain Date and Close columns.")
    df = df.rename(columns={date:"Date"}); df.Date = pd.to_datetime(df.Date, errors="coerce"); df.Close = pd.to_numeric(df.Close, errors="coerce")
    if "Volume" not in df: df["Volume"] = 0.0
    df.Volume = pd.to_numeric(df.Volume, errors="coerce"); return df.sort_values("Date").drop_duplicates(subset=["Date"]).dropna(subset=["Date","Close"]).reset_index(drop=True)

def indicators(df):
    d=df.copy(); d["Return_1d"]=d.Close.pct_change(); d["MA_7"]=d.Close.rolling(7).mean(); d["MA_21"]=d.Close.rolling(21).mean(); d["Volatility_7"]=d.Return_1d.rolling(7).std()
    delta=d.Close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); d["RSI_14"]=100-(100/(1+(gain/(loss+1e-9))))
    ema12=d.Close.ewm(span=12, adjust=False).mean(); ema26=d.Close.ewm(span=26, adjust=False).mean(); d["MACD"]=ema12-ema26
    mid=d.Close.rolling(20).mean(); std=d.Close.rolling(20).std(); d["BB_Position"]=(d.Close-(mid-2*std))/(4*std+1e-9); d["Target_Next_Close"]=d.Close.shift(-1)
    return d.dropna(subset=FEATURES+["Target_Next_Close"]).reset_index(drop=True)

def price_chart(d):
    fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.7,.3], subplot_titles=("Price and trend indicators","RSI / MACD"))
    fig.add_trace(go.Candlestick(x=d.Date, open=d.get("Open",d.Close), high=d.get("High",d.Close), low=d.get("Low",d.Close), close=d.Close, name="Price"),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.Date,y=d.MA_7,name="MA 7"),row=1,col=1); fig.add_trace(go.Scatter(x=d.Date,y=d.MA_21,name="MA 21"),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.Date,y=d.RSI_14,name="RSI 14"),row=2,col=1); fig.add_trace(go.Scatter(x=d.Date,y=d.MACD,name="MACD"),row=2,col=1)
    fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False); return fig

def run_models(d, train_ratio, val_ratio, horizon):
    d=d.copy(); d["Target"]=d.Close.shift(-horizon); d=d.dropna(subset=FEATURES+["Target"]); n=len(d); a=int(n*train_ratio); b=int(n*(train_ratio+val_ratio)); train, val, test=d.iloc[:a],d.iloc[a:b],d.iloc[b:]
    models={"Linear Regression":LinearRegression(),"Decision Tree":DecisionTreeRegressor(max_depth=6,random_state=42),"Random Forest":RandomForestRegressor(n_estimators=250,max_depth=10,random_state=42,n_jobs=-1)}
    results=[]; fitted={}; predictions={"Date":test.Date,"Actual":test.Target,"Naive Baseline":test.Close}
    for name,model in models.items():
        model.fit(train[FEATURES],train.Target); fitted[name]=model; vp=model.predict(val[FEATURES]); tp=model.predict(test[FEATURES]); results.append({"Model":name,**metric_row(val.Target,vp)}); predictions[name]=tp
    results.append({"Model":"Naive Baseline",**metric_row(val.Target,val.Close)}); table=pd.DataFrame(results).sort_values("RMSE"); best=table.iloc[0].Model
    if best != "Naive Baseline":
        final=models[best].fit(pd.concat([train,val])[FEATURES],pd.concat([train,val]).Target); joblib.dump({"model":final,"features":FEATURES,"horizon":horizon},ROOT/"outputs"/"best_model.joblib")
    test_table=pd.DataFrame([{ "Model":k,**metric_row(test.Target,v) } for k,v in {"Naive Baseline":test.Close,**{n:predictions[n] for n in models}}.items()]).sort_values("RMSE")
    return table,test_table,pd.DataFrame(predictions),best,fitted

st.markdown("# 📊 MarketLens")
st.caption("A market analytics and machine-learning laboratory for academic research — not financial advice.")
with st.sidebar:
    st.header("Data source")
    source=st.radio("Choose source",["Included sample","Upload CSV","Download with yfinance"])
    if source=="Upload CSV": upload=st.file_uploader("CSV with Date and Close",type="csv")
    elif source=="Download with yfinance": ticker=st.text_input("Ticker", "AAPL"); period=st.selectbox("History",["6mo","1y","2y","5y","10y"],index=2)
    st.divider(); st.header("Experiment settings"); horizon=st.slider("Forecast horizon (days)",1,5,1); train_ratio=st.slider("Training share",.5,.8,.6,.05); val_ratio=st.slider("Validation share",.1,.3,.2,.05)

try:
    if source=="Included sample": raw=load_csv(ROOT/"data"/"sample_prices.csv")
    elif source=="Upload CSV": raw=load_csv(upload) if upload else None
    else:
        import yfinance as yf
        downloaded=yf.download(ticker,period=period,auto_adjust=False,progress=False)
        if isinstance(downloaded.columns,pd.MultiIndex): downloaded.columns=downloaded.columns.get_level_values(0)
        downloaded=downloaded.reset_index(); raw=load_csv(io.BytesIO(downloaded.to_csv(index=False).encode())) if not downloaded.empty else None
    if raw is None: st.info("Select a data source to begin."); st.stop()
    d=indicators(raw)
except Exception as exc: st.error(str(exc)); st.stop()

c1,c2,c3,c4=st.columns(4); c1.metric("Observations",f"{len(raw):,}"); c2.metric("Latest close",f"{raw.Close.iloc[-1]:,.2f}"); c3.metric("Average daily return",f"{raw.Close.pct_change().mean()*100:.2f}%"); c4.metric("Data quality",f"{raw.isna().sum().sum()} missing")

with st.expander("Data quality report", expanded=False):
    st.dataframe(pd.DataFrame({"Check":["Rows","Duplicate dates","Missing values","Start date","End date"],"Result":[len(raw),raw.Date.duplicated().sum(),raw.isna().sum().sum(),raw.Date.min().date(),raw.Date.max().date()]}),hide_index=True,use_container_width=True)

tab1,tab2,tab3,tab4=st.tabs(["📈 Market overview","🤖 Model laboratory","🧪 Backtest results","📥 Reports"])
with tab1:
    st.plotly_chart(price_chart(d),use_container_width=True); st.dataframe(d.tail(20),use_container_width=True,hide_index=True)
with tab2:
    if len(d)<40: st.warning("Use at least 40 rows after feature calculation for a meaningful experiment.")
    else:
        validation,test,pred,best,fitted=run_models(d,train_ratio,val_ratio,horizon)
        st.success(f"Best validation model: {best}")
        st.subheader("Validation comparison"); st.dataframe(validation.style.format({"MAE":"{:.3f}","MSE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.3f}"}),hide_index=True,use_container_width=True)
        st.subheader("Feature importance")
        if best in fitted and hasattr(fitted[best],"coef_"): imp=pd.DataFrame({"Feature":FEATURES,"Importance":np.abs(fitted[best].coef_)}).sort_values("Importance",ascending=False)
        elif best in fitted and hasattr(fitted[best],"feature_importances_"): imp=pd.DataFrame({"Feature":FEATURES,"Importance":fitted[best].feature_importances_}).sort_values("Importance",ascending=False)
        else: imp=pd.DataFrame({"Feature":[],"Importance":[]})
        if not imp.empty: st.bar_chart(imp.set_index("Feature"))
with tab3:
    if "test" not in locals(): validation,test,pred,best,fitted=run_models(d,train_ratio,val_ratio,horizon)
    st.subheader("Chronological test-set results"); st.dataframe(test.style.format({"MAE":"{:.3f}","MSE":"{:.3f}","RMSE":"{:.3f}","R²":"{:.3f}"}),hide_index=True,use_container_width=True)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=pred.Date,y=pred.Actual,name="Actual"))
    for col in pred.columns:
        if col not in {"Date","Actual"}: fig.add_trace(go.Scatter(x=pred.Date,y=pred[col],name=col))
    fig.update_layout(template="plotly_dark",height=500,title=f"{horizon}-day ahead forecast comparison"); st.plotly_chart(fig,use_container_width=True)
with tab4:
    if "pred" not in locals(): validation,test,pred,best,fitted=run_models(d,train_ratio,val_ratio,horizon)
    st.download_button("Download predictions CSV",pred.to_csv(index=False),"predictions.csv","text/csv")
    st.download_button("Download experiment report",json.dumps({"best_model":best,"validation":validation.to_dict("records"),"test":test.to_dict("records"),"features":FEATURES},indent=2),"experiment_report.json","application/json")
    st.info("The saved model is written to outputs/best_model.joblib after the experiment runs.")
