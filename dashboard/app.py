from __future__ import annotations

import json
from pathlib import Path
import sys
from io import BytesIO
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from database.service import analytics_summary, save_prediction
from src.inference import predict, predict_batch

st.set_page_config(page_title="Customer Voice Intelligence", page_icon="💬", layout="wide")
st.markdown("""
<style>
.hero{padding:1.6rem 2rem;border-radius:18px;background:linear-gradient(120deg,#172a46,#245ca6);color:white;margin-bottom:1rem}
.hero h1{margin:0 0 .35rem}.hero p{margin:0;color:#dcecff}[data-testid=stMetric]{border:1px solid #dce5ef;background:#f7f9fc;padding:12px;border-radius:12px}
</style><div class="hero"><h1>Customer Voice Intelligence</h1><p>客户之声智能分析平台 · Sentiment, topic routing, risk triage, batch analytics and model evidence</p></div>
""", unsafe_allow_html=True)

metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"; benchmark_path = PROJECT_ROOT / "outputs" / "benchmark.csv"
tabs = st.tabs(["Overview / 总览", "Single Prediction / 单条预测", "Batch Analysis / 批量分析", "Model Evidence / 模型证据"])

with tabs[0]:
    summary = analytics_summary(); measured = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    cols = st.columns(5)
    cols[0].metric("Stored predictions", summary["total_predictions"])
    cols[1].metric("Test rows", measured.get("test_rows", "—"))
    cols[2].metric("Production Macro F1", f"{measured.get('metrics',{}).get('macro_f1',0):.2%}")
    cols[3].metric("ROC-AUC", f"{measured.get('metrics',{}).get('roc_auc',0):.2%}")
    cols[4].metric("High-risk cases", summary["high_risk"])
    st.markdown("**Measured on a held-out UCI test split / 指标来自 UCI 留出测试集，不是演示样本。**")
    c1,c2=st.columns(2)
    if summary["sentiments"]: c1.bar_chart(pd.Series(summary["sentiments"], name="count"))
    if summary["topics"]: c2.bar_chart(pd.Series(summary["topics"], name="count"))

with tabs[1]:
    example = "The battery is good but delivery was terrible and support refused my refund."
    text = st.text_area("Customer feedback / 客户反馈", example, height=130)
    if st.button("Analyze / 分析", type="primary"):
        try:
            result = predict(text); save_prediction(result, "dashboard"); st.session_state["prediction"] = result
        except Exception as error: st.error(str(error))
    result = st.session_state.get("prediction")
    if result:
        cols=st.columns(4); cols[0].metric("Sentiment",result["sentiment"]); cols[1].metric("Confidence",f"{result['confidence']:.1%}"); cols[2].metric("Topic",result["topic"]); cols[3].metric("Risk",result["risk"],f"score {result['risk_score']}")
        st.json({k:result[k] for k in ("topic_keywords","risk_keywords","model_name","model_version","disclaimer")})

with tabs[2]:
    uploaded=st.file_uploader("Upload CSV with a text column / 上传含 text 列的 CSV",type=["csv"])
    if uploaded:
        frame=pd.read_csv(uploaded)
        if "text" not in frame.columns: st.error("CSV must contain a text column.")
        elif len(frame)>5000: st.error("Maximum batch size is 5,000 rows.")
        else:
            st.dataframe(frame.head(20),width="stretch")
            if st.button("Run batch / 批量分析",type="primary"):
                results=predict_batch(frame.text.fillna("").astype(str).tolist()); result_frame=pd.DataFrame(results); st.session_state["batch_result"]=result_frame
            result_frame=st.session_state.get("batch_result")
            if result_frame is not None:
                c1,c2=st.columns(2); c1.bar_chart(result_frame.sentiment.value_counts()); c2.bar_chart(result_frame.topic.value_counts())
                st.dataframe(result_frame,width="stretch"); st.download_button("Download results / 下载结果",result_frame.to_csv(index=False).encode("utf-8-sig"),"voc_predictions.csv","text/csv")

with tabs[3]:
    if benchmark_path.exists():
        table=pd.read_csv(benchmark_path); st.dataframe(table.style.format({"accuracy":"{:.3f}","macro_f1":"{:.3f}","cv_f1_mean":"{:.3f}","cv_f1_std":"{:.3f}"}),width="stretch")
        c1,c2=st.columns(2); c1.image(str(PROJECT_ROOT/"outputs"/"model_comparison.png"),caption="Five-model benchmark"); c2.image(str(PROJECT_ROOT/"outputs"/"confusion_matrix.png"),caption="Production model confusion matrix")
        st.image(str(PROJECT_ROOT/"outputs"/"feature_importance.png"),caption="Interpretable positive and negative features")
    else: st.info("Run python -m src.benchmark first.")

