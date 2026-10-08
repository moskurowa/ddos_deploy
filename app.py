
import joblib
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load("ddos_pipeline")
features = list(model.feature_names_in_)
classes = list(model.classes_)
THRESHOLD = 0.80
MAX_ROWS = 1_000_000

st.set_page_config(page_title="DDoS (HTTP Flood) classifier", layout="wide")


def prep(data: pd.DataFrame, features: list) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.str.strip()
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError("ไฟล์มีปัญหา: ขาดคอลัมน์ " + ", ".join(missing))
    return data[features].replace([np.inf, -np.inf], np.nan).astype("float32")


@st.cache_data(show_spinner=False)
def run_prediction(file_bytes: bytes):
    import io

    data = pd.read_csv(io.BytesIO(file_bytes), nrows=MAX_ROWS)
    X = prep(data, features)

    p = model.predict_proba(X)
    conf = p.max(axis=1)
    label = np.array(classes)[p.argmax(axis=1)]
    status = np.where(conf < THRESHOLD, "ไม่มั่นใจ", label)

    summary = (
        pd.Series(status)
        .value_counts()
        .rename_axis("ผลทำนาย")
        .reset_index(name="จำนวน")
    )

    detail = pd.DataFrame(
        {
            "แถวที่": np.arange(1, len(data) + 1),
            "ความมั่นใจ": conf.round(4),
            "สถานะ": status,
        }
    )
    order = {"ไม่มั่นใจ": 0, "DDoS": 1, "BENIGN": 2}
    detail["_ord"] = detail["สถานะ"].map(order)
    detail = detail.sort_values(["_ord", "ความมั่นใจ"]).drop(columns="_ord")
    return summary, detail, len(data)

st.title("DDoS (HTTP Flood) classifier")
st.markdown(
    f"ตรวจได้เฉพาะ HTTP flood (ถ้าโมเดลทายแล้วมั่นใจต่ำกว่า {THRESHOLD:.0%} "
    "แถวนั้นจะถูก label ว่า **ไม่มั่นใจ**)\n\n"
    "อัปโหลดไฟล์ CSV จาก CICFlowMeter"
)


up = st.file_uploader(f"ไฟล์ CSV (อ่านสูงสุด {MAX_ROWS:,} แถว)", type="csv")

if st.button("ทำนาย", type="primary", disabled=up is None):
    try:
        with st.spinner("กำลังทำนาย..."):
            summary, detail, n = run_prediction(up.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.caption(f"อ่านทั้งหมด {n:,} แถว")

    st.subheader("สรุปผลตามคลาส")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("รายแถว")
    st.dataframe(detail,use_container_width=True, hide_index=True)
