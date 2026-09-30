
import joblib
import pandas as pd
import streamlit as st

pipeline = joblib.load("./heart_disease_pipeline.joblib")

CONFIDENCE_THRESHOLD = 0.70

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction")
st.caption("Cleveland Heart Disease Dataset")

st.info(
    "ระบบนี้เป็นต้นแบบเพื่อการเรียนรู้ด้าน Machine Learning เท่านั้น "
    "ไม่ใช่เครื่องมือวินิจฉัยโรค และไม่ควรใช้แทนการประเมินโดยแพทย์"
)

st.markdown(
    """
กรอกข้อมูลให้ครบทั้ง 13 รายการ แล้วกด **ทำนาย (Predict)**  
ชื่อภาษาอังกฤษในวงเล็บคือชื่อ feature ที่ใช้ใน dataset
"""
)

with st.form("prediction_form"):

    st.subheader("1. ข้อมูลพื้นฐาน")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "อายุ (Age)",
            min_value=29,
            max_value=77,
            value=54,
            step=1,
            help="อายุ หน่วยเป็นปี"
        )

    with col2:
        sex = st.selectbox(
            "เพศ (Sex)",
            options=[0, 1],
            format_func=lambda x: {
                0: "หญิง (Female)",
                1: "ชาย (Male)"
            }[x]
        )

    st.subheader("2. อาการและผลตรวจขณะพัก")

    col1, col2 = st.columns(2)

    with col1:
        cp = st.selectbox(
            "ประเภทอาการเจ็บหน้าอก (Chest pain type)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical angina",
                1: "Atypical angina",
                2: "Non-anginal pain",
                3: "Asymptomatic"
            }[x]
        )

        trestbps = st.number_input(
            "ความดันโลหิตขณะพัก (Resting blood pressure)",
            min_value=94,
            max_value=200,
            value=130,
            step=1,
            help="หน่วย mmHg"
        )

        chol = st.number_input(
            "คอเลสเตอรอลในเลือด (Serum cholesterol)",
            min_value=126,
            max_value=564,
            value=240,
            step=1,
            help="หน่วย mg/dL"
        )

    with col2:
        fbs = st.selectbox(
            "น้ำตาลขณะอดอาหาร > 120 mg/dL (Fasting blood sugar)",
            options=[0, 1],
            format_func=lambda x: {
                0: "ไม่",
                1: "ใช่"
            }[x]
        )

        restecg = st.selectbox(
            "ผลคลื่นไฟฟ้าหัวใจขณะพัก (Resting ECG)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T abnormality",
                2: "Left ventricular hypertrophy"
            }[x]
        )

        ca = st.selectbox(
            "จำนวนหลอดเลือดหัวใจหลัก (Major vessels)",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} เส้น"
        )

    st.subheader("3. ผลตรวจจากการออกกำลังกาย")

    col1, col2 = st.columns(2)

    with col1:
        thalach = st.number_input(
            "อัตราการเต้นหัวใจสูงสุด (Maximum heart rate)",
            min_value=71,
            max_value=202,
            value=150,
            step=1,
            help="หน่วยครั้งต่อนาที"
        )

        exang = st.selectbox(
            "มีอาการเจ็บหน้าอกจากการออกกำลังกายหรือไม่",
            options=[0, 1],
            format_func=lambda x: {
                0: "ไม่มี",
                1: "มี"
            }[x]
        )

        oldpeak = st.number_input(
            "ระดับ ST depression (Oldpeak)",
            min_value=0.0,
            max_value=6.2,
            value=1.0,
            step=0.1,
            format="%.1f"
        )

    with col2:
        slope = st.selectbox(
            "ลักษณะความชันของ ST segment (Slope)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )

        thal = st.selectbox(
            "ผลการตรวจ Thal",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed defect",
                2: "Reversible defect"
            }[x]
        )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True
    )

if submitted:

    features = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    features = pd.DataFrame([features])

    prediction = int(pipeline.predict(features)[0])
    probability = pipeline.predict_proba(features)[0]

    probability_class_0 = float(probability[0])
    probability_class_1 = float(probability[1])
    confidence = float(max(probability))

    st.divider()

    st.subheader("ผลการทำนาย")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Class 0",
        f"{probability_class_0:.1%}"
    )

    col2.metric(
        "Class 1",
        f"{probability_class_1:.1%}"
    )

    col3.metric(
        "Confidence",
        f"{confidence:.1%}"
    )

    if confidence < CONFIDENCE_THRESHOLD:

        st.warning(
            f"โมเดลมีความมั่นใจต่ำกว่า {CONFIDENCE_THRESHOLD:.0%} "
            "จึงควรตีความผลด้วยความระมัดระวัง"
        )

    elif prediction == 1:

        st.error(
            "Predicted Class: 1\n\n"
            "โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม condition = 1"
        )

    else:

        st.success(
            "Predicted Class: 0\n\n"
            "โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม condition = 0"
        )

    st.caption(
        "Probability และ Confidence เป็นผลจากโมเดลต้นแบบ "
        "ไม่ใช่ค่าความเสี่ยงทางการแพทย์"
    )

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(
            features,
            use_container_width=True
        )
