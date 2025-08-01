import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# page configuration
st.set_page_config(
    page_title="Salmonella Ceftriaxone Resistance Prediction Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load models
@st.cache_resource
def load_models():
    model_c1 = joblib.load(r'E:/RS/sm/耐药性分析/文章图/特征递减/C1/5/C1-5.pkl')
    model_e1 = joblib.load(r'E:/RS/sm/耐药性分析/文章图/特征递减/E1/20/E1-20.pkl')
    model_st = joblib.load(r'E:/RS/sm/耐药性分析/文章图/特征递减/鼠伤寒/20/鼠伤寒-14.pkl')
    return model_c1, model_e1, model_st

model1, model2, model3 = load_models()

# mapping models to features
data_info = {
    'C1 Group Ceftriaxone Resistance': {
        'model': model1,
        'features': ['6009', '6093', '9329', '12191', '3869']
    },
    'E1 Group Ceftriaxone Resistance': {
        'model': model2,
        'features': ['10447', '3158', '3582', '13407', '13530', '2225', '2181', '3275', '2421', '5078',
                     '3321', '8830', '5611', '8526', '10779', '2687', '11759', '2957', '3671', '12190']
    },
    'Salmonella Typhimurium Ceftriaxone Resistance': {
        'model': model3,
        'features': ['9406', '3550', '2856', '4048', '4619', '9061', '6564', '5374', '4449', '7099',
                     '3129', '3646', '2594', '3303']
    }
}

# sidebar for model selection
st.sidebar.title("Model Selection")
selected = st.sidebar.selectbox(
    "Choose a model:",
    options=list(data_info.keys())
)

# main page title
st.title("Salmonella Ceftriaxone Resistance Prediction Tool")
st.subheader(f"Selected model: {selected}")

# get model and features
info = data_info[selected]
model = info['model']
features = info['features']

# predefined default values (scaled to 1e-4)
default_values = {
    'C1 Group Ceftriaxone Resistance': [0.0004942, 0.0008160, 0.00005961, 0.00004461, 0.00004147],
    'E1 Group Ceftriaxone Resistance': [
        0.00002656, 0.00023202, 0.00015214, 0.00001446, 0.00002562, 0.00004841, 0.00001890, 0.00002820,
        0.00000592, 0.00000430, 0.00001265, 0.00007364, 0.00008401, 0.00006309, 0.00000090, 0.00029921,
        0.00003830, 0.00001903, 0.00010178, 0.00000799
    ],
    'Salmonella Typhimurium Ceftriaxone Resistance': [
        0.00004261, 0.00007952, 0.00005446, 0.00000646, 0.00007760, 0.0002460, 0.00011672, 0.00083649,
        0.00013354, 0.00033460, 0.00019590, 0.00009203, 0.00003673, 0.00001467
    ]
}

# input fields with values in unit of 1e-4
cols = st.columns(2)
inputs = {}
defaults = default_values[selected] if selected in default_values else [0.0] * len(features)

for idx, feat in enumerate(features):
    col = cols[idx % 2]
    raw_value = defaults[idx] if idx < len(defaults) else 0.0
    scaled_value = raw_value * 1e4  # scaled to user-friendly unit
    user_input = col.number_input(
        label=f"{feat} m/z (×10⁻⁴)",
        format="%.4f",
        value=scaled_value,
        key=f"{selected}_{feat}"
    )
    # convert back to real value
    inputs[feat] = user_input * 1e-4

# prediction
if st.button("Predict"):
    # prepare data
    df = pd.DataFrame([inputs[f] for f in features]).T
    df.columns = features
    # get probabilities and prediction
    proba = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    label = 'Resistant' if pred == 1 else 'Sensitive'

    # display probability bar chart with different colors
    fig, ax = plt.subplots()
    bars = ax.bar(['Sensitive', 'Resistant'], [proba[0], proba[1]],
                  color=['#4C72B0', '#FF8C42'])  # Set colors for each bar
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1.1)
    ax.set_title('Prediction Probabilities')

    # optionally add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

    st.pyplot(fig)

    st.markdown("---")

    # SHAP for single sample
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    if isinstance(shap_values, list):
        shap_for_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_for_pos = shap_values
    # shap_for_pos might be shape (1, n_features) or flattened or multi-dimensional
    import numpy as np
    sample_shap = np.array(shap_for_pos)
    # flatten and ensure correct length
    sample_shap = sample_shap.flatten()[:len(features)]

    # display SHAP bar chart with positive/negative colors
    fig2, ax2 = plt.subplots()

    # y positions
    y_pos = range(len(features))

    # 使用颜色映射：正值为橙色，负值为蓝色（可自定义）
    colors = ['#FF8C42' if v >= 0 else '#4C72B0' for v in sample_shap]  # 橙 / 蓝

    # 绘制水平柱状图
    ax2.barh(y_pos, sample_shap, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.set_xlabel('SHAP value')
    ax2.set_title('Feature Contribution (SHAP)')
    ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # 添加0刻度线
    st.pyplot(fig2)

# footer# footer
st.markdown("---")
st.markdown("&copy; 2025 Antimicrobial Resistance Lab")


