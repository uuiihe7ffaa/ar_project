import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Анализ отзывов по аспектам", page_icon="📝")

st.title("📝 Анализ отзывов по аспектам")
st.write("Выберите тип места, введите отзыв, и система определит его основной аспект.")

place_type = st.selectbox(
    "Тип места",
    [
        "кафе",
        "ресторан",
        "доставка",
    ]
)

text = st.text_area("Введите отзыв", height=200)

if st.button("Анализировать"):
    if not text.strip():
        st.warning("Введите текст отзыва.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"place_type": place_type, "text": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Основной аспект")
            st.success(data["predicted_aspect"])

            st.subheader("Вероятности по аспектам")
            probs = data.get("probabilities", {})
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            for label, score in sorted_probs:
                st.progress(float(score), text=f"{label}: {score:.3f}")

        except Exception as e:
            st.error(f"Ошибка при обращении к API: {e}")