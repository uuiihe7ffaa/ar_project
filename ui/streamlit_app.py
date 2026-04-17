import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Анализ отзывов по аспектам", page_icon="📝")

st.title("📝 Анализ отзывов по аспектам")
st.write("Выберите тип места, введите отзыв, и система определит аспекты текста.")

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

            st.subheader("Обнаруженные аспекты")
            for aspect in data["predicted_aspects"]:
                st.write(f"• {aspect}")

            st.subheader("Вероятности по аспектам")
            probs = data.get("probabilities", {})
            for label, score in probs.items():
                st.progress(float(score), text=f"{label}: {score:.3f}")

        except Exception as e:
            st.error(f"Ошибка при обращении к API: {e}")