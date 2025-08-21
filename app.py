import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI
import json
import os

# ----------------------------
# Setup AIML API Client
# ----------------------------
AIML_BASE_URL = "https://api.aimlapi.com/v1"
TEXT_MODEL = "gpt-5-chat-latest"   # GPT-5 for text/Q&A
VISION_MODEL = "gpt-4o"            # optional, if you add image features

# Load API key from Streamlit secrets
AIML_API_KEY = st.secrets["AIML_API_KEY"]

# Initialize the client
client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🎨 PaletteGenie", layout="centered")
st.title("🎨 PaletteGenie – AI Color Palette Generator")
st.write("Turn ideas into beautiful color palettes with GPT-5 ✨")

# User input
base_color = st.color_picker("Pick a base color", "#3498db")
theme = st.text_input("Or describe a theme (e.g. 'summer pastel mood')")

# Generate button
if st.button("Generate Palette"):
    with st.spinner("🎨 Mixing colors with GPT-5..."):
        prompt = f"""
        You are a professional color designer.
        Based on the base color {base_color} and the theme '{theme}',
        suggest 5 harmonious HEX color codes.
        Output strictly in JSON array with this format:
        [
          {{"hex": "#RRGGBB", "reason": "short explanation"}}
        ]
        """

        try:
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            palette = json.loads(content)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            palette = []

        # ----------------------------
        # Display palette
        # ----------------------------
        if palette:
            st.subheader("✨ Generated Palette")

            fig, ax = plt.subplots(1, figsize=(10, 2))
            ax.axis("off")

            for i, color in enumerate(palette):
                rect = patches.Rectangle((i, 0), 1, 1, facecolor=color["hex"])
                ax.add_patch(rect)
                ax.text(i + 0.5, -0.2, color["hex"], ha="center", fontsize=9)

            plt.xlim(0, len(palette))
            plt.ylim(0, 1)
            st.pyplot(fig)

            st.subheader("💡 Why these colors?")
            for c in palette:
                st.markdown(f"- **{c['hex']}** → {c['reason']}")
        else:
            st.error("❌ Could not generate palette. Try again.")
