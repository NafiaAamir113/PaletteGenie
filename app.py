# import streamlit as st
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from openai import OpenAI
# import json
# import os

# # ----------------------------
# # Setup AIML API Client
# # ----------------------------
# AIML_BASE_URL = "https://api.aimlapi.com/v1"
# TEXT_MODEL = "gpt-5-chat-latest"   # GPT-5 for text/Q&A
# VISION_MODEL = "gpt-4o"            # optional, if you add image features

# # Load API key from Streamlit secrets
# AIML_API_KEY = st.secrets["AIML_API_KEY"]

# # Initialize the client
# client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# # ----------------------------
# # Streamlit UI
# # ----------------------------
# st.set_page_config(page_title="🎨 PaletteGenie", layout="centered")
# st.title("🎨 PaletteGenie – AI Color Palette Generator")
# st.write("Turn ideas into beautiful color palettes with GPT-5 ✨")

# # User input
# base_color = st.color_picker("Pick a base color", "#3498db")
# theme = st.text_input("Or describe a theme (e.g. 'summer pastel mood')")

# # Generate button
# if st.button("Generate Palette"):
#     with st.spinner("🎨 Mixing colors with GPT-5..."):
#         prompt = f"""
#         You are a professional color designer.
#         Based on the base color {base_color} and the theme '{theme}',
#         suggest 5 harmonious HEX color codes.
#         Output strictly in JSON array with this format:
#         [
#           {{"hex": "#RRGGBB", "reason": "short explanation"}}
#         ]
#         """

#         try:
#             response = client.chat.completions.create(
#                 model=TEXT_MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.7
#             )
#             content = response.choices[0].message.content.strip()
#             palette = json.loads(content)
#         except Exception as e:
#             st.error(f"⚠️ Error: {e}")
#             palette = []

#         # ----------------------------
#         # Display palette
#         # ----------------------------
#         if palette:
#             st.subheader("✨ Generated Palette")

#             fig, ax = plt.subplots(1, figsize=(10, 2))
#             ax.axis("off")

#             for i, color in enumerate(palette):
#                 rect = patches.Rectangle((i, 0), 1, 1, facecolor=color["hex"])
#                 ax.add_patch(rect)
#                 ax.text(i + 0.5, -0.2, color["hex"], ha="center", fontsize=9)

#             plt.xlim(0, len(palette))
#             plt.ylim(0, 1)
#             st.pyplot(fig)

#             st.subheader("💡 Why these colors?")
#             for c in palette:
#                 st.markdown(f"- **{c['hex']}** → {c['reason']}")
#         else:
#             st.error("❌ Could not generate palette. Try again.")

# # SECOND CODE.

# import streamlit as st 
# import os, io
# from colorthief import ColorThief
# from PIL import Image
# from openai import OpenAI

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="PaletteGenie", page_icon="🎨", layout="wide")
# st.title("🎨 PaletteGenie (GPT-5) — Palette + Shade Ideas + Art Q&A")

# # ----------------------------
# # Setup AIML API Client
# # ----------------------------
# AIML_BASE_URL = "https://api.aimlapi.com/v1"
# TEXT_MODEL = "gpt-5-chat-latest"
# AIML_API_KEY = st.secrets["AIML_API_KEY"]  # safely stored in Streamlit Secrets

# client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# # ------------------ HELPERS ------------------
# def rgb_to_hex(rgb):
#     r, g, b = rgb
#     return f"#{r:02x}{g:02x}{b:02x}"

# def extract_palette(img_bytes: bytes, n_colors: int = 6):
#     bio = io.BytesIO(img_bytes)
#     ct = ColorThief(bio)
#     palette = ct.get_palette(color_count=n_colors)
#     return [rgb_to_hex(c) for c in palette]

# def gpt5_chat_answer(context_block, history, new_question):
#     messages = [
#         {"role": "system", "content": (
#             "You are an art + fashion design mentor. "
#             "Help the user create new color shade combinations, palettes, and provide art suggestions. "
#             "Base your answers on the provided palette but also propose new shade variations, tints, tones, and blends."
#         )},
#         {"role": "user", "content": context_block},
#     ] + history + [{"role": "user", "content": new_question}]

#     resp = client.chat.completions.create(
#         model=TEXT_MODEL,
#         messages=messages,
#         temperature=0.8,
#         max_tokens=600,
#     )
#     return resp.choices[0].message.content

# def render_palette_boxes(hex_list):
#     cols = st.columns(len(hex_list))
#     for i, h in enumerate(hex_list):
#         with cols[i]:
#             st.markdown(
#                 f"""
#                 <div style="border-radius:12px; height:80px; width:100%; 
#                 border:1px solid #ddd; background:{h};"></div>
#                 <div style="font-family:monospace; margin-top:6px; text-align:center;">{h}</div>
#                 """,
#                 unsafe_allow_html=True
#             )

# # ------------------ UI ------------------
# st.sidebar.header("🎨 Palette Options")
# mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Enter HEX Colors"])

# hex_palette = []

# if mode == "Upload Image":
#     uploaded = st.file_uploader("Upload a drawing / fabric / artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         image = Image.open(uploaded).convert("RGB")
#         st.image(image, caption="Your Uploaded Artwork", width=300)

#         with st.spinner("🎨 Extracting main palette..."):
#             hex_palette = extract_palette(uploaded.getvalue())
#         st.subheader("Extracted Palette")
#         render_palette_boxes(hex_palette)

# elif mode == "Enter HEX Colors":
#     user_colors = st.text_input("Enter HEX colors (comma-separated, e.g. #FF5733, #33FFCE, #112233)")
#     if user_colors:
#         hex_palette = [c.strip() for c in user_colors.split(",") if c.strip()]
#         st.subheader("Your Custom Palette")
#         render_palette_boxes(hex_palette)

# # ------------------ Chat Section ------------------
# if hex_palette:
#     context_block = f"PALETTE={hex_palette}"

#     st.subheader("💬 Chat with GPT-5 about Colors & Design")
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_msg = st.chat_input("Ask about new shades, color mixing, outfit combos, etc.")
#     if user_msg:
#         with st.chat_message("user"):
#             st.markdown(user_msg)
#         try:
#             answer = gpt5_chat_answer(context_block, st.session_state.chat_history, user_msg)
#         except Exception as e:
#             answer = f"⚠️ Sorry, I couldn't get an answer: {e}"

#         st.session_state.chat_history.append({"role": "user", "content": user_msg})
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})

#         with st.chat_message("assistant"):
#             st.markdown(answer)

# else:
#     st.info("👉 Upload an image OR enter your own HEX colors to start exploring palettes with GPT-5.")


import streamlit as st
import os, io, hashlib, base64
from colorthief import ColorThief
from PIL import Image
from openai import OpenAI
import webcolors

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="PaletteGenie+", page_icon="🎨", layout="wide")
st.title("🎨 PaletteGenie+ (GPT-5) — Art Mentor, Palette, & Style Guide")

# ----------------------------
# Setup AIML API Client
# ----------------------------
AIML_BASE_URL = "https://api.aimlapi.com/v1"
TEXT_MODEL = "gpt-5-chat-latest"
AIML_API_KEY = st.secrets["AIML_API_KEY"]

client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# ------------------ HELPERS ------------------
def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def extract_palette(img_bytes: bytes, n_colors: int = 6):
    bio = io.BytesIO(img_bytes)
    ct = ColorThief(bio)
    palette = ct.get_palette(color_count=n_colors)
    return [rgb_to_hex(c) for c in palette]

def closest_paint_name(requested_hex):
    css3_names = {name: webcolors.name_to_hex(name) for name in webcolors.CSS3_NAMES_TO_HEX.keys()}
    requested_rgb = webcolors.hex_to_rgb(requested_hex)
    min_diff = float("inf")
    closest_name = None

    for name, hex_val in css3_names.items():
        r, g, b = webcolors.hex_to_rgb(hex_val)
        diff = (r - requested_rgb[0]) ** 2 + (g - requested_rgb[1]) ** 2 + (b - requested_rgb[2]) ** 2
        if diff < min_diff:
            min_diff = diff
            closest_name = name

    return closest_name

def gpt5_chat_answer(context_block, history, new_question, artist=None, image_bytes=None):
    system_prompt = "You are an art + fashion design mentor. Help the user with palettes, shades, and design advice."
    if artist:
        system_prompt += f" Answer in the style and philosophy of {artist}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_block},
    ] + history + [{"role": "user", "content": new_question}]

    if image_bytes:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages.append({
            "role": "user",
            "content": {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
        })

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=600,
    )
    return resp.choices[0].message.content

def render_palette_boxes(hex_list):
    cols = st.columns(len(hex_list))
    for i, h in enumerate(hex_list):
        with cols[i]:
            st.markdown(
                f"""
                <div style="border-radius:12px; height:80px; width:100%; 
                border:1px solid #ddd; background:{h};"></div>
                <div style="font-family:monospace; margin-top:6px; text-align:center;">{h}</div>
                """,
                unsafe_allow_html=True
            )

# ------------------ UI ------------------
st.sidebar.header("🎨 Features")
mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Enter HEX Colors"])

artist = st.sidebar.selectbox(
    "👩‍🎨 Choose an Artist Mentor:",
    ["None", "Picasso", "Van Gogh", "Frida Kahlo", "Monet", "Salvador Dalí"]
)

hex_palette = []
uploaded_image_bytes = None

# Upload or enter colors
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your Uploaded Artwork", width=300)
        uploaded_image_bytes = uploaded.getvalue()

        with st.spinner("🎨 Extracting main palette..."):
            hex_palette = extract_palette(uploaded_image_bytes)
        st.subheader("Extracted Palette")
        render_palette_boxes(hex_palette)

elif mode == "Enter HEX Colors":
    user_colors = st.text_input("Enter HEX colors (comma-separated, e.g. #FF5733, #33FFCE, #112233)")
    if user_colors:
        hex_palette = [c.strip() for c in user_colors.split(",") if c.strip()]
        st.subheader("Your Custom Palette")
        render_palette_boxes(hex_palette)

# ------------------ Chat Section ------------------
if hex_palette or uploaded_image_bytes:
    context_block = f"PALETTE={hex_palette}" if hex_palette else "User uploaded an artwork."

    st.subheader("💬 Chat with GPT-5 about Art, Colors & Design")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask for critique, style advice, or new shades...")
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)

        try:
            answer = gpt5_chat_answer(
                context_block,
                st.session_state.chat_history,
                user_msg,
                artist=artist if artist != "None" else None,
                image_bytes=uploaded_image_bytes
            )
        except Exception as e:
            answer = f"⚠️ Sorry, I couldn't get an answer: {e}"

        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("👉 Upload an image OR enter your own HEX colors to start exploring palettes with GPT-5.")

