# import streamlit as st
# import os, io, hashlib, base64
# from colorthief import ColorThief
# from PIL import Image
# from openai import OpenAI
# import webcolors

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="PaletteGenie+", page_icon="🎨", layout="wide")
# st.title("🎨 PaletteGenie+ (GPT-5) — Art Mentor, Palette, & Style Guide")

# # ----------------------------
# # Setup AIML API Client
# # ----------------------------
# AIML_BASE_URL = "https://api.aimlapi.com/v1"
# TEXT_MODEL = "gpt-5-chat-latest"
# AIML_API_KEY = st.secrets["AIML_API_KEY"]

# client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# # ------------------ HELPERS ------------------
# def rgb_to_hex(rgb):
#     r, g, b = rgb
#     return f"#{r:02x}{g:02x}{b:02x}"

# def hex_to_rgb(hex_color: str):
#     hex_color = hex_color.lstrip('#')
#     return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# def extract_palette(img_bytes: bytes, n_colors: int = 6):
#     bio = io.BytesIO(img_bytes)
#     ct = ColorThief(bio)
#     palette = ct.get_palette(color_count=n_colors)
#     return [rgb_to_hex(c) for c in palette]

# def closest_paint_name(requested_hex):
#     css3_names = {name: webcolors.name_to_hex(name) for name in webcolors.CSS3_NAMES_TO_HEX.keys()}
#     requested_rgb = webcolors.hex_to_rgb(requested_hex)
#     min_diff = float("inf")
#     closest_name = None

#     for name, hex_val in css3_names.items():
#         r, g, b = webcolors.hex_to_rgb(hex_val)
#         diff = (r - requested_rgb[0]) ** 2 + (g - requested_rgb[1]) ** 2 + (b - requested_rgb[2]) ** 2
#         if diff < min_diff:
#             min_diff = diff
#             closest_name = name

#     return closest_name

# def gpt5_chat_answer(context_block, history, new_question, artist=None, image_bytes=None):
#     system_prompt = "You are an art + fashion design mentor. Help the user with palettes, shades, and design advice."
#     if artist:
#         system_prompt += f" Answer in the style and philosophy of {artist}."

#     # ✅ Build message list
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": context_block},
#     ] + history

#     # ✅ Handle multimodal case properly
#     if image_bytes:
#         img_b64 = base64.b64encode(image_bytes).decode("utf-8")
#         messages.append({
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": new_question},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ]
#         })
#     else:
#         messages.append({"role": "user", "content": new_question})

#     # ✅ Call API
#     resp = client.chat.completions.create(
#         model=TEXT_MODEL,
#         messages=messages,
#         temperature=0.8,
#         max_tokens=600,
#     )

#     # Some SDKs return dict-style, others object-style
#     content = resp.choices[0].message.get("content") if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
#     return content

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
# st.sidebar.header("🎨 Features")
# mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Enter HEX Colors"])

# artist = st.sidebar.selectbox(
#     "👩‍🎨 Choose an Artist Mentor:",
#     ["None", "Picasso", "Van Gogh", "Frida Kahlo", "Monet", "Salvador Dalí"]
# )

# hex_palette = []
# uploaded_image_bytes = None

# # Upload or enter colors
# if mode == "Upload Image":
#     uploaded = st.file_uploader("Upload an artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         image = Image.open(uploaded).convert("RGB")
#         st.image(image, caption="Your Uploaded Artwork", width=300)
#         uploaded_image_bytes = uploaded.getvalue()

#         with st.spinner("🎨 Extracting main palette..."):
#             hex_palette = extract_palette(uploaded_image_bytes)
#         st.subheader("Extracted Palette")
#         render_palette_boxes(hex_palette)

# elif mode == "Enter HEX Colors":
#     user_colors = st.text_input("Enter HEX colors (comma-separated, e.g. #FF5733, #33FFCE, #112233)")
#     if user_colors:
#         hex_palette = [c.strip() for c in user_colors.split(",") if c.strip()]
#         st.subheader("Your Custom Palette")
#         render_palette_boxes(hex_palette)

# # ------------------ Chat Section ------------------
# if hex_palette or uploaded_image_bytes:
#     context_block = f"PALETTE={hex_palette}" if hex_palette else "User uploaded an artwork."

#     st.subheader("💬 Chat with GPT-5 about Art, Colors & Design")
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_msg = st.chat_input("Ask for critique, style advice, or new shades...")
#     if user_msg:
#         with st.chat_message("user"):
#             st.markdown(user_msg)

#         try:
#             answer = gpt5_chat_answer(
#                 context_block,
#                 st.session_state.chat_history,
#                 user_msg,
#                 artist=artist if artist != "None" else None,
#                 image_bytes=uploaded_image_bytes
#             )
#         except Exception as e:
#             answer = f"⚠️ Sorry, I couldn't get an answer: {e}"

#         st.session_state.chat_history.append({"role": "user", "content": user_msg})
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})

#         with st.chat_message("assistant"):
#             st.markdown(answer)

# else:
#     st.info("👉 Upload an image OR enter your own HEX colors to start exploring palettes with GPT-5.")

# # SECOND CODE

# import streamlit as st
# import os, io, hashlib, base64
# from colorthief import ColorThief
# from PIL import Image
# from openai import OpenAI
# import webcolors

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="PaletteGenie+", page_icon="🎨", layout="wide")
# st.title("🎨 PaletteGenie+ (GPT-5) — Art Mentor, Palette, & Style Guide")

# # ----------------------------
# # Setup AIML API Client
# # ----------------------------
# AIML_BASE_URL = "https://api.aimlapi.com/v1"
# TEXT_MODEL = "gpt-5-chat-latest"
# AIML_API_KEY = st.secrets["AIML_API_KEY"]

# client = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL)

# # ------------------ HELPERS ------------------
# def rgb_to_hex(rgb):
#     r, g, b = rgb
#     return f"#{r:02x}{g:02x}{b:02x}"

# def hex_to_rgb(hex_color: str):
#     hex_color = hex_color.lstrip('#')
#     return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# def extract_palette(img_bytes: bytes, n_colors: int = 6):
#     bio = io.BytesIO(img_bytes)
#     ct = ColorThief(bio)
#     palette = ct.get_palette(color_count=n_colors)
#     return [rgb_to_hex(c) for c in palette]

# def closest_paint_name(requested_hex):
#     css3_names = {name: webcolors.name_to_hex(name) for name in webcolors.CSS3_NAMES_TO_HEX.keys()}
#     requested_rgb = webcolors.hex_to_rgb(requested_hex)
#     min_diff = float("inf")
#     closest_name = None

#     for name, hex_val in css3_names.items():
#         r, g, b = webcolors.hex_to_rgb(hex_val)
#         diff = (r - requested_rgb[0]) ** 2 + (g - requested_rgb[1]) ** 2 + (b - requested_rgb[2]) ** 2
#         if diff < min_diff:
#             min_diff = diff
#             closest_name = name

#     return closest_name

# # ✅ FIXED FUNCTION (avoids resp.choices[0].message issue)
# def gpt5_chat_answer(context_block, history, new_question, artist=None, image_bytes=None):
#     system_prompt = "You are an art + fashion design mentor. Help the user with palettes, shades, and design advice."
#     if artist:
#         system_prompt += f" Explain how {artist} would apply these colors in real compositions and why."

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": context_block},
#     ] + history

#     if image_bytes:
#         img_b64 = base64.b64encode(image_bytes).decode("utf-8")
#         messages.append({
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": new_question},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ]
#         })
#     else:
#         messages.append({"role": "user", "content": new_question})

#     resp = client.chat.completions.create(
#         model=TEXT_MODEL,
#         messages=messages,
#         temperature=0.8,
#         max_tokens=600,
#     )

#     # ✅ Correct parsing of response
#     if hasattr(resp.choices[0], "message"):
#         content = resp.choices[0].message.get("content")
#     else:
#         content = resp.choices[0].delta.get("content", "")

#     return content

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

# # ---------------- NEW FEATURE: Mockup Preview ----------------
# def render_mockup(hex_list):
#     if not hex_list: return
#     st.markdown("### 🖼️ Sample Mockup Preview")
#     mockup = f"""
#     <div style="border-radius:12px; padding:20px; background:{hex_list[0]}; color:{hex_list[-1]}; font-family:sans-serif; margin-top:10px;">
#         <h2 style="margin:0;">PaletteGenie+ Demo Card</h2>
#         <p style="margin:0;">Primary: {hex_list[0]} | Accent: {hex_list[1]} | Neutral: {hex_list[-1]}</p>
#         <button style="margin-top:10px; padding:6px 12px; background:{hex_list[1]}; border:none; border-radius:6px; color:white;">Action</button>
#     </div>
#     """
#     st.markdown(mockup, unsafe_allow_html=True)

# # ---------------- NEW FEATURE: Refinement ----------------
# if "refined_palette" not in st.session_state:
#     st.session_state.refined_palette = None

# def refine_palette_request(user_msg, palette):
#     refine_prompt = f"The current palette is {palette}. User wants: {user_msg}. Suggest a new refined palette with HEX codes, and explain why."
#     resp = gpt5_chat_answer(
#         f"REFINE_PALETTE={palette}",
#         st.session_state.chat_history,
#         refine_prompt,
#         artist=None
#     )
#     return resp

# # ------------------ UI ------------------
# st.sidebar.header("🎨 Features")
# mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Enter HEX Colors"])

# artist = st.sidebar.selectbox(
#     "👩‍🎨 Choose an Artist Mentor:",
#     ["None", "Picasso", "Van Gogh", "Frida Kahlo", "Monet", "Salvador Dalí"]
# )

# translation_mode = st.sidebar.selectbox(
#     "🌍 Translate Palette Into:",
#     ["None", "Music Mood", "Fashion Style", "Interior Design"]
# )

# hex_palette = []
# uploaded_image_bytes = None

# # Upload or enter colors
# if mode == "Upload Image":
#     uploaded = st.file_uploader("Upload an artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         image = Image.open(uploaded).convert("RGB")
#         st.image(image, caption="Your Uploaded Artwork", width=300)
#         uploaded_image_bytes = uploaded.getvalue()

#         with st.spinner("🎨 Extracting main palette..."):
#             hex_palette = extract_palette(uploaded_image_bytes)
#         st.subheader("Extracted Palette")
#         render_palette_boxes(hex_palette)
#         render_mockup(hex_palette)

# elif mode == "Enter HEX Colors":
#     user_colors = st.text_input("Enter HEX colors (comma-separated, e.g. #FF5733, #33FFCE, #112233)")
#     if user_colors:
#         hex_palette = [c.strip() for c in user_colors.split(",") if c.strip()]
#         st.subheader("Your Custom Palette")
#         render_palette_boxes(hex_palette)
#         render_mockup(hex_palette)

# # ------------------ Cross-Domain Translator ------------------
# if translation_mode != "None" and hex_palette:
#     trans_prompt = f"Translate this palette {hex_palette} into {translation_mode}. Give a vivid, practical description."
#     translation = gpt5_chat_answer("PALETTE_TRANSLATION", [], trans_prompt)
#     st.subheader(f"🎭 Palette as {translation_mode}")
#     st.markdown(translation)

# # ------------------ Chat Section ------------------
# if hex_palette or uploaded_image_bytes:
#     context_block = f"PALETTE={hex_palette}" if hex_palette else "User uploaded an artwork."

#     st.subheader("💬 Chat with GPT-5 about Art, Colors & Design")
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_msg = st.chat_input("Ask for critique, style advice, refinements, or new shades...")
#     if user_msg:
#         with st.chat_message("user"):
#             st.markdown(user_msg)

#         try:
#             if "refine" in user_msg.lower() and hex_palette:
#                 answer = refine_palette_request(user_msg, hex_palette)
#             else:
#                 answer = gpt5_chat_answer(
#                     context_block,
#                     st.session_state.chat_history,
#                     user_msg,
#                     artist=artist if artist != "None" else None,
#                     image_bytes=uploaded_image_bytes
#                 )
#         except Exception as e:
#             answer = f"⚠️ Sorry, I couldn't get an answer: {e}"

#         st.session_state.chat_history.append({"role": "user", "content": user_msg})
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})

#         with st.chat_message("assistant"):
#             st.markdown(answer)

# else:
#     st.info("👉 Upload an image OR enter your own HEX colors to start exploring palettes with GPT-5.")

# Latest Code

# import streamlit as st
# import os, io, hashlib, base64, requests
# from colorthief import ColorThief
# from PIL import Image
# import webcolors

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="PaletteGenie+", page_icon="🎨", layout="wide")
# st.title("🎨 PaletteGenie+ (GPT-5) — Art Mentor, Palette, & Style Guide")

# # ----------------------------
# # Setup AIML API Client (✅ Hackathon compatible)
# # ----------------------------
# AIML_BASE_URL = "https://api.aimlapi.com/v1"
# TEXT_MODEL = "gpt-5-chat-latest"   # GPT-5 chat model
# AIML_API_KEY = st.secrets["AIML_API_KEY"]  # put this in .streamlit/secrets.toml

# # ------------------ HELPERS ------------------
# def rgb_to_hex(rgb):
#     r, g, b = rgb
#     return f"#{r:02x}{g:02x}{b:02x}"

# def hex_to_rgb(hex_color: str):
#     hex_color = hex_color.lstrip('#')
#     return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# def extract_palette(img_bytes: bytes, n_colors: int = 6):
#     bio = io.BytesIO(img_bytes)
#     ct = ColorThief(bio)
#     palette = ct.get_palette(color_count=n_colors)
#     return [rgb_to_hex(c) for c in palette]

# def closest_paint_name(requested_hex):
#     css3_names = {name: webcolors.name_to_hex(name) for name in webcolors.CSS3_NAMES_TO_HEX.keys()}
#     requested_rgb = webcolors.hex_to_rgb(requested_hex)
#     min_diff = float("inf")
#     closest_name = None

#     for name, hex_val in css3_names.items():
#         r, g, b = webcolors.hex_to_rgb(hex_val)
#         diff = (r - requested_rgb[0]) ** 2 + (g - requested_rgb[1]) ** 2 + (b - requested_rgb[2]) ** 2
#         if diff < min_diff:
#             min_diff = diff
#             closest_name = name

#     return closest_name

# # ✅ FIXED FUNCTION (uses requests to call AI/ML API)
# def gpt5_chat_answer(context_block, history, new_question, artist=None, image_bytes=None):
#     system_prompt = "You are an art + fashion design mentor. Help the user with palettes, shades, and design advice."
#     if artist:
#         system_prompt += f" Explain how {artist} would apply these colors in real compositions and why."

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": context_block},
#     ] + history

#     if image_bytes:
#         img_b64 = base64.b64encode(image_bytes).decode("utf-8")
#         messages.append({
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": new_question},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
#             ]
#         })
#     else:
#         messages.append({"role": "user", "content": new_question})

#     try:
#         resp = requests.post(
#             f"{AIML_BASE_URL}/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {AIML_API_KEY}",
#                 "Content-Type": "application/json",
#             },
#             json={
#                 "model": TEXT_MODEL,
#                 "messages": messages,
#                 "temperature": 0.8,
#                 "max_tokens": 600,
#             },
#             timeout=60
#         )
#         resp.raise_for_status()
#         data = resp.json()
#         return data["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"⚠️ API call failed: {e}"

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

# # ---------------- NEW FEATURE: Mockup Preview ----------------
# def render_mockup(hex_list):
#     if not hex_list: return
#     st.markdown("### 🖼️ Sample Mockup Preview")
#     mockup = f"""
#     <div style="border-radius:12px; padding:20px; background:{hex_list[0]}; color:{hex_list[-1]}; font-family:sans-serif; margin-top:10px;">
#         <h2 style="margin:0;">PaletteGenie+ Demo Card</h2>
#         <p style="margin:0;">Primary: {hex_list[0]} | Accent: {hex_list[1]} | Neutral: {hex_list[-1]}</p>
#         <button style="margin-top:10px; padding:6px 12px; background:{hex_list[1]}; border:none; border-radius:6px; color:white;">Action</button>
#     </div>
#     """
#     st.markdown(mockup, unsafe_allow_html=True)

# # ---------------- NEW FEATURE: Refinement ----------------
# if "refined_palette" not in st.session_state:
#     st.session_state.refined_palette = None

# def refine_palette_request(user_msg, palette):
#     refine_prompt = f"The current palette is {palette}. User wants: {user_msg}. Suggest a new refined palette with HEX codes, and explain why."
#     resp = gpt5_chat_answer(
#         f"REFINE_PALETTE={palette}",
#         st.session_state.chat_history,
#         refine_prompt,
#         artist=None
#     )
#     return resp

# # ------------------ UI ------------------
# st.sidebar.header("🎨 Features")
# mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Enter HEX Colors"])

# artist = st.sidebar.selectbox(
#     "👩‍🎨 Choose an Artist Mentor:",
#     ["None", "Picasso", "Van Gogh", "Frida Kahlo", "Monet", "Salvador Dalí"]
# )

# translation_mode = st.sidebar.selectbox(
#     "🌍 Translate Palette Into:",
#     ["None", "Music Mood", "Fashion Style", "Interior Design"]
# )

# hex_palette = []
# uploaded_image_bytes = None

# # Upload or enter colors
# if mode == "Upload Image":
#     uploaded = st.file_uploader("Upload an artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded:
#         image = Image.open(uploaded).convert("RGB")
#         st.image(image, caption="Your Uploaded Artwork", width=300)
#         uploaded_image_bytes = uploaded.getvalue()

#         with st.spinner("🎨 Extracting main palette..."):
#             hex_palette = extract_palette(uploaded_image_bytes)
#         st.subheader("Extracted Palette")
#         render_palette_boxes(hex_palette)
#         render_mockup(hex_palette)

# elif mode == "Enter HEX Colors":
#     user_colors = st.text_input("Enter HEX colors (comma-separated, e.g. #FF5733, #33FFCE, #112233)")
#     if user_colors:
#         hex_palette = [c.strip() for c in user_colors.split(",") if c.strip()]
#         st.subheader("Your Custom Palette")
#         render_palette_boxes(hex_palette)
#         render_mockup(hex_palette)

# # ------------------ Cross-Domain Translator ------------------
# if translation_mode != "None" and hex_palette:
#     trans_prompt = f"Translate this palette {hex_palette} into {translation_mode}. Give a vivid, practical description."
#     translation = gpt5_chat_answer("PALETTE_TRANSLATION", [], trans_prompt)
#     st.subheader(f"🎭 Palette as {translation_mode}")
#     st.markdown(translation)

# # ------------------ Chat Section ------------------
# if hex_palette or uploaded_image_bytes:
#     context_block = f"PALETTE={hex_palette}" if hex_palette else "User uploaded an artwork."

#     st.subheader("💬 Chat with GPT-5 about Art, Colors & Design")
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     user_msg = st.chat_input("Ask for critique, style advice, refinements, or new shades...")
#     if user_msg:
#         with st.chat_message("user"):
#             st.markdown(user_msg)

#         try:
#             if "refine" in user_msg.lower() and hex_palette:
#                 answer = refine_palette_request(user_msg, hex_palette)
#             else:
#                 answer = gpt5_chat_answer(
#                     context_block,
#                     st.session_state.chat_history,
#                     user_msg,
#                     artist=artist if artist != "None" else None,
#                     image_bytes=uploaded_image_bytes
#                 )
#         except Exception as e:
#             answer = f"⚠️ Sorry, I couldn't get an answer: {e}"

#         st.session_state.chat_history.append({"role": "user", "content": user_msg})
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})

#         with st.chat_message("assistant"):
#             st.markdown(answer)

# else:
#     st.info("👉 Upload an image OR enter your own HEX colors to start exploring palettes with GPT-5.")

import streamlit as st
import requests
from colorthief import ColorThief
from PIL import Image
import io, os
import webcolors

# -------------------------------
# Initialize Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Helper Functions
# -------------------------------
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def closest_paint_name(requested_color):
    try:
        # Convert RGB to HEX
        requested_hex = rgb_to_hex(requested_color)
        # Try to get exact CSS3 name
        return webcolors.hex_to_name(requested_hex, spec="css3")
    except ValueError:
        # If not exact, find the closest by Euclidean distance
        min_diff = float("inf")
        closest_name = None
        for name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
            diff = (r_c - requested_color[0]) ** 2 + (g_c - requested_color[1]) ** 2 + (b_c - requested_color[2]) ** 2
            if diff < min_diff:
                min_diff = diff
                closest_name = name
        return closest_name

def extract_palette(image_file, num_colors=6):
    color_thief = ColorThief(image_file)
    palette = color_thief.get_palette(color_count=num_colors)
    hex_palette = [rgb_to_hex(color) for color in palette]
    named_palette = [closest_paint_name(color) for color in palette]
    return hex_palette, named_palette

def get_gpt_response(prompt, api_key, model="gpt-5-chat-latest"):
    url = "https://api.aimlapi.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are PaletteGenie, a color palette assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ API Error: {response.status_code} - {response.text}"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎨 PaletteGenie")
st.write("Upload an image or input HEX codes to generate & refine color palettes.")

api_key = st.secrets["AIML_API_KEY"]

mode = st.radio("Choose input mode:", ["Upload Image", "Enter HEX codes"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ fixed deprecation

        # Extract Palette
        hex_palette, named_palette = extract_palette(uploaded_file)

        st.subheader("🎨 Extracted Palette")
        cols = st.columns(len(hex_palette))
        for i, hex_code in enumerate(hex_palette):
            with cols[i]:
                st.color_picker(named_palette[i], hex_code, disabled=True)

        st.write("**Palette Names:**", ", ".join(named_palette))

        # Ask GPT for suggestions
        prompt = f"Suggest 3 color palette names and uses for this palette: {hex_palette} ({named_palette})."
        gpt_response = get_gpt_response(prompt, api_key)
        st.subheader("✨ PaletteGenie Suggestions")
        st.write(gpt_response)

elif mode == "Enter HEX codes":
    hex_input = st.text_area("Enter HEX codes separated by commas (e.g. #FF5733, #33FF57, #3357FF)")
    if hex_input:
        hex_palette = [code.strip() for code in hex_input.split(",")]
        st.subheader("🎨 Custom Palette")
        cols = st.columns(len(hex_palette))
        for i, hex_code in enumerate(hex_palette):
            with cols[i]:
                st.color_picker(f"Color {i+1}", hex_code, disabled=True)

        # Ask GPT
        prompt = f"Suggest names, themes, and creative uses for this custom palette: {hex_palette}"
        gpt_response = get_gpt_response(prompt, api_key)
        st.subheader("✨ PaletteGenie Suggestions")
        st.write(gpt_response)

# -------------------------------
# Refinement Chat
# -------------------------------
st.subheader("💬 Refine Your Palette with AI")

user_message = st.text_input("Type a refinement instruction (e.g., 'make it more pastel')")
if st.button("Refine Palette") and user_message:
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    # Send chat history to GPT
    url = "https://api.aimlapi.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-5-chat-latest",
        "messages": [{"role": "system", "content": "You are PaletteGenie, a color palette refinement assistant."}] + st.session_state.chat_history
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        gpt_reply = response.json()["choices"][0]["message"]["content"]
        st.session_state.chat_history.append({"role": "assistant", "content": gpt_reply})
    else:
        st.error(f"⚠️ API Error: {response.status_code} - {response.text}")

# Display chat
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 PaletteGenie:** {msg['content']}")
