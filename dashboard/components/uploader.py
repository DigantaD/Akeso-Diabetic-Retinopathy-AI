import streamlit as st
from zipfile import ZipFile
from io import BytesIO
from PIL import Image

def upload_images():
    uploaded = st.file_uploader(
        "Upload .jpg/.png or a .zip",
        type=["jpg", "png", "zip"],
        accept_multiple_files=False
    )
    images = []

    if uploaded:
        if uploaded.name.endswith(".zip"):
            with ZipFile(uploaded) as z:
                for name in z.namelist():
                    if name.lower().endswith((".jpg", ".png")):
                        with z.open(name) as img_file:
                            img = Image.open(img_file).convert("RGB")
                            images.append((name, img))
        else:
            img = Image.open(uploaded).convert("RGB")
            images.append((uploaded.name, img))

        st.success(f"✅ {len(images)} image(s) ready for analysis.")
    
    return images