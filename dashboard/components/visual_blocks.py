import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from dashboard.llm.report_generator import generate_llm_report

def display_visuals(file_data, user_role="Patient", show_legend=True, inference_results=None):
    name, image = file_data
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"**🖼️ Original Image: `{name}`**")
        st.image(image, caption="Original Retinal Image", use_container_width=True)

    # Default fallbacks
    grade_label = "Unknown"
    lesion_summary = "Not detected"
    region = "Unknown"

    with col2:
        # 🎯 Disease Grading
        if inference_results and "grading" in inference_results:
            logits = inference_results["grading"]
            pred_idx = int(np.argmax(logits))
            confidence = float(logits[pred_idx]) * 100
            label_map = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
            grade_label = label_map[pred_idx]
            st.markdown(f"🎯 **Disease Grade**: *{grade_label} ({confidence:.1f}%)*")
        else:
            st.markdown("🎯 **Disease Grade**: *Unavailable*")

        # 🧠 Segmentation Overlay
        if inference_results and "segmentation" in inference_results:
            seg_mask = inference_results["segmentation"][0]
            seg_img = render_segmentation_overlay(image, seg_mask)
            st.image(seg_img, caption="Lesion Segmentation Overlay", use_container_width=True)
            lesion_labels = ["Soft Exudates", "Microaneurysms", "Hard Exudates", "Hemorrhages", "Optic Disc"]
            present_lesions = []

            for i, label in enumerate(lesion_labels[:-1]):  # Skip optic disc
                lesion_area = (seg_mask[i] > 0.5).sum()
                if lesion_area > 100:  # Threshold to ignore noise
                    present_lesions.append(label)

            lesion_summary = ", ".join(present_lesions) if present_lesions else "No prominent lesions"
        else:
            st.markdown("🧠 **Lesion Overlay**: *Unavailable*")

   # 📍 Localization (Clinician-only)
    if user_role == "Clinician" and inference_results and "localization" in inference_results:
        loc_img = image.copy()
        draw = ImageDraw.Draw(loc_img)
        fovea, disc = inference_results["localization"]
        draw.ellipse((fovea[0]-5, fovea[1]-5, fovea[0]+5, fovea[1]+5), fill='red')
        draw.ellipse((disc[0]-5, disc[1]-5, disc[0]+5, disc[1]+5), fill='blue')
        st.image(loc_img, caption="Fovea (🔴) and Disc (🔵) Localization", use_container_width=True)

        # 📍 Determine anatomical region
        dx = fovea[0] - disc[0]
        dy = fovea[1] - disc[1]

        region = []
        if abs(dx) > 20:  # Threshold in pixels
            region.append("temporal" if dx > 0 else "nasal")
        if abs(dy) > 20:
            region.append("superior" if dy < 0 else "inferior")

        if region:
            region = " and ".join(region) + " region"
        else:
            region = "central region"

    # 🟡 Legend
    if show_legend:
        st.markdown("**Overlay Legend**: 🟢 SE, 🔴 MA, 🟡 HE, 🔵 Disc, 🔴 Fovea, 🔵 Optic Disc")

    # 💬 GPT-based Role Report
    with st.expander("📄 LLM Report", expanded=True):
        st.markdown(f"*(Role: {user_role})*")
        with st.spinner("Generating medical explanation..."):
            llm_report = generate_llm_report(
                user_role=user_role,
                grade=grade_label,
                lesions=lesion_summary,
                location=region
            )
            st.markdown(llm_report)

    st.markdown("---")

def render_segmentation_overlay(image, seg_mask):
    # Handle 2D or 3D masks
    if len(seg_mask.shape) == 3:
        _, h, w = seg_mask.shape
    elif len(seg_mask.shape) == 2:
        h, w = seg_mask.shape
    else:
        raise ValueError(f"Unsupported seg_mask shape: {seg_mask.shape}")

    base = image.resize((w, h)).convert("RGBA")

    # Generate color overlay per class
    color_map = [
        (0, 0, 0, 0),         # Background
        (255, 0, 0, 100),     # Microaneurysms
        (0, 255, 0, 100),     # Haemorrhages
        (0, 0, 255, 100),     # Hard Exudates
        (255, 255, 0, 100),   # Soft Exudates
        (255, 165, 0, 100),   # Optic Disc
    ]

    overlay = Image.new("RGBA", (w, h))
    for i in range(1, min(len(color_map), seg_mask.shape[0] if len(seg_mask.shape) == 3 else 1)):
        mask = seg_mask[i] if len(seg_mask.shape) == 3 else (seg_mask == i)
        color_layer = Image.new("RGBA", (w, h), color_map[i])
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        overlay = Image.composite(color_layer, overlay, mask_img)

    return Image.alpha_composite(base, overlay)