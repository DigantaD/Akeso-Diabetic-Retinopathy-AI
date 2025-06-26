import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from dashboard.llm.report_generator import generate_llm_report

def display_visuals(file_data, user_role="Patient", show_legend=True, inference_results=None):
    name, image = file_data
    st.markdown(f"**🖼️ Original Image: `{name}`**")
    st.image(image, caption="Original Retinal Image", use_container_width=True)

    grade_label = "Unknown"
    lesion_summary = "Not detected"
    region = "Unknown"

    # 🎯 DISEASE GRADING
    if inference_results and "grading" in inference_results:
        pred_label, confidence = inference_results["grading"]
        grade_label = pred_label
        st.markdown(f"🎯 **Disease Grade**: *{pred_label} ({confidence:.1f}%)*")
    else:
        st.markdown("🎯 **Disease Grade**: *Unavailable*")

    # 🧠 SEGMENTATION COMPOSITE OVERLAY
    if inference_results and "segmentation" in inference_results:
        lesion_masks = inference_results["segmentation"]
        present_lesions = [label for label, mask in lesion_masks.items() if np.sum(mask) > 100]

        seg_img = render_segmentation_overlay(image, lesion_masks)
        st.image(seg_img, caption="Combined Lesion Segmentation Overlay", use_container_width=True)

        lesion_summary = ", ".join(present_lesions) if present_lesions else "No prominent lesions"
    else:
        st.markdown("🧠 **Lesion Masks**: *Unavailable*")

    # 📍 LOCALIZATION
    if inference_results and "localization" in inference_results:
        st.markdown("📍 **Localization**")

        if "points" in inference_results["localization"]:
            points = inference_results["localization"]["points"]
            loc_img = image.copy()
            draw = ImageDraw.Draw(loc_img)
            if "fovea" in points:
                f = points["fovea"]
                draw.ellipse((f[0]-5, f[1]-5, f[0]+5, f[1]+5), fill='red')
            if "optic_disc" in points:
                d = points["optic_disc"]
                draw.ellipse((d[0]-5, d[1]-5, d[0]+5, d[1]+5), fill='blue')
            st.image(loc_img, caption="Fovea (🔴) and Disc (🔵) Overlay", use_container_width=True)

            dx = points["fovea"][0] - points["optic_disc"][0]
            dy = points["fovea"][1] - points["optic_disc"][1]
            region = []
            if abs(dx) > 20:
                region.append("temporal" if dx > 0 else "nasal")
            if abs(dy) > 20:
                region.append("superior" if dy < 0 else "inferior")
            region = " and ".join(region) + " region" if region else "central region"

        elif "heatmaps" in inference_results["localization"]:
            for label, heatmap in inference_results["localization"]["heatmaps"].items():
                st.image(heatmap, caption=f"{label} Heatmap", use_container_width=True)

    if show_legend:
        st.markdown("**Legend**: 🧠 Lesions - 🟡 SE, 🔴 MA, 🔵 HE, 🟢 HR, 🟠 Disc | 📍 Fovea (🔴), Optic Disc (🔵)")

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


def render_segmentation_overlay(image, mask_dict):
    w, h = image.size
    base = image.convert("RGBA").resize((w, h))

    color_map = {
        "Soft Exudates": (255, 255, 0, 120),     # 🟡 Yellow
        "Microaneurysms": (255, 0, 0, 120),      # 🔴 Red
        "Hard Exudates": (0, 0, 255, 120),       # 🔵 Blue
        "Hemorrhages": (0, 255, 0, 120),         # 🟢 Green
        "Optic Disc": (255, 165, 0, 120),        # 🟠 Orange
    }

    overlay = Image.new("RGBA", (w, h))

    for label, mask in mask_dict.items():
        if np.sum(mask) < 100:
            continue
        color = color_map.get(label, (255, 255, 255, 120))
        color_layer = Image.new("RGBA", (w, h), color)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h)).convert("L")
        overlay = Image.composite(color_layer, overlay, mask_img)

    return Image.alpha_composite(base, overlay)