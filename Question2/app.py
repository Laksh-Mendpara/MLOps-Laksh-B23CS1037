import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# Local imports
from mymodel import UNet

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Question2/final_model.pth"
PLOT_PATH = "Question2/training_curves.png"
NUM_CLASSES = 23

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    """Loads the UNet model with the trained weights."""
    model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        # Fix: map_location and weights_only for security/compatibility
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """Resizes and normalizes input image for the model."""
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

def preprocess_mask(mask):
    """Resizes ground truth mask for visualization."""
    mask = np.array(mask.convert('L'))
    mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
    return mask

# --- APP LAYOUT ---
st.set_page_config(page_title="UNet Segmentation Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Training Analytics", "Model Inference"])

# --- PAGE 1: TRAINING ANALYTICS ---
if page == "Training Analytics":
    st.title("📊 Training Performance Analytics")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Progress Curves")
        if os.path.exists(PLOT_PATH):
            st.image(PLOT_PATH, caption="Loss, mIOU, and mDICE over 15 Epochs", use_container_width=True)
        else:
            st.error(f"Plot not found at {PLOT_PATH}. Please ensure the training script generated the image.")

    with col2:
        st.subheader("Final Test Set Metrics")
        st.write("Results obtained after 15 epochs of training:")
        st.metric(label="mIOU Score", value="0.9443", delta="Excellent")
        st.metric(label="mDICE Score", value="0.9663", delta="Excellent")
        
        st.info("**Threshold Check:** Both metrics are significantly higher than the required 0.48.")

# --- PAGE 2: MODEL INFERENCE ---
elif page == "Model Inference":
    st.title("🧠 UNet Model Inference")
    st.write("Upload 4 test images and their matching ground-truth masks to see the model in action.")
    st.markdown("---")

    model = load_model()

    # Uploaders
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        uploaded_imgs = st.file_uploader("Upload 4 Images", type=['png', 'jpg'], accept_multiple_files=True)
    with up_col2:
        uploaded_masks = st.file_uploader("Upload 4 Corresponding Masks", type=['png', 'jpg'], accept_multiple_files=True)

    if uploaded_imgs and uploaded_masks:
        if len(uploaded_imgs) == 4 and len(uploaded_masks) == 4:
            # Sort files by name to ensure pairs match
            uploaded_imgs.sort(key=lambda x: x.name)
            uploaded_masks.sort(key=lambda x: x.name)

            st.subheader("Segmentation Results Comparison")
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            column_titles = ["Input Image", "Ground Truth Mask", "Model Prediction"]
            
            for i in range(4):
                # Load images
                img_pil = Image.open(uploaded_imgs[i])
                mask_pil = Image.open(uploaded_masks[i])
                
                # Preprocess and Predict
                input_tensor = preprocess_image(img_pil)
                gt_mask = preprocess_mask(mask_pil)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # Plotting Row i
                # 1. Original Image
                axes[i, 0].imshow(img_pil)
                # 2. Ground Truth
                axes[i, 1].imshow(gt_mask, cmap='tab20')
                # 3. Prediction
                axes[i, 2].imshow(pred_mask, cmap='tab20')
                
                # Formatting
                for j in range(3):
                    axes[i, j].axis('off')
                    if i == 0:
                        axes[i, j].set_title(column_titles[j], fontsize=16, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please upload exactly 4 images and 4 masks.")
    else:
        st.info("Awaiting file uploads...")