import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import time
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import io

# Streamlit page configuration
st.set_page_config(page_title="Super Resolution - ESPCN", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'lpips_model' not in st.session_state:
    st.session_state.lpips_model = None
if 'input_image' not in st.session_state:
    st.session_state.input_image = None
if 'output_image' not in st.session_state:
    st.session_state.output_image = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Constants
SCALE_FACTOR = 4

# Register custom functions for Keras serialization
@tf.keras.utils.register_keras_serializable()
def pixel_shuffle(x):
    return tf.nn.depth_to_space(x, 4)

def log_message(message):
    """Add message to logs with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

def load_model():
    """Load trained ESPCN model"""
    try:
        model_path = "models/espcn_final.h5"
        if os.path.exists(model_path):
            st.session_state.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'pixel_shuffle': pixel_shuffle,
                    'mse': tf.keras.losses.MeanSquaredError()
                }
            )
            log_message("✅ ESPCN model loaded successfully")
            # Load LPIPS model
            st.session_state.lpips_model = lpips.LPIPS(net='alex')
            log_message("✅ LPIPS model loaded successfully")
        else:
            log_message("❌ Model file not found. Please run training first.")
            st.error("Model file not found. Please run training notebook first.")
    except Exception as e:
        log_message(f"❌ Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")

def process_image(input_image):
    """Process image with ESPCN model"""
    if input_image is None or st.session_state.model is None:
        st.error("Please upload an image and ensure model is loaded.")
        return None
    
    try:
        with st.spinner("Processing image with ESPCN..."):
            log_message("🔄 Processing image with ESPCN...")
            
            # Prepare input
            input_image = input_image.copy()
            
            # Resize to multiple of scale factor if needed
            h, w = input_image.shape[:2]
            new_h = (h // SCALE_FACTOR) * SCALE_FACTOR
            new_w = (w // SCALE_FACTOR) * SCALE_FACTOR
            
            if new_h != h or new_w != w:
                input_image = cv2.resize(input_image, (new_w, new_h))
                log_message(f"📏 Resized to {new_w}x{new_h} for processing")
            
            # Create low-res version
            lr_h, lr_w = new_h // SCALE_FACTOR, new_w // SCALE_FACTOR
            lr_image = cv2.resize(input_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            
            # Normalize
            lr_normalized = lr_image.astype(np.float32) / 255.0
            
            # Add batch dimension
            lr_batch = np.expand_dims(lr_normalized, axis=0)
            
            # Predict
            start_time = time.time()
            sr_batch = st.session_state.model.predict(lr_batch, verbose=0)
            inference_time = time.time() - start_time
            
            # Post-process
            sr_image = sr_batch[0]
            sr_image = np.clip(sr_image, 0, 1)
            sr_image = (sr_image * 255).astype(np.uint8)
            
            # Calculate metrics
            bicubic_upscaled = cv2.resize(lr_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            psnr_val = psnr(input_image, sr_image, data_range=255)
            ssim_val = ssim(input_image, sr_image, data_range=255, channel_axis=2)
            psnr_bicubic = psnr(input_image, bicubic_upscaled, data_range=255)
            ssim_bicubic = ssim(input_image, bicubic_upscaled, data_range=255, channel_axis=2)
            
            # Log results
            log_message("✅ Processing completed!")
            log_message(f"⏱ Inference time: {inference_time:.4f} seconds")
            log_message(f"📊 ESPCN vs Original:")
            log_message(f"   PSNR: {psnr_val:.4f} dB")
            log_message(f"   SSIM: {ssim_val:.4f}")
            log_message(f"📊 Bicubic vs Original:")
            log_message(f"   PSNR: {psnr_bicubic:.4f} dB")
            log_message(f"   SSIM: {ssim_bicubic:.4f}")
            
            improvement_psnr = psnr_val - psnr_bicubic
            improvement_ssim = ssim_val - ssim_bicubic
            log_message(f"📈 Improvement over Bicubic:")
            log_message(f"   PSNR: {improvement_psnr:+.4f} dB")
            log_message(f"   SSIM: {improvement_ssim:+.4f}")
            
            return sr_image
    
    except Exception as e:
        log_message(f"❌ Error during processing: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
        return None

# Main UI
st.title("Super Resolution dengan ESPCN")
st.markdown("Upload an image to enhance its resolution using the ESPCN model.")

# Load model on startup
if st.session_state.model is None:
    load_model()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file:
        # Load and display image
        try:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot read image file")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.input_image = image
            st.image(image, caption="Input Image", use_container_width=True)
            h, w = image.shape[:2]
            log_message(f"📁 Image loaded: {uploaded_file.name}")
            log_message(f"📏 Image size: {w}x{h} pixels")
        except Exception as e:
            log_message(f"❌ Error loading image: {str(e)}")
            st.error(f"Error loading image: {str(e)}")

with col2:
    st.subheader("Output Image")
    if st.session_state.output_image is not None:
        st.image(st.session_state.output_image, caption="Output Image", use_container_width=True)
    else:
        st.write("No output yet")

# Process button
if st.button("Process Image", disabled=(st.session_state.input_image is None)):
    output_image = process_image(st.session_state.input_image)
    if output_image is not None:
        st.session_state.output_image = output_image
        st.rerun()

# Download button
if st.session_state.output_image is not None:
    # Convert output image to bytes for download
    output_bgr = cv2.cvtColor(st.session_state.output_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", output_bgr)
    st.download_button(
        label="Save Result",
        data=buffer.tobytes(),
        file_name="super_resolution_output.jpg",
        mime="image/jpeg"
    )

# Metrics and Logs
st.subheader("Metrics & Information")
log_container = st.container()
with log_container:
    for log in st.session_state.logs:
        st.write(log)

# Footer
st.markdown("---")
st.markdown("Super Resolution GUI using ESPCN Model | Version 1.0")