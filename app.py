import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import random


# --- Streamlit App Layout ---
st.set_page_config(
    page_title="PixelSense AI",
    page_icon="🤖",
    layout="wide", # Use wide layout for better side-by-side display
    initial_sidebar_state="expanded"
)

# U-Net-driven Semantic Image Segmentation
from model import unet_model, CustomMeanIoU, preprocess_image, create_mask_display

# --- Model Configuration ---
IMG_HEIGHT = 96
IMG_WIDTH = 128
NUM_CHANNELS = 3
NUM_CLASSES = 23

EXISTING_IMAGES_DIR = "data/CameraRGB"

@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_segmentation_model():
    """
    Loads the pre-trained U-Net model weights.
    This function is cached to prevent re-loading the model on every Streamlit rerun.
    """
    model = unet_model((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), n_classes=NUM_CLASSES, debug=False)
        
    try:
        message_placeholder = st.empty()
        model.load_weights("model_weights.h5")
        message_placeholder.success("Model loaded successfully!")
        time.sleep(3)
        message_placeholder.empty()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.info("Please ensure 'model_weights.h5' is in the same directory as 'app.py'.")
    return model

unet_model_loaded = load_segmentation_model()


# --- Header Section ---
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.title("✨ PixelSense AI ✨")
st.markdown("""
    Welcome to this interactive demonstration of semantic image segmentation using a U-Net model!
    Upload an image, or select from our curated examples, and our deep learning model will automatically identify and outline
    different objects and regions within it.
""")
st.markdown("---")

st.subheader("Choose Your Image Source")

# Radio button to select input method
image_source_option = st.radio(
    "How would you like to provide an image?",
    ("Upload an image", "Select from existing examples"),
    index=0
)

original_image_pil = None

if image_source_option == "Upload an image":
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG, or WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image to see it segmented by the U-Net model."
    )
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB") # Ensure RGB format
elif image_source_option == "Select from existing examples":
    st.markdown("---")
    st.subheader("Select from Existing Images")
    
    # Check if the directory exists
    if os.path.isdir(EXISTING_IMAGES_DIR):
        all_image_files = [f for f in os.listdir(EXISTING_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not all_image_files:
            st.warning(f"No image files found in the directory: `{EXISTING_IMAGES_DIR}`. Please ensure images are present.")
        else:
            # Select up to 20 random images
            num_images_to_sample = min(20, len(all_image_files))
            random_selected_files = random.sample(all_image_files, num_images_to_sample)
            
            selected_filename = st.selectbox(
                "Choose an image from the random selection:",
                random_selected_files,
                help="These are random images from the 'data/CameraRGB' directory."
            )
            
            if selected_filename:
                image_path = os.path.join(EXISTING_IMAGES_DIR, selected_filename)
                try:
                    original_image_pil = Image.open(image_path).convert("RGB")
                    st.success(f"Selected image: `{selected_filename}`")
                except Exception as e:
                    st.error(f"Error loading selected image `{selected_filename}`: {e}")
    else:
        st.error(f"The directory `{EXISTING_IMAGES_DIR}` was not found. Please ensure it exists and contains images.")

st.markdown("---")

# --- Segmentation Processing and Display ---
if original_image_pil is not None:
    st.markdown("### Segmentation Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image_pil, caption="Original Image")

    with st.spinner("🚀 Segmenting image... Please wait! This is where the U-Net's 'brain' is working."):
        try:
            processed_image = preprocess_image(original_image_pil, IMG_WIDTH, IMG_HEIGHT)
            print(f"Processed image shape: {processed_image.shape}") 
            
            prediction = unet_model_loaded.predict(processed_image, verbose=0)
            print(f"Prediction shape: {prediction.shape}")  

            segmented_mask_pil = create_mask_display(prediction, NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT)

            with col2:
                st.image(segmented_mask_pil, caption="Segmented Mask", use_column_width=True)
                st.info("Each color in the segmented mask represents a different identified class.")

            st.success("Segmentation complete! Check out the results above.")

        except Exception as e:
            st.error(f"An error occurred during segmentation: {e}")
            st.info("Please ensure the selected/uploaded image is valid and the model weights are correctly loaded.")
else:
    st.info("Please upload an image or select an example to get started with the segmentation!")

st.markdown("---")

# --- How It Works Section (Updated Content) ---
st.subheader("💡 How It Works: Demystifying the U-Net")
with st.expander("Click to learn about U-Net and this App's process"):
    st.markdown("""
    This application uses a powerful type of Convolutional Neural Network (CNN) called **U-Net**
    for **semantic image segmentation**. Think of it as teaching a computer to "see" and
    "understand" what's in an image at a pixel level, assigning a specific category (like "car," "road," or "sky")
    to every single pixel.

    #### The U-Net Architecture: An Intuitive Analogy

    The U-Net gets its name from its 'U' shaped architecture, which is divided into two main paths:

    1.  **Encoder (The Contracting Path - Left Side of 'U')**:
        * **Purpose:** This part is like the model "squinting" at the image. It progressively
            reduces the spatial dimensions (height and width) of the image while increasing
            the number of feature channels. It learns to capture the "what" of the image
            (e.g., "there's a car here"), extracting increasingly abstract features.
        * **Process:** It consists of repeated **convolutional blocks**. Each block typically involves:
            * Two $3 *  3$ `Conv2D` layers, followed by a `ReLU` activation function. These layers learn to detect patterns and features. Our model uses `kernel_initializer='he_normal'` for robust weight initialization.
            * `Dropout` layers (used in deeper blocks, with $0.3$ probability) to prevent the model from memorizing the training data too much, helping it generalize better to new images.
            * A `MaxPooling2D` layer, which acts like "zooming out" by taking the maximum value in small regions, effectively halving the spatial dimensions (e.g., from $H , W$ to $H/2 , W/2$). This reduces computational load and helps capture larger-scale features.

    2.  **Decoder (The Expanding Path - Right Side of 'U')**:
        * **Purpose:** Now, the model "un-squints" and reconstructs the image, but this time with precise segmentation outlines. It takes the compact, high-level features from the encoder and progressively
            upsamples them back to the original image resolution. It learns the "where" of the objects, restoring spatial detail.
        * **Process:** This path uses:
            * `Conv2DTranspose` layers (also known as "deconvolution" or "upsampling convolution") to effectively "stretch" the feature maps, doubling their spatial dimensions (e.g., from $H/2 , W/2$ to $H , W$).
            * Following the upsampling, it applies two more $3 * 3$ `Conv2D` layers with `ReLU` activation to refine the features at the new resolution.

    3.  **Skip Connections (The Bridges of the 'U')**:
        * **Purpose:** This is the magic of the U-Net! These connections directly link feature maps
            from the encoder to the corresponding upsampled feature maps in the decoder.
        * **Why it's important:** When the encoder squints and compresses the image, it inevitably loses some fine-grained spatial information (like exact edges). The skip connections act like "bridges," bringing back these high-resolution, low-level features from the encoder directly to the decoder. By concatenating (`concatenate` layer) these features, the decoder can combine both the "what" (semantic information from deep layers) and the "where" (spatial
            information from shallow layers) to produce very precise and detailed segmentation masks.

    #### Visualizing the U-Net:

    """)
    # Assuming unet.png is in the root directory
    st.image("unet.png", caption="U-Net Architecture Overview", use_column_width=True)

    st.markdown(f"""
    #### How This App Works:
    1.  **Input Image:** You provide an image to the application, either by uploading it or selecting from existing examples.
    2.  **Pre-processing:** The image is automatically resized to the model's required input dimensions (${IMG_HEIGHT} \times {IMG_WIDTH}$ pixels)
        and normalized (pixel values scaled from $0-255$ to $0-1$). This ensures the image format is perfectly matched for the U-Net.
    3.  **Prediction:** The pre-processed image is fed into the loaded U-Net model.
        The model then processes the image through its encoder-decoder path, outputting a prediction for each pixel indicating which of the ${NUM_CLASSES}$
        categories it belongs to.
    4.  **Post-processing & Visualization:** The raw numerical predictions are converted into a
        colorful segmentation mask. Each detected class is assigned a unique color,
        making the segmentation visually clear.
    5.  **Display:** The original image and the segmented mask are displayed side-by-side
        for a direct and clear comparison.
    """)

# Moved Model Summary from sidebar to here
if st.checkbox("Show Model Summary (Developer Info)"):
    st.text("U-Net Model Summary:")
    from io import StringIO
    string_io = StringIO()
    unet_model_loaded.summary(print_fn=lambda x: string_io.write(x + '\n'))
    st.code(string_io.getvalue())

st.markdown("---")

# --- Sidebar Information ---
st.sidebar.header("About This Application")
st.sidebar.info(f"""
    This U-Net Image Segmentation App was developed by an aspiring AI/ML Engineer
    as a practical demonstration of deep learning in computer vision.

    **Key Technologies Used:**
    -   **TensorFlow/Keras:** For building and loading the U-Net model which is trained from scratch.
    -   **Streamlit:** For creating this interactive and user-friendly web interface.
    -   **PIL (Pillow) & NumPy:** For efficient image manipulation.

    **Model Specifications:**
    -   Input Image Size: ${IMG_HEIGHT} \times {IMG_WIDTH} \times {NUM_CHANNELS}$
    -   Number of Output Classes: ${NUM_CLASSES}$
    -   Model Weights: `model_weights.h5`

    This project aims to showcase robust image segmentation capabilities and is a step
    towards high-impact research publications and open-source contributions.
""")
