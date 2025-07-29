import tensorflow as tf
from tensorflow.keras.layers import Input,  Conv2D, Conv2DTranspose, concatenate, Dropout, MaxPooling2D
import tensorflow as tf
import numpy as np
from PIL import Image


def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Builds a convolutional downsampling block for the U-Net encoder.

    This block applies two Conv2D layers with ReLU activation, optional dropout,
    and optional MaxPooling2D for spatial downsampling.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, H, W, C).
        n_filters (int): Number of filters for both Conv2D layers.
        dropout_prob (float): Dropout rate applied after Conv2D layers if > 0.
        max_pooling (bool): If True, applies MaxPooling2D with pool size = 2 and strides=2
                            to downsample the spatial dimensions.

    Returns:
        next_layer (tf.Tensor): 
            - If max_pooling is True: shape → (batch_size, H/2, W/2, n_filters)
            - Else: shape → (batch_size, H, W, n_filters)
        
        skip_connection (tf.Tensor): Output before pooling, used in skip connection.
            - shape → (batch_size, H, W, n_filters)
    """
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    
    if dropout_prob > 0:
        conv = Dropout(rate=dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=2)(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection




def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Builds an upsampling block for the U-Net decoder.

    Applies transpose convolution to upscale the input, concatenates it with
    the corresponding encoder output (skip connection), and applies two Conv2D layers.

    Args:
        expansive_input (tf.Tensor): Decoder input tensor, shape (B, H, W, C).
        contractive_input (tf.Tensor): Encoder skip connection tensor, shape (B, H, W, C).
        n_filters (int): Number of filters for the Conv2D layers.

    Returns:
        tf.Tensor: Output tensor of shape (B, H*2, W*2, n_filters).
    """
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    
    return conv



def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23, debug=False):
    """
    Builds a U-Net model for semantic segmentation.

    U-Net follows an encoder–decoder architecture:
    - The encoder path (contracting) applies repeated 3x3 Conv + ReLU + MaxPool to extract spatial features while reducing resolution.
    - The decoder path (expanding) upsamples the feature maps using transpose convolutions, and merges them with corresponding encoder features (skip connections) to recover spatial details.
    - A final 1x1 convolution maps the features to the desired number of output classes.

    Args:
        input_size (tuple): Shape of the input image (height, width, channels).
        n_filters (int): Number of filters for the first conv block (doubles at each level).
        n_classes (int): Number of segmentation output classes.

    Returns:
        tf.keras.Model: Compiled U-Net model.
    """

    inputs = Input(input_size)
    if debug: print("Input:", inputs.shape)

    cblock1 = conv_block(inputs, n_filters)
    if debug: print("After cblock1:", cblock1[0].shape, "(skip:", cblock1[1].shape, ")")

    cblock2 = conv_block(cblock1[0], n_filters*2)
    if debug: print("After cblock2:", cblock2[0].shape, "(skip:", cblock2[1].shape, ")")

    cblock3 = conv_block(cblock2[0], n_filters*4)
    if debug: print("After cblock3:", cblock3[0].shape, "(skip:", cblock3[1].shape, ")")

    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)
    if debug: print("After cblock4:", cblock4[0].shape, "(skip:", cblock4[1].shape, ")")

    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)
    if debug: print("After cblock5 (bottom):", cblock5[0].shape)

    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    if debug: print("After ublock6:", ublock6.shape)

    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters*4)
    if debug: print("After ublock7:", ublock7.shape)

    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters*2)
    if debug: print("After ublock8:", ublock8.shape)

    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)
    if debug: print("After ublock9:", ublock9.shape)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    if debug: print("After conv9:", conv9.shape)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    if debug: print("Final Output:", conv10.shape)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model





class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, ay_pred, sample_weight=None):
        # Apply argmax to predictions (convert logits to class predictions)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def preprocess_image(image: Image.Image, img_width: int, img_height: int) -> np.ndarray:
    """
    Resizes the PIL image and normalizes it for model input.

    Args:
        image (PIL.Image.Image): The input image as a PIL Image object.
        img_width (int): The target width for the image.
        img_height (int): The target height for the image.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array, ready for model prediction.
                    Shape: (1, img_height, img_width, 3)
    """
    img_array = image.resize((img_width, img_height))
    img_array = np.array(img_array).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def create_mask_display(pred_mask: np.ndarray, num_classes: int, img_width: int, img_height: int) -> Image.Image:
    """
    Converts model output (logits) to a displayable segmentation mask with distinct colors.

    Args:
        pred_mask (np.ndarray): The raw prediction output from the U-Net model.
                                Expected shape: (1, H, W, N_CLASSES) where N_CLASSES is
                                the number of segmentation classes.
        num_classes (int): The total number of segmentation classes the model predicts.
        img_width (int): The width of the output mask (should match model input width).
        img_height (int): The height of the output mask (should match model input height).

    Returns:
        PIL.Image.Image: A colored segmentation mask as a PIL Image object, suitable for display.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1) 
    if pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0] 
    mask_array = pred_mask.numpy().astype(np.uint8) 
    colors = []
    for i in range(num_classes):
        hue = (i * 137) % 256 
        r = (hue * 17) % 256
        g = (hue * 23) % 256
        b = (hue * 31) % 256
        colors.append((r, g, b))

    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        if class_id in np.unique(mask_array):
            colored_mask[mask_array == class_id] = colors[class_id]
    return Image.fromarray(colored_mask)

