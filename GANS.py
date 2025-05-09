import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_models():
    try:
        generator = tf.keras.models.load_model("generator.keras")  
        discriminator = tf.keras.models.load_model("discriminator.keras")  
        return generator, discriminator
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure 'generator.keras' and 'discriminator.keras' are in the same directory.")
        return None, None

def generate_images(generator, num_images):
    noise = tf.random.normal([num_images, 100])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)
    return generated_images

def preprocess_image(image):
    image = image.resize((32, 32))
    image = image.convert("RGB")
    image_array = np.array(image, dtype=np.float32)
    image_array = (image_array - 127.5) / 127.5
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def classify_image(discriminator, image_array):
    logits = discriminator(image_array, training=False)
    probability = tf.sigmoid(logits).numpy()[0][0]
    label = "Real" if probability > 0.5 else "Fake"
    return label, probability


st.title("CIFAR-10 R3GAN: Image Generator and Real/Fake Classifier")
st.write("Use this app to generate CIFAR-10-like images or classify an uploaded image as Real (CIFAR-10) or Fake (generated).")


generator, discriminator = load_models()

if generator is None or discriminator is None:
    st.stop()

st.header("Generate CIFAR-10-like Images")
st.write("Select the number of images to generate (1–16) and click the button below.")
num_images = st.slider("Number of images to generate:", 1, 16, 4, key="generate_slider")
if st.button("Generate Images"):
    with st.spinner("Generating images..."):
        images = generate_images(generator, num_images)
    st.success(f"Generated {num_images} images!")
    cols = st.columns(4)
    for i, img in enumerate(images):
        with cols[i % 4]:
            st.image(img, caption=f"Generated Image {i+1}", use_container_width=True)


st.header("Classify Image: Real or Fake")
st.write("Upload a 32x32 RGB image (PNG/JPG) to classify it as Real or Fake.")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
       
        with st.spinner("Classifying image..."):
            image_array = preprocess_image(image)
            label, probability = classify_image(discriminator, image_array)
        
        st.write(f"**Prediction**: {label}")
        st.write(f"**Probability of being Real**: {probability:.4f}")
    except Exception as e:
        st.error(f"Error processing image: {e}. Ensure the image is a valid 32x32 RGB PNG/JPG.")
