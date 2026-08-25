import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from model import predict


st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit.")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

# canvas = st_canvas(
#     fill_color="black",
#     stroke_width=15,
#     stroke_color="white",
#     background_color="black",
#     height=280,
#     width=280,
#     drawing_mode="freedraw",
#     key="canvas",
# )

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to NumPy
    X = np.array(image)

    # Normalize
    X = X / 255.0

    # Reshape: (28,28) → (784,1)
    X = X.reshape(784, 1)

    # Predict
    prediction, probabilities = predict(X)

    digit = prediction[0]
    confidence = probabilities[:, 0][digit]

    # Display
    st.image(image, caption="Input image", width=200)

    st.success(f"Prediction: {digit}")

    st.write(f"Confidence: {confidence * 100:.2f}%")






# if st.button("Predict"):
#     if canvas.image_data is not None:
#         image = canvas.image_data

#         gray = image[:, :, 0]
#         img = Image.fromarray(gray.astype(np.uint8))

#         img = img.resize((28, 28))
#         x = np.array(img).astype(np.float32) / 255.0
#         x = x.reshape(784, 1)

#         prediction = predict(x)
#         digit = prediction[0][0]
#         st.success(f"{digit}")