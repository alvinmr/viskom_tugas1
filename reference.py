import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Fungsi untuk mengubah ukuran gambar
def resize_image(image, width, height):
    return image.resize((width, height))


# Fungsi untuk mengubah warna gambar
def change_image_color(image, red_scale, green_scale, blue_scale):
    image_array = np.array(image)
    image_array[:, :, 0] = (image_array[:, :, 0] * red_scale).astype(np.uint8)
    image_array[:, :, 1] = (image_array[:, :, 1] * green_scale).astype(np.uint8)
    image_array[:, :, 2] = (image_array[:, :, 2] * blue_scale).astype(np.uint8)
    return Image.fromarray(image_array)


# Fungsi untuk menghitung intensitas warna di tiap pixel
def calculate_pixel_intensity(image):
    image_array = np.array(image)
    intensity = np.mean(image_array, axis=2)
    return intensity


# Tampilan aplikasi Streamlit
st.title("Aplikasi Resize dan Ubah Warna Gambar")

# Upload gambar
uploaded_image = st.file_uploader("Pilih gambar:", type=["jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Gambar Asli", use_column_width=True)

    # Input ukuran baru
    st.subheader("Resize Gambar")
    new_width = st.number_input("Masukkan lebar baru:", min_value=1)
    new_height = st.number_input("Masukkan tinggi baru:", min_value=1)

    # Input perubahan warna
    st.subheader("Ubah Warna Gambar")
    red_scale = st.slider("Skala Merah", 0.0, 2.0, 1.0)
    green_scale = st.slider("Skala Hijau", 0.0, 2.0, 1.0)
    blue_scale = st.slider("Skala Biru", 0.0, 2.0, 1.0)

    if st.button("Proses"):
        # Baca gambar yang diunggah
        image = Image.open(uploaded_image)

        # Resize gambar
        resized_image = resize_image(image, new_width, new_height)
        st.image(
            resized_image, caption="Gambar Setelah Diresize", use_column_width=True
        )

        # Ubah warna gambar
        colored_image = change_image_color(
            resized_image, red_scale, green_scale, blue_scale
        )
        st.image(
            colored_image, caption="Gambar Setelah Diubah Warna", use_column_width=True
        )

        # Hitung intensitas warna di tiap pixel
        pixel_intensity = calculate_pixel_intensity(colored_image)
        st.subheader("Intensitas Warna di Tiap Pixel")

        # Tampilkan gambar intensitas warna dengan colormap 'gray'
        plt.subplot(1, 2, 1)
        plt.imshow(pixel_intensity, cmap="gray")
        plt.title("Grayscale")

        # Tampilkan gambar intensitas warna dengan colormap berwarna
        plt.subplot(1, 2, 2)
        plt.imshow(
            pixel_intensity, cmap="viridis"
        )  # Ganti 'viridis' dengan colormap berwarna lainnya
        plt.title("Colormap Berwarna")

        plt.tight_layout()
        st.pyplot(plt)
