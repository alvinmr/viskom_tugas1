"""
# My first app
Here's our first attempt at using data to create a table:
"""
# ekstrak nilai dari gambar rgb_pict.jpg
import streamlit as st
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def brightness_contrass(data_ekstrak_to_image, kecerahan, kontras):
    # kecerahan
    for i in range(len(data_ekstrak_to_image)):
        # st.write(i)
        for j in range(len(data_ekstrak_to_image[i])):
            # st.write(data_ekstrak_to_image[i][j])
            data_ekstrak_to_image[i][j] += kecerahan

    # kontras
    for i in range(len(data_ekstrak_to_image)):
        for j in range(len(data_ekstrak_to_image[i])):
            data_ekstrak_to_image[i][j] *= kontras

    # tampilkan data
    # st.dataframe(pd.DataFrame(data_ekstrak_to_image, columns=['r', 'g', 'b']))
    
    # Clip values to the 0-255 range
    data_ekstrak_to_image = np.clip(data_ekstrak_to_image, 0, 255).astype(np.uint8)

    # Convert data_ekstrak_to_image to a NumPy array
    data_ekstraksi_array = np.array(data_ekstrak_to_image, dtype=np.uint8)

    # Reshape the array to match the image dimensions
    image_shape = (resize_img.shape[0], resize_img.shape[1], 3) # shape[0] for height, shape[1] for width, 3 for RGB
    data_ekstraksi_array = data_ekstraksi_array.reshape(image_shape) # Reshape the array to match the image dimensions

    return data_ekstraksi_array


y = 0
x = 0

# tampilkan data
st.markdown("# Ekstraksi Nilai RGB")
st.sidebar.markdown("# Visi Komputer by Ardhiya :bulb:")

st.text("Engine v1.0 - Ekstraksi Nilai RGB")

st.write("Silahkan upload gambar yang ingin diekstrak nilainya:")

# Inisialisasi list untuk menyimpan data
data_ekstraksi = []
data_ekstrak_to_image = []
data_intensitas = []
hasil_kecerahan = []

# Input gambar
size = st.slider('Size', min_value=20, max_value=500, value=100, step=1, key='size')

img = st.file_uploader(label='Unggah Gambar', type=['png', 'jpg'], key='gambar') 

if img is not None:
    st.write('Data Gambar berhasil di upload')
    img_pil = Image.open(img)
    img_array = np.array(img_pil, dtype=np.uint8)
    
    # resize image with cv2
    resize_img = cv2.resize(img_array, (size, size))
    # st.image(resize_img, caption='Hasil Resize.', use_column_width=False)

    # tampilkan seluruh pixel_value dengan for loop
    for y in range(resize_img.shape[0]):
        for x in range(resize_img.shape[1]):
            pixel_value = resize_img[y, x]
            data_ekstraksi.append([y, x, pixel_value[0], pixel_value[1], pixel_value[2]])
            data_ekstrak_to_image.append([pixel_value[0], pixel_value[1], pixel_value[2]])

    kecerahan = st.slider('Kecerahan', min_value=-255, max_value=255, value=0, step=1, key='kecerahan')
    kontras = st.slider('Kontras', min_value=-3.0, max_value=3.0, value=1.0, step=0.1, key='kontras')

    for loop in range(2):
        if loop == 1:       
            # grey scale
            for i in data_ekstrak_to_image:
                i[0] = i[0] * 0.299
                i[1] = i[1] * 0.587
                i[2] = i[2] * 0.114

                i[0] = i[0] + i[1] + i[2]
                i[1] = i[0]
                i[2] = i[0]

                if i[0] > 255:
                    i[0] = 255
                if i[1] > 255:
                    i[1] = 255
                if i[2] > 255:
                    i[2] = 255

                if i[0] < 0:
                    i[0] = 0
                if i[1] < 0:
                    i[1] = 0
                if i[2] < 0:
                    i[2] = 0

                i[0] = int(i[0])
                i[1] = int(i[1])
                i[2] = int(i[2])

            hasil_kecerahan.append(brightness_contrass(data_ekstrak_to_image, kecerahan, kontras))
        
        else:
            hasil_kecerahan.append(brightness_contrass(data_ekstrak_to_image, kecerahan, kontras))


    col1, col2 = st.columns(2)
    img_result1 = Image.fromarray(hasil_kecerahan[0])
    img_result2 = Image.fromarray(hasil_kecerahan[1])
    col1.image(img_result1, caption='Hasil Kecerahan dan Kontras Gambar Asli.', use_column_width=False)
    col2.image(img_result2, caption='Hasil Kecerahan dan Kontras Gambar Greyscale.', use_column_width=False)

    # kontras
    # for i in range(len(data_ekstrak_to_image)):
    #     for j in range(len(data_ekstrak_to_image[i])):
    #         data_ekstrak_to_image[i][j] *= kontras

    # for i in range(data_ekstrak_to_image.shape[0]):
    #     for j in range(data_ekstrak_to_image.shape[1]):
    #         for k in range(data_ekstrak_to_image.shape[2]):
    #             data_ekstrak_to_image[i, j, k] += kecerahan

# batas
#     
    # Input koordinat
    row = st.columns([1.25, 1, 1, 1])

    ekstraksi_nilai = row[0].button('Ekstraksi Nilai', key='ekstraksi_nilai')
    intensitas = row[1].button('Intensitas', key='intensitas')
    line_chart = row[2].button('Line Chart', key='line_chart')
    histogram = row[3].button('Histogram', key='histogram')

    if ekstraksi_nilai: 
        st.dataframe(pd.DataFrame(data_ekstraksi, columns=['y', 'x', 'r', 'g', 'b']))

    if intensitas:
        b, g, r = cv2.split(resize_img)

        intensitas_r = r.sum() / (r.shape[0] * r.shape[1])
        intensitas_g = g.sum() / (g.shape[0] * g.shape[1])
        intensitas_b = b.sum() / (b.shape[0] * b.shape[1])
        intensitas_pixel = resize_img.sum() / (resize_img.shape[0] * resize_img.shape[1] * resize_img.shape[2])
        data_intensitas.append([intensitas_r, intensitas_g, intensitas_b, intensitas_pixel])
        st.dataframe(pd.DataFrame(data_intensitas, columns=['r', 'g', 'b', 'pixel']))
            
    if line_chart:
        st.line_chart(pd.DataFrame(data_ekstraksi, columns=['y', 'x', 'r', 'g', 'b']))

    if histogram:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        b, g, r = cv2.split(resize_img)

        plt.hist(r.ravel(), 256, [0, 256])
        plt.title('Histogram Merah')
        plt.xlim([0, 256])
        st.pyplot()

        plt.hist(g.ravel(), 256, [0, 256])
        plt.title('Histogram Hijau')
        plt.xlim([0, 256])
        st.pyplot()

        plt.hist(b.ravel(), 256, [0, 256])
        plt.title('Histogram Biru')
        plt.xlim([0, 256])
        st.pyplot()

        plt.hist(resize_img.ravel(), 256, [0, 256])
        plt.title('Histogram RGB')
        plt.xlim([0, 256])
        st.pyplot()


# if submit_button:
#     st.write('Data Gambar berhasil di upload')
#     gambar = st.file_uploader(label='Upload your image', type=['png', 'jpg'])
#     img = cv2.imread(gambar)
    
#     pixel_value = img[y, x]

#     
        

# simpan data ke file csv
# berikan nama kolom saat menyimpan data
# np.savetxt("pixel_value.csv", data, delimiter=",", fmt='%d', header="y,x,r,g,b")


