import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import extra_streamlit_components as stx

with st.expander("Setting", expanded=True):
    size = st.number_input("Size Image", min_value=1, max_value=100, value=50)
    is_grey = st.checkbox("Grey Scale", value=False)
        

typeImage = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="Generate Image",
                       description="Generate Random RGB Image"),
    stx.TabBarItemData(id=2, title="Upload", description="From Image Upload"),
], default=1)

if typeImage == "1":
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    st.image(img, caption='Random Image', use_column_width=True)
    # change to grey scale
    if is_grey:
        grey_kernel = np.array([0.2989, 0.5870, 0.1140])
        for i in range(size):
            for j in range(size):
                img[i, j] = np.dot(img[i, j], grey_kernel)
        st.image(img, caption='Grey Image', use_column_width=True)
        
    with st.expander("Show RGB Value"):
        col1, col2 = st.columns(2)
        col1.header('Coordinates')
        col2.header('RGB Value')
        for i in range(size):
            for j in range(size):
                col1.write(f'({i}, {j})')
                col2.write(f'{img[i, j]}')
    with st.expander("Histogram"):
        fig, ax = plt.subplots()
        ax.hist(img.ravel(), bins=256, range=(0, 255))
        ax.set_title('Histogram')
        st.pyplot(fig)
    # make image from array
    img = Image.fromarray(img)
    # save image
    img.save('random.png')


else:
    uploaded_image = st.file_uploader(
        "Pilih gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Real Image", use_column_width=True)
        img = Image.open(uploaded_image)
        img = img.resize((size, size))
        img = np.array(img)
        st.image(img, caption='Resized Image', use_column_width=True)
        
        

        if uploaded_image.type == "image/png":
            if is_grey:
                grey_kernel = np.array([0.2989, 0.5870, 0.1140, 1])
                for i in range(size):
                    for j in range(size):
                        img[i, j] = np.dot(img[i, j], grey_kernel)
                st.image(img, caption='Grey Image Resized', use_column_width=True)
            with st.expander("Show RGBA Value"):
                col1, col2 = st.columns(2)
                col1.header('Coordinates')
                col2.header('RGBA Value')
                for i in range(size):
                    for j in range(size):
                        col1.write(f'({i}, {j})')
                        col2.write(f'{img[i, j]}')
        else:
            if is_grey:
                grey_kernel = np.array([0.2989, 0.5870, 0.1140])
                for i in range(size):
                    for j in range(size):
                        img[i, j] = np.dot(img[i, j], grey_kernel)
                st.image(img, caption='Grey Image Resized', use_column_width=True)
            with st.expander("Show RGB Value"):
                col1, col2 = st.columns(2)
                col1.header('Coordinates')
                col2.header('RGB Value')
                for i in range(size):
                    for j in range(size):
                        col1.write(f'({i}, {j})')
                        col2.write(f'{img[i, j]}')

        # histogram
        with st.expander("Histogram"):
            fig, ax = plt.subplots()
            ax.hist(img.ravel(), bins=256, range=(0, 255))
            ax.set_title('Histogram')
            st.pyplot(fig)

    