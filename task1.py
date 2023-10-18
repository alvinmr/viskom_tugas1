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
        col1, col2 = st.columns(2)
        with col1:
            real_img = Image.open(uploaded_image)
            real_img = np.array(real_img)
            st.image(real_img, caption="Real Image", use_column_width=True)
        with col2:
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
            col1, col2 = st.columns(2)
            col1.header('Real Image')
            col2.header('Resized Image')
            with col1:
                fig, ax = plt.subplots()
                ax.hist(real_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Real Image')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                ax.hist(img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Resized Image')
                st.pyplot(fig)
            
        with st.expander("Brightness Modif"):
            brightness = st.slider("Brightness", min_value=0, max_value=255, value=0)
            width, height, channel = real_img.shape
            if channel == 4:
                grey_kernel = np.array([0.2989, 0.5870, 0.1140, 1])
            else:
                grey_kernel = np.array([0.2989, 0.5870, 0.1140])
            brightness_img = np.zeros((width, height, 3), dtype=np.uint8)
            gray_img = np.zeros((width, height, 3), dtype=np.uint8)
            for i in range(width):
                for j in range(height):
                    gray_img[i, j] = np.dot(real_img[i, j], grey_kernel)
                    brightness_img[i, j] = np.dot(real_img[i, j], grey_kernel) + brightness
            st.image(brightness_img, caption='Brightness Modif', use_column_width=True)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.hist(gray_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Real Image (Grey)')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                ax.hist(brightness_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Brightness Modif Image')
                st.pyplot(fig)
                
        
        with st.expander("Contrast Modif"):
            contrast = st.slider("Contrast level", 0.5, 2.0, 1.0, 0.1)
            width, height, channel = real_img.shape
            contrast_img = np.zeros((width, height, 3), dtype=np.uint8)
            gray_img = np.zeros((width, height, 3), dtype=np.uint8)
            for i in range(width):
                for j in range(height):
                    gray_img[i, j] = np.dot(real_img[i, j], grey_kernel)
                    contrast_img[i, j] = np.clip((gray_img[i, j] - 128) * contrast + 128, 0, 255)
            st.image(contrast_img, caption='Contrast Modif', use_column_width=True)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.hist(gray_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Real Image (Grey)')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                ax.hist(contrast_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Contrast Modif Image')
                st.pyplot(fig)
        
        with st.expander("Negasi Modif"):
            width, height, channel = real_img.shape
            negasi_img = np.zeros((width, height, 3), dtype=np.uint8)
            gray_img = np.zeros((width, height, 3), dtype=np.uint8)
            for i in range(width):
                for j in range(height):
                    gray_img[i, j] = np.dot(real_img[i, j], grey_kernel)
                    negasi_img[i, j] = 255 - gray_img[i, j]
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray_img, caption='Real Image (Grey)', use_column_width=True)
                fig, ax = plt.subplots()
                ax.hist(gray_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Real Image (Grey)')
                st.pyplot(fig)
            with col2:
                st.image(negasi_img, caption='Negasi Modif', use_column_width=True)
                fig, ax = plt.subplots()
                ax.hist(negasi_img.ravel(), bins=256, range=(0, 255))
                ax.set_title('Histogram Negasi Modif Image')
                st.pyplot(fig)
        
        with st.expander("Flipping Modif"):
            width, height, channel = real_img.shape
            if channel == 4:
                flipping_img = np.zeros((width, height, 4), dtype=np.uint8)
            else:
                flipping_img = np.zeros((width, height, 3), dtype=np.uint8)
            st.image(real_img, caption='Real Image', use_column_width=True)
            # button flipping right 90 degree
            if st.button("Flipping Right 90"):
                if channel == 4:
                    flipping_img = np.zeros((height, width, 4), dtype=np.uint8)
                else:
                    flipping_img = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        flipping_img[i, j] = real_img[width-j-1, i]
            # button flipping left 90 degree
            if st.button("Flipping Left 90"):
                if channel == 4:
                    flipping_img = np.zeros((height, width, 4), dtype=np.uint8)
                else:
                    flipping_img = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        flipping_img[i, j] = real_img[j, height-i-1]
            # button flipping 180 degree
            if st.button("Flipping Vertical Mirror"):
                for i in range(width):
                    for j in range(height):
                        flipping_img[i, j] = real_img[i, height-j-1]
            # button flipping horizontal mirror
            if st.button("Flipping Horizontal Mirror"):
                for i in range(width):
                    for j in range(height):
                        flipping_img[i, j] = real_img[width-i-1, j]
            # button flipping horizontal and vertical mirror
            if st.button("Flipping Combination Mirror"):
                for i in range(width):
                    for j in range(height):
                        flipping_img[i, j] = real_img[width-i-1, height-j-1]
            st.image(flipping_img, caption='Flipping Modif', use_column_width=True)
            
            