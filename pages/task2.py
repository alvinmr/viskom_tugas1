# dilation, erosion, opening, closing, hit or miss, boundary extraction, thinning, skeletonization, hough transform line, hough transform circle

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import extra_streamlit_components as stx

with st.expander("Setting", expanded=True):
    size = st.number_input("Size Image", min_value=1, max_value=100, value=50)
    is_grey = st.checkbox("Grey Scale", value=True)
        

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
        
        img_grey = np.dot(img, grey_kernel)
        img_grey = img_grey.astype(np.uint8)  # Ensure the data type is uint8
        st.image(img_grey, caption='Grey Image', use_column_width=True)
        
    with st.expander("Dilation"):
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(img_grey,kernel,iterations = 1)
        st.image(dilation, caption='Dilation', use_column_width=True)
        
    with st.expander("Erosion"):
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img_grey,kernel,iterations = 1)
        st.image(erosion, caption='Erosion', use_column_width=True)

    with st.expander("Opening"):
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(img_grey, cv2.MORPH_OPEN, kernel)
        st.image(opening, caption='Opening', use_column_width=True)

    with st.expander("Closing"):
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel)
        st.image(closing, caption='Closing', use_column_width=True)

    with st.expander("Hit or Miss"):
        # kernel = np.array([[0, 1, 0],
        #               [1, -1, 1],
        #               [0, 1, 0]], dtype=np.uint8)  # Example kernel for Hit or Miss
        kernel = np.ones((5,5),np.uint8)
        hitormiss = cv2.morphologyEx(img_grey, cv2.MORPH_HITMISS, kernel)
        st.image(hitormiss, caption='Hit or Miss', use_column_width=True)

    with st.expander("Boundary Extraction"):
        kernel = np.ones((5,5),np.uint8)
        boundary = cv2.morphologyEx(img_grey, cv2.MORPH_GRADIENT, kernel)
        st.image(boundary, caption='Boundary Extraction', use_column_width=True)

    with st.expander("Thinning"):
        kernel = np.ones((5,5),np.uint8)
        thinning = cv2.ximgproc.thinning(img_grey, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        st.image(thinning, caption='Thinning', use_column_width=True)

    with st.expander("Skeletonization"):
        kernel = np.ones((5,5),np.uint8)
        skeletonization = cv2.ximgproc.thinning(img_grey, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        st.image(skeletonization, caption='Skeletonization', use_column_width=True)

    with st.expander("Hough Transform Line"):
        kernel = np.ones((5,5),np.uint8)
        houghline = cv2.HoughLines(img_grey,1,np.pi/180,200)
        if houghline is not None: 
            for i in range(len(houghline)):
                for rho, theta in houghline[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(img_grey, (x1, y1), (x2, y2), (0, 0, 255), 2)
            st.image(img_grey, caption='Hough Transform Line', use_column_width=True)
        else:
            st.write("No lines found in the image.")

    with st.expander("Hough Transform Circle"):
        kernel = np.ones((5,5),np.uint8)
        houghcircle = cv2.HoughCircles(img_grey,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=30,minRadius=0,maxRadius=0)
        if houghline is not None:
            houghcircle = np.uint16(np.around(houghcircle))
            for i in houghcircle[0,:]:
                # draw the outer circle
                cv2.circle(img_grey,(i[0],i[1]),i[2],(255,255,255),2)
                # draw the center of the circle
                cv2.circle(img_grey,(i[0],i[1]),2,(255,255,255),3)
            st.image(img_grey, caption='Hough Transform Circle', use_column_width=True)
        else:
            st.write("No circle found in the image.")
        
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
            # img = Image.open(uploaded_image)
            img = cv2.imread(uploaded_image, cv2.IMREAD_GRAYSCALE)
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
            with st.expander("Dilation"):
                kernel = np.ones((5,5),np.uint8)
                dilation = cv2.dilate(img,kernel,iterations = 1)
                st.image(dilation, caption='Dilation', use_column_width=True)
                
            with st.expander("Erosion"):
                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(img,kernel,iterations = 1)
                st.image(erosion, caption='Erosion', use_column_width=True)

            with st.expander("Opening"):
                kernel = np.ones((5,5),np.uint8)
                opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                st.image(opening, caption='Opening', use_column_width=True)

            with st.expander("Closing"):
                kernel = np.ones((5,5),np.uint8)
                closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                st.image(closing, caption='Closing', use_column_width=True)

            with st.expander("Hit or Miss"):
                img = img.astype(np.uint8)  # Konversi tipe data citra ke np.uint8 jika tipe data yang tidak sesuai
                kernel = np.ones((5,5),np.uint8)
                hitormiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
                st.image(hitormiss, caption='Hit or Miss', use_column_width=True)

            with st.expander("Boundary Extraction"):
                kernel = np.ones((5,5),np.uint8)
                boundary = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
                st.image(boundary, caption='Boundary Extraction', use_column_width=True)

            with st.expander("Thinning"):
                kernel = np.ones((5,5),np.uint8)
                thinning = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                st.image(thinning, caption='Thinning', use_column_width=True)

            with st.expander("Skeletonization"):
                kernel = np.ones((5,5),np.uint8)
                skeletonization = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
                st.image(skeletonization, caption='Skeletonization', use_column_width=True)

            with st.expander("Hough Transform Line"):
                kernel = np.ones((5,5),np.uint8)
                houghline = cv2.HoughLines(img,1,np.pi/180,200)
                for i in range(len(houghline)):
                    for rho,theta in houghline[i]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                st.image(img, caption='Hough Transform Line', use_column_width=True)

            with st.expander("Hough Transform Circle"):
                kernel = np.ones((5,5),np.uint8)
                houghcircle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                            param1=50,param2=30,minRadius=0,maxRadius=0)
                houghcircle = np.uint16(np.around(houghcircle))
                for i in houghcircle[0,:]:
                    # draw the outer circle
                    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),2)
                    # draw the center of the circle
                    cv2.circle(img,(i[0],i[1]),2,(255,255,255),3)
                st.image(img, caption='Hough Transform Circle', use_column_width=True)
        else:
            if is_grey:
                grey_kernel = np.array([0.2989, 0.5870, 0.1140])
                for i in range(size):
                    for j in range(size):
                        img[i, j] = np.dot(img[i, j], grey_kernel)
                st.image(img, caption='Grey Image Resized', use_column_width=True)
            with st.expander("Dilation"):
                kernel = np.ones((5,5),np.uint8)
                dilation = cv2.dilate(img,kernel,iterations = 1)
                st.image(dilation, caption='Dilation', use_column_width=True)
                
            with st.expander("Erosion"):
                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(img,kernel,iterations = 1)
                st.image(erosion, caption='Erosion', use_column_width=True)

            with st.expander("Opening"):
                kernel = np.ones((5,5),np.uint8)
                opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                st.image(opening, caption='Opening', use_column_width=True)

            with st.expander("Closing"):
                kernel = np.ones((5,5),np.uint8)
                closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                st.image(closing, caption='Closing', use_column_width=True)

            with st.expander("Hit or Miss"):
                kernel = np.ones((5,5),np.uint8)
                hitormiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
                st.image(hitormiss, caption='Hit or Miss', use_column_width=True)

            with st.expander("Boundary Extraction"):
                kernel = np.ones((5,5),np.uint8)
                boundary = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
                st.image(boundary, caption='Boundary Extraction', use_column_width=True)

            with st.expander("Thinning"):
                kernel = np.ones((5,5),np.uint8)
                thinning = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                st.image(thinning, caption='Thinning', use_column_width=True)

            with st.expander("Skeletonization"):
                kernel = np.ones((5,5),np.uint8)
                skeletonization = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
                st.image(skeletonization, caption='Skeletonization', use_column_width=True)

            with st.expander("Hough Transform Line"):
                kernel = np.ones((5,5),np.uint8)
                houghline = cv2.HoughLines(img,1,np.pi/180,200)
                for i in range(len(houghline)):
                    for rho,theta in houghline[i]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                st.image(img, caption='Hough Transform Line', use_column_width=True)

            with st.expander("Hough Transform Circle"):
                kernel = np.ones((5,5),np.uint8)
                houghcircle = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                            param1=50,param2=30,minRadius=0,maxRadius=0)
                houghcircle = np.uint16(np.around(houghcircle))
                for i in houghcircle[0,:]:
                    # draw the outer circle
                    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),2)
                    # draw the center of the circle
                    cv2.circle(img,(i[0],i[1]),2,(255,255,255),3)
                st.image(img, caption='Hough Transform Circle', use_column_width=True)



