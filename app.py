import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků")

# Komponenta pro kameru
img_file = st.camera_input("Vyfoťte papír s křížky")

if img_file is not None:
    # Převod na OpenCV formát
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Zpracování obrazu
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Prahování pro zvýraznění křížků
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Počítání (filtrace podle velikosti)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000:
            count += 1
            
    st.write(f"### Počet nalezených křížků: {count}")
    st.image(cv2_img, channels="BGR", caption="Analyzovaný snímek")
