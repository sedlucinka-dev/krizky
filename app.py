import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Vylepšený)")

img_file = st.camera_input("Vyfoťte papír s křížky")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptivní prahování (lépe zvládá stíny)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 10)
    
    # 2. Vyčištění šumu (spojí části křížku k sobě)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    # 3. Filtrace: hledáme objekty, které nejsou příliš malé ani příliš velké
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Upravte tyto limity podle toho, jak velké křížky fotíte
        if 200 < area < 5000: 
            # Aproximace tvaru - pomůže eliminovat drobné tečky
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Křížek má obvykle více než 4 rohy (často 8-12 podle tloušťky čar)
            if len(approx) > 4:
                count += 1
                cv2.drawContours(cv2_img, [cnt], -1, (0, 255, 0), 3)
            
    st.write(f"### Počet nalezených křížků: {count}")
    st.image(cv2_img, channels="BGR")
