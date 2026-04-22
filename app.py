import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Final)")

# Komponenta pro kameru
img_file = st.camera_input("Vyfoťte papír s křížky")

if img_file is not None:
    # Převod na OpenCV formát
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Převod na šedou
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptivní prahování (citlivost 15, 7 pro detailnější detekci)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 7)
    
    # 3. Odstranění šumu
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Watershed - pro rozdělení nahuštěných křížků
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(cv2_img, markers)
    
    # 5. Počítání a vykreslení
    unique_markers = np.unique(markers)
    count = 0
    debug_img = cv2_img.copy()
    
    for marker in unique_markers:
        if marker <= 1: continue
        
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            # Snížená hranice plochy na 20 pro detekci i menších křížků
            if 20 < area < 10000:
                count += 1
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            
    st.write(f"### Počet nalezených křížků: {count}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Nalezené křížky", width=400)
