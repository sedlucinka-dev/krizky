import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Blob Detection)")

img_file = st.camera_input("Vyfoťte papír")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Invertujeme barvy (křížky budou bílé na černém pozadí)
    inverted = cv2.bitwise_not(gray)
    
    # 2. Nastavení Blob detektoru
    params = cv2.SimpleBlobDetector_Params()
    
    # Filtrujeme podle plochy (nastav podle velikosti křížků)
    params.filterByArea = True
    params.minArea = 100 
    params.maxArea = 2000
    
    # Filtrujeme podle kruhovitosti (křížek je celkem "kompaktní" objekt)
    params.filterByCircularity = False
    
    # Filtrujeme podle konvexnosti (aby to nebralo mřížku)
    params.filterByConvexity = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    
    # 3. Vykreslení
    debug_img = cv2_img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(debug_img, (x, y), 10, (0, 255, 0), -1)
            
    st.write(f"### Počet nalezených křížků: {len(keypoints)}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Nalezené středy")
