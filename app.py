import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Detekce 3.0)")

img_file = st.camera_input("Vyfoťte papír s křížky")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Převedeme na šedou
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Inteligentní prahování (Otsu) - lépe najde černé objekty na bílém
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Morfologické uzavření (spojí čáry křížku do jednoho objektu)
    kernel = np.ones((5,5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Najdeme kontury
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    # Vykreslení pro kontrolu
    debug_img = cv2_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 3. Upravený filtr - teď budeme počítat vše, co vypadá jako objekt
        if 100 < area < 10000:
            count += 1
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            
    st.write(f"### Počet nalezených objektů: {count}")
    # Ukážeme i to, co aplikace vidí po zpracování (binární obrázek)
    st.image([cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), processed], 
             caption=["Nalezené kontury", "Jak to vidí počítač"], width=300)
