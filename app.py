import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Final-v2)")

img_file = st.camera_input("Vyfoťte papír")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Rozostření - klíčový krok! Odstraní mřížku na papíře
    blurred = cv2.medianBlur(gray, 5)
    
    # 2. Inteligentní prahování (Otsu) - lépe izoluje inkoust od papíru
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morfologie: Spojení kousků k sobě a odříznutí sousedů
    # Použijeme větší kernel, aby se kousky křížku "slepily"
    kernel_connect = np.ones((5,5), np.uint8)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
    
    # Teď agresivní eroze, aby se oddělily křížky blízko u sebe
    kernel_separate = np.ones((3,3), np.uint8)
    processed = cv2.erode(connected, kernel_separate, iterations=3)
    
    # Najdeme kontury na vyčištěném obrázku
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    debug_img = cv2_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 4. Filtr plochy - teď můžeme být přísnější, protože kousky jsou spojené
        if 80 < area < 10000:
            count += 1
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)
            
    st.write(f"### Počet nalezených křížků: {count}")
    
    # Vizualizace pro kontrolu
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Detekce")
    with col2:
        st.image(processed, caption="Jak to vidí počítač")
