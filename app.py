import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Automat)")

img_file = st.camera_input("Vyfoťte papír")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Převod na černobílou
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Rozmazání mřížky (mřížka je tenká, takže zmizí, tlusté křížky zůstanou)
    blurred = cv2.medianBlur(gray, 5)
    
    # 3. Inteligentní boj se stíny
    # Extrémně velký blok (81) zajistí, že stíny od ruky/telefonu budou ignorovány
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 81, 12)
    
    # 4. Vyčištění od zbytků mřížky (drobný prach)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. Nalezení přesných středů (i když se křížky dotýkají)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Najdeme ty nejjasnější body (středy křížků)
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Z těch středů uděláme objekty, abychom je mohli spočítat
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = cv2_img.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 6. Pojistka: střed křížku musí být aspoň malá tečka, ne prach
        if area > 3: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Namalujeme hezkou zelenou tečku přesně doprostřed
                cv2.circle(debug_img, (cX, cY), 12, (0, 255, 0), -1)
                count += 1
                
    st.write(f"### Počet nalezených křížků: {count}")
    
    # Ukážeme výsledek
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Nalezené středy")
