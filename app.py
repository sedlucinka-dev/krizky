import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Chytré ladění)")

img_file = st.camera_input("Vyfoťte papír")

# TOTO JE NOVINKA: Posuvník přímo v aplikaci!
citlivost = st.slider("Citlivost filtru (laďte, dokud nezmizí mřížka)", min_value=5, max_value=50, value=25)

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptivní prahování (odolné vůči stínům)
    # Používá hodnotu z posuvníku k odlišení tužky od mřížky
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 41, citlivost)
    
    # 2. Vyčištění od prachu
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Hledání středů (Topologie) - oddělí křížky u sebe
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = cv2_img.copy()
    count = 0
    
    for cnt in contours:
        # Vypočítáme těžiště pro umístění tečky přesně doprostřed
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(debug_img, (cX, cY), 10, (0, 255, 0), -1)
            count += 1
            
    st.write(f"### Počet nalezených křížků: {count}")
    
    # Vykreslíme dva obrázky vedle sebe pro snadné ladění
    st.write("---")
    st.write("### Nápověda pro ladění:")
    st.write("Dívejte se na pravý černobílý obrázek. Posouvejte posuvníkem nahoře tak, aby mřížka papíru byla úplně černá a křížky svítily čistě bíle.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Výsledek")
    with col2:
        st.image(thresh, caption="Pohled počítače (Zde laďte)")
