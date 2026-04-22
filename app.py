import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Foťák i Galerie)")

# 1. Výběr zdroje obrázku
zdroj = st.radio("Vyberte způsob zadání:", ("Vyfotit přímo", "Nahrát z galerie"))

img_file = None

if zdroj == "Vyfotit přímo":
    img_file = st.camera_input("Vyfoťte papír")
else:
    img_file = st.file_uploader("Vyberte fotku z galerie", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Převod souboru na obrázek pro OpenCV
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # --- LOGIKA "AUTOMAT", KTERÁ TI FUNGOVALA NEJLÉPE ---
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Odstranění mřížky a šumu
    blurred = cv2.medianBlur(gray, 5)
    
    # Adaptivní prahování odolné vůči stínům (velký blok 81)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 81, 12)
    
    # Vyčištění zbytků
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Separace křížků pomocí vzdálenostní transformace (hledání středů)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Nalezení a spočítání středů
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = cv2_img.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3: # Ignorujeme mikroskopický šum
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Nakreslíme zelenou tečku na střed
                cv2.circle(debug_img, (cX, cY), 12, (0, 255, 0), -1)
                count += 1
                
    st.write(f"### Počet nalezených křížků: {count}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Výsledek detekce")
