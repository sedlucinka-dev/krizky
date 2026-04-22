import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Finalní Logic)")

img_file = st.camera_input("Vyfoťte papír s křížky")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Adaptivní prahování - citlivé na tenké čáry
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # 2. Skeletizace / Hledání středů (Topologie)
    # Zjistíme "vzdálenost k okraji". Středy čar křížku budou mít nejvyšší hodnotu.
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    
    # Najdeme "lokální maxima" - to jsou středy křížků.
    # Práh nastavíme na 50 % maximální vzdálenosti, abychom ignorovali šum.
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Označení propojených komponent
    # Každý odsouhlasený "střed" získá unikátní ID.
    num_labels, labels = cv2.connectedComponents(sure_fg)
    
    # num_labels obsahuje počet unikátních ID, včetně pozadí (ID 0).
    count = num_labels - 1 
    
    # Vykreslení pro kontrolu
    debug_img = cv2_img.copy()
    
    # Najdeme středy detekovaných komponent a zakreslíme k nim tečky
    for i in range(1, num_labels):
        # Maska pro aktuální komponentu
        component_mask = np.zeros_like(thresh)
        component_mask[labels == i] = 255
        
        # Najdeme těžiště komponenty
        M = cv2.moments(component_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(debug_img, (cX, cY), 7, (0, 255, 0), -1)
            
    st.write(f"### Počet nalezených křížků: {count}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Nalezené středy křížků", width=400)
