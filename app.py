import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Separace vzorů)")

img_file = st.camera_input("Vyfoťte nahuštěný vzor")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Prahování (izolace inkoustu)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # 2. Odstranění šumu a zvýraznění středů
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Určení oblastí, které jsou URČITĚ pozadí (dilatace)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 4. Určení oblastí, které jsou URČITĚ popředí (vzdálenostní transformace)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 5. Nalezení neznámé oblasti (bg - fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6. Označení markerů (vytvoření "povodí")
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    
    # 7. Aplikace algoritmu Watershed (rozříznutí propojených objektů)
    markers = cv2.watershed(cv2_img, markers)
    
    # Nalezení unikátních markerů (mínus pozadí)
    unique_markers = np.unique(markers)
    count = 0
    debug_img = cv2_img.copy()
    
    for marker in unique_markers:
        if marker <= 1: continue # Přeskočit pozadí
        
        # Vytvoření masky pro daný marker a výpočet plochy
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            # Upravený filtr plochy
            if 50 < area < 10000:
                count += 1
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
            
    st.write(f"### Počet nalezených křížků: {count}")
    # Ukážeme i to, jak Watershed objekty "rozřezal"
    watershed_viz = np.uint8(np.abs(markers)) # Vizualizace markerů
    st.image([cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), watershed_viz], 
             caption=["Nalezené kontury", "Mapa Watershed (rozřezání)"], width=300)
