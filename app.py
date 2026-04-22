import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Detekce průsečíků)")

img_file = st.camera_input("Vyfoťte papír")

if img_file is not None:
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Rozostření pro odstranění mřížky
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Detekce rohů (Harris Corner Detection)
    # Křížek má v průsečíku silný "roh"
    gray_float = np.float32(blurred)
    dst = cv2.cornerHarris(gray_float, blockSize=5, ksize=3, k=0.04)
    
    # Roztažení detekovaných bodů, aby byly lépe vidět
    dst = cv2.dilate(dst, None)
    
    # 3. Filtr: Ponecháme jen nejsilnější body (prahování)
    # Tím odfiltrujeme mřížku, protože ta není tak "ostrá" jako průsečík křížku
    threshold_value = 0.05 * dst.max()
    points = np.where(dst > threshold_value)
    
    # 4. Seskupení blízkých bodů (aby jeden křížek nebyl 5 teček)
    points_list = list(zip(points[1], points[0])) # (x, y)
    final_points = []
    
    for pt in points_list:
        # Přidáme bod, pokud je dostatečně daleko od ostatních již nalezených
        if all(np.linalg.norm(np.array(pt) - np.array(fpt)) > 20 for fpt in final_points):
            final_points.append(pt)
    
    # Vykreslení
    debug_img = cv2_img.copy()
    for pt in final_points:
        cv2.circle(debug_img, pt, 10, (0, 255, 0), -1)
            
    st.write(f"### Počet nalezených křížků: {len(final_points)}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Nalezené středy")
