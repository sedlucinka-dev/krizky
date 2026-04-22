import streamlit as st
import cv2
import numpy as np

st.title("Počítač křížků (Hledání šablony)")

st.write("Nahrajte fotku a dolaďte posuvníky tak, aby každý křížek měl právě jednu tečku.")

# Výběr zdroje obrázku
zdroj = st.radio("Zvolte způsob nahrání:", ("Nahrát z galerie", "Vyfotit kamerou"))

if zdroj == "Vyfotit kamerou":
    img_file = st.camera_input("Vyfoťte papír")
else:
    img_file = st.file_uploader("Vyberte fotku", type=["jpg", "jpeg", "png"])

# Posuvníky pro manuální doladění detekce
col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    citlivost = st.slider("Citlivost detekce", min_value=0.40, max_value=0.90, value=0.65, step=0.05, 
                          help="Nižší hodnota najde více křížků (i ty hůře nakreslené), ale může najít i šum.")
with col_slider2:
    min_vzdalenost = st.slider("Minimální vzdálenost", min_value=10, max_value=100, value=35, step=5,
                               help="Pokud to jeden křížek označí více tečkami, zvyšte toto číslo. Odděluje to shluky.")

# Funkce, která virtuálně "nakreslí" šablonu ideálního křížku
def vytvor_sablonu(velikost=30, tloustka=3):
    tpl = np.full((velikost, velikost), 255, dtype=np.uint8)
    cv2.line(tpl, (5, 5), (velikost-5, velikost-5), 0, tloustka)
    cv2.line(tpl, (velikost-5, 5), (5, velikost-5), 0, tloustka)
    # Lehké rozmazání, aby to tolerovalo ruční kresbu
    tpl = cv2.GaussianBlur(tpl, (5, 5), 0)
    return tpl

if img_file is not None:
    # Načtení obrázku
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Změna velikosti fotky (aby to fungovalo stejně rychle na starém i novém mobilu)
    max_šířka = 800
    if cv2_img.shape[1] > max_šířka:
        pomer = max_šířka / cv2_img.shape[1]
        cv2_img = cv2.resize(cv2_img, (max_šířka, int(cv2_img.shape[0] * pomer)))

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Odstranění jemné mřížky z papíru
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Vygenerování šablony a její hledání v obrázku
    sablona = vytvor_sablonu()
    vysledek = cv2.matchTemplate(blurred, sablona, cv2.TM_CCOEFF_NORMED)
    
    # 2. Filtrace podle "Citlivosti" z posuvníku
    lokace = np.where(vysledek >= citlivost)
    body = list(zip(*lokace[::-1])) # Souřadnice (x, y) všech potenciálních shod
    
    # 3. Slučování blízkých bodů (řeší problém se shluky a vícenásobným označením)
    finalni_body = []
    for pt in body:
        ulozit = True
        for fpt in finalni_body:
            # Výpočet vzdálenosti mezi body
            vzdalenost = np.sqrt((pt[0] - fpt[0])**2 + (pt[1] - fpt[1])**2)
            if vzdalenost < min_vzdalenost:
                ulozit = False # Bod je moc blízko jinému, ignorujeme ho
                break
        
        if ulozit:
            # Pokud bod není blízko žádnému jinému, přidáme ho (střed upravíme o půlku šablony)
            stred_x = pt[0] + 15
            stred_y = pt[1] + 15
            finalni_body.append((stred_x, stred_y))
            
    # Vykreslení výsledků
    debug_img = cv2_img.copy()
    for pt in finalni_body:
        # Nakreslení jasně zelené tečky
        cv2.circle(debug_img, pt, 12, (0, 255, 0), -1)
        # Nakreslení tenkého černého okraje, aby to bylo vidět i na světlém pozadí
        cv2.circle(debug_img, pt, 12, (0, 0, 0), 2)
            
    st.write(f"### Počet nalezených křížků: {len(finalni_body)}")
    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Výsledek")
