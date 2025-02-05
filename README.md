# 🛰️ Detekcija osoba s visine pomoću YOLOv8n

## 🔧 Instalacija

Prvo instalirajte sve potrebne zavisnosti pokretanjem naredbe u terminalu:

```bash
pip install -r requirements.txt
```

## 🚀 Korištenje

### 1️⃣ Postavljanje videa
Postavite putanju u `extract_frames.py` (linija 4).

### 2️⃣ Ekstrakcija frejmova
Pokrenite sljedeću naredbu:

```bash
python extract_frames.py
```

Frejmovi će biti spremljeni u `/frames` direktorij.

> Svaki 5. frejm se sprema (možete promijeniti u kodu `frame_count % 5`).

### 3️⃣ Generiranje labela i priprema dataseta
Pokrenite sljedeću naredbu:

```bash
python train_yolo.py
```

Ova skripta radi sljedeće:
- 📥 Automatski preuzima YOLOv8n model (prvi put)
- 🏷️ Generira labele za sve frejmove
- 📂 Dijeli podatke u `train/test/val` skupove
- ⚙️ Kreira `dataset.yaml` konfiguraciju
- 🎨 Generira vizualizaciju detekcija u `/annotated_images`

---

## 📁 Struktura projekta

```
📂 project/
├── 📄 extract_frames.py
├── 📄 train_yolo.py
├── 🎥 vid1.mp4 # Originalni video
├── 📂 frames/ # Ekstrahirani frejmovi
├── 📂 labels/ # Generirani labele
├── 📂 dataset/ # Finalni dataset
│   ├── 📂 train/
│   │   ├── 🖼️ images/
│   │   └── 🏷️ labels/
│   ├── 📂 val/
        ├── 🖼️ images/
│       └── 🏷️ labels/
│   └── 📂 test/
        ├── 🖼️ images/
│       └── 🏷️ labels/
├── 📂 annotated_images/ # Vizualizirane detekcije
└── 📄 requirements.txt
```

## ℹ️ Napomene
- Skripte su napisane u Pythonu i zahtijevaju odgovarajuće biblioteke.
- Možete prilagoditi kod prema vlastitim potrebama.

📌 Za dodatna podešavanja i optimizacije pogledajte kod ili prilagodite parametre prema vašim potrebama.

