import cv2
import os
import shutil
import random
from ultralytics import YOLO

# 1. Automatsko označavanje frejmova
def generate_labels(model_path, frames_dir, labels_dir):
    # Kreiraj labels direktorij
    os.makedirs(labels_dir, exist_ok=True)
    
    # Učitaj YOLO model
    model = YOLO(model_path)
    
    # Procesuiraj sve frejmove
    for frame_file in os.listdir(frames_dir):
        if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Napravi predikciju
            results = model.predict(frame_path, conf=0.5, verbose=False)
            
            # Pripremi label file
            label_file = os.path.splitext(frame_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # Sačuvaj labele u YOLO formatu
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    if box.cls == 0:  # Samo osobe (class 0 u COCO datasetu)
                        x_center = (box.xyxyn[0][0] + box.xyxyn[0][2]) / 2
                        y_center = (box.xyxyn[0][1] + box.xyxyn[0][3]) / 2
                        width = box.xyxyn[0][2] - box.xyxyn[0][0]
                        height = box.xyxyn[0][3] - box.xyxyn[0][1]
                        
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 2. Podjela podataka u train/test/val
def split_dataset(frames_dir, labels_dir, dataset_dir, ratios=(0.7, 0.2, 0.1)):
    # Pripremi putanje
    dirs = {
        'train': os.path.join(dataset_dir, 'train'),
        'test': os.path.join(dataset_dir, 'test'),
        'val': os.path.join(dataset_dir, 'val')
    }
    
    # Kreiraj direktorije
    for split in dirs.values():
        os.makedirs(os.path.join(split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split, 'labels'), exist_ok=True)
    
    # Lista svih slika
    all_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)
    
    # Podijeli dataset
    n = len(all_files)
    train_end = int(n * ratios[0])
    test_end = train_end + int(n * ratios[1])
    
    splits = {
        'train': all_files[:train_end],
        'test': all_files[train_end:test_end],
        'val': all_files[test_end:]
    }
    
    # Kopiraj fajlove
    for split_name, files in splits.items():
        for file in files:
            # Kopiraj sliku
            src_img = os.path.join(frames_dir, file)
            dst_img = os.path.join(dirs[split_name], 'images', file)
            shutil.copy(src_img, dst_img)
            
            # Kopiraj labelu
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(dirs[split_name], 'labels', label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

# 3. Vizualizacija labela
def visualize_labels(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Učitaj sliku
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # Učitaj labele
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, xc, yc, bw, bh = map(float, line.strip().split())
                        
                        # Konvertuj YOLO format u pixel koordinate
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        
                        # Nacrtaj bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Sačuvaj sliku s oznakama
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, img)

# Konfiguracija
model_path = 'yolov8n.pt'
frames_dir = 'frames/'
labels_dir = 'labels/'
dataset_dir = 'dataset/'

# Glavni tok izvršavanja
if __name__ == "__main__":
    # 1. Generiraj labele
    generate_labels(model_path, frames_dir, labels_dir)
    
    # 2. Podijeli dataset
    split_dataset(frames_dir, labels_dir, dataset_dir)
    
    # 3. Kreiraj dataset.yaml
    yaml_content = f"""path: {dataset_dir}
    train: train/images
    val: val/images
    test: test/images

    names:
      0: person
    """
    with open(os.path.join(dataset_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # 4. Vizualiziraj labele za train skup
    visualize_labels(
        images_dir=os.path.join(dataset_dir, 'train/images'),
        labels_dir=os.path.join(dataset_dir, 'train/labels'),
        output_dir='annotated_images/'
    )
    
    print("Sve operacije završene!")