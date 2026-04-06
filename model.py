import cv2
import json
import os
import torch
import multiprocessing
from ultralytics import YOLO
from datetime import datetime
import time

# Load konfigurasi kamera dari file JSON
with open('data.json', 'r') as file:
    cameras = json.load(file)

def process_camera(camera):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO('DevSynapse-hybrid.pt').to(device)
    fire_model = YOLO('firedetect-11x.pt').to(device)

    cap = cv2.VideoCapture(camera['url'])

    if not cap.isOpened():
        print(f"❌ Gagal membuka kamera: {camera['name']}")
        return

    os.makedirs(camera['path'], exist_ok=True)
    print(f"✅ Memproses {camera['name']} di {device}...")

    last_capture_time = 0
    capture_delay = 5  # detik antar-capture
    violation_grace_period = 0.75  # waktu tunda sebelum menyimpan ss
    violation_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()  # ⬅️ Simpan versi tanpa bounding box

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        class_names = results[0].names

        fire_results = fire_model(frame)
        fire_detections = fire_results[0].boxes.data.cpu().numpy()
        fire_class_names = fire_results[0].names

        save_violation = False
        violation_labels = set()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf > 0.81:
                label_name = class_names[cls]
                label_text = f"{label_name} {conf:.2f}"
                color = (0, 255, 0)

                if label_name == 'Non Helmet':
                    color = (0, 0, 255)
                    save_violation = True
                    violation_labels.add('No Helmet')
                elif label_name == 'Non Coverall':
                    color = (0, 0, 255)
                    save_violation = True
                    violation_labels.add('No Coverall')

                # Gambar bounding box dan label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for det in fire_detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf > 0.81:
                label_name = fire_class_names[cls]
                label_text = f"{label_name} {conf:.2f}"
                color = (0, 69, 255)  # Orange-red for fire

                save_violation = True
                violation_labels.add(label_name)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        current_time = time.time()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H;%M;%S')

        if save_violation:
            if violation_start_time is None:
                violation_start_time = current_time
            elif current_time - violation_start_time >= violation_grace_period:
                if current_time - last_capture_time > capture_delay:
                    # Tentukan folder pelanggaran (dengan bounding box)
                    ppe_labels = violation_labels & {'No Helmet', 'No Coverall'}
                    fire_labels = violation_labels - {'No Helmet', 'No Coverall'}

                    if ppe_labels == {'No Helmet', 'No Coverall'}:
                        violation_folder = os.path.join(camera['path'], 'No Helmet & No Coverall')
                    elif 'No Helmet' in ppe_labels:
                        violation_folder = os.path.join(camera['path'], 'No Helmet')
                    elif 'No Coverall' in ppe_labels:
                        violation_folder = os.path.join(camera['path'], 'No Coverall')
                    elif fire_labels:
                        violation_folder = os.path.join(camera['path'], 'Fire')
                    else:
                        violation_folder = camera['path']

                    os.makedirs(violation_folder, exist_ok=True)

                    filename_with_box = os.path.join(
                        violation_folder, f"{camera['name']}_{timestamp}.jpg"
                    )
                    cv2.imwrite(filename_with_box, frame)

                    # Simpan juga ke folder dummy versi tanpa bounding box
                    # dummy_folder = os.path.join('dummy', camera['path'])
                    os.makedirs("dummy", exist_ok=True)

                    filename_raw = os.path.join(
                        "dummy", f"{camera['name']}_{timestamp}.jpg"
                    )
                    cv2.imwrite(filename_raw, raw_frame)

                    last_capture_time = current_time
                    print(f"📸 Pelanggaran {', '.join(violation_labels)} di {camera['name']}!")
                    print(f"   ➤ Dengan bounding box disimpan: {filename_with_box}")
                    print(f"   ➤ Tanpa bounding box disimpan: {filename_raw}")

                    violation_start_time = None  # reset
        else:
            violation_start_time = None  # reset jika tidak ada pelanggaran

    cap.release()

if __name__ == '__main__':
    processes = []
    for cam in cameras:
        p = multiprocessing.Process(target=process_camera, args=(cam,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
