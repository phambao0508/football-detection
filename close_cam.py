import cv2
import time
import numpy as np
import os

try:
    import pytesseract
except ImportError:
    print("Do not have pytesseract")

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics error")
    class YOLO:
        def __init__(self, weights): pass
        def predict(self, frame, conf, verbose): return []


WEIGHTS_PATH = '/content/drive/MyDrive/Model_Bảo/jersey_detection.pt'
FACE_DETECTION_WEIGHTS = '/content/drive/MyDrive/Model_Bảo/face_detection.pt' 
VIDEO_SOURCE = '/content/drive/MyDrive/Model_Bảo/test - Made with Clipchamp.mp4'
CONFIDENCE_THRESHOLD = 0.5
TESSERACT_PATH = '/usr/bin/tesseract'
OUTPUT_VIDEO_PATH = 'output_ocr_face_detection.mp4'


CLASS_LABELS = {
    0: 'Ball',
    1: 'Jersey',
    2: 'Person',
    3: 'Face',  
}

COLOR_MAP = {
    0: (0, 0, 255),
    1: (36, 237, 58),
    2: (255, 255, 0),
    3: (255, 0, 255), 
}


if 'pytesseract' in globals() and TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    if TESSERACT_PATH:
        print(f"Cannot found Pytessract at {TESSERACT_PATH}.")

def run_ocr_on_region(image_crop):
    if 'pytesseract' not in globals():
        return "OCR_ERR"

    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    ocr_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'

    try:
        text = pytesseract.image_to_string(gray, config=ocr_config)
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return "OCR_FAIL"


class YOLODetector:
    def __init__(self, primary_weights_path, face_weights_path):
        self.primary_model = self._load_model(primary_weights_path, "Primary Model")
        self.face_model = self._load_model(face_weights_path, "Face Model")
    
    def _load_model(self, weights_path, model_name):
        try:
            model = YOLO(weights_path)
            print(f"Loaded {model_name} successfully.")
            return model
        except Exception as e:
            print(f"Cannot load {model_name}: '{weights_path}'. Error: {e}")
            return None

    def predict(self, frame):
        all_detections = []

        if self.primary_model is not None:
            primary_results = self.primary_model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            all_detections.extend(self._extract_detections(primary_results, offset_class_id=0))

       
        if self.face_model is not None:
            face_results = self.face_model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            all_detections.extend(self._extract_detections(face_results, offset_class_id=3))

        return all_detections

    def _extract_detections(self, results, offset_class_id=0):
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0]) + offset_class_id 

                detections.append((x1, y1, x2, y2, confidence, class_id))

        return detections

def process_video():
    detector = YOLODetector(
        primary_weights_path=WEIGHTS_PATH, 
        face_weights_path=FACE_DETECTION_WEIGHTS
    )
    try:
        if VIDEO_SOURCE == '0':
            print("Error: Webcam input ('0') is not supported in this environment.")
            return
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    except Exception as e:
        print(f"Cannot start VideoCapture {e}")
        return

    if not cap.isOpened():
        print("Cannot open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"Start to process: '{VIDEO_SOURCE}'.")
    print(f"The output will be saved at: {OUTPUT_VIDEO_PATH}")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame.ndim == 2:
             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        detected_objects = detector.predict(frame)

        for (x1, y1, x2, y2, conf, class_id) in detected_objects:

            label_text = CLASS_LABELS.get(class_id, f"Class {class_id}")
            color = COLOR_MAP.get(class_id, (255, 0, 0))

            recognized_number = ""
            if class_id == 1:
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)

                jersey_crop = frame[y1:y2, x1:x2]

                if jersey_crop.size > 0 and jersey_crop.shape[0] > 10 and jersey_crop.shape[1] > 10:
                    recognized_number = run_ocr_on_region(jersey_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if class_id == 1 and recognized_number and recognized_number not in ["OCR_FAIL", "OCR_ERR"]:
                display_label = f"Number: {recognized_number} ({conf:.2f})"
            else:
                display_label = f"{label_text}: {conf:.2f}"

            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        elapsed_time = time.time() - start_time
        fps_calc = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps_calc:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        video_writer.write(frame)

    cap.release()
    video_writer.release()

    print(f"\nProcessing done")
    print(f"The number of processed frames is: {frame_count}")
    print(f"Saved at: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()