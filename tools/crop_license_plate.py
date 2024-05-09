from ultralytics import YOLO
import cv2
import os 
import sys

save_folder = "./data_save/license_plate/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Created folder: {save_folder}")

def crop_license_image(img, models) :
    model = YOLO(f'{str(models)}') 
    images = cv2.imread(f'{str(img)}')
    confident = 0.75
    input_size = (640, 640)
    list_name = []
    img = cv2.resize(images, input_size)
    results = model.predict(img, confident)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            roi = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            filename = f'./data_save/license_plate/license_plate_save.jpg'
            list_name.append(filename)
            cv2.imwrite(filename, roi)
    return filename        

def crop_license_video(video, model) :
    model = YOLO(f'{str(model)}') 
    cap = cv2.VideoCapture(f'{str(video)}')
    confident = 0.80
    list_name = []
    input_size = (640, 640)
    i = 0
    while True:
        i = i+1
        if i % 60 == 0 : 
            ret, img = cap.read()
            if not ret:
                print("No more video frames. Exiting...")
                break
            img = cv2.resize(img, input_size)
            results = model.predict(img, conf = confident)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    roi = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    filename = f'./data_save/license_plate/license_plate_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'
                    
                    list_name.append(filename)
                    cv2.imwrite(filename, roi)
    cap.release()
    cv2.destroyAllWindows()
    return list_name

if __name__ == "__main__" :
    media_path = sys.argv[1]
    model_path = sys.argv[2]
    if media_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        crop_license_video(media_path, model_path)
    else:
        crop_license_image(media_path, model_path)
    
