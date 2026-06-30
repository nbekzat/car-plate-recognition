import cv2
from backend.car_plate_detection_tesseract import read_img, detect_with_yolo


def test():
    img = read_img("data/Cars3.png")
    text_list, conf_list, crop_list = detect_with_yolo(img)
    print(text_list)
    print(conf_list)
    for idx, crop in enumerate(crop_list):
        cv2.imwrite(f"data/crop_{idx}.jpg", crop)


test()
