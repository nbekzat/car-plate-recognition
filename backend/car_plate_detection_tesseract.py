import cv2
from ultralytics import YOLO
import pytesseract
import platform
import torch
# import matplotlib.pyplot as plt

# if app is running from Mac
if platform.system() == "Darwin":
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

pytesseract_config = (
    r"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-\ --psm 6"
)

license_plate_detector = YOLO("models/license-plate-finetune-v1n.pt")


# def show_img(img):
#     plt.imshow(img, cmap="gray")
#     plt.axis("off")
#     plt.show()


def save_img(img):
    img.save("data/crop_car1.jpg")


def detect_car_plate_text(car_plate_crop):
    print("detecting text from crop with tesseract...")

    gray = cv2.cvtColor(car_plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plate_number = pytesseract.image_to_string(thresh, config=pytesseract_config)
    print(plate_number)

    return plate_number


def read_img(image_path: str):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}. Check file path exists.")
        return None
    return image


def detect_with_yolo(image):

    with torch.no_grad():
        results = license_plate_detector(image)

    plate_number_list = []
    conf_list = []
    plate_crop_img = []
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # crop license plate
            plate_crop = image[int(y1) : int(y2), int(x1) : int(x2)]
            plate_text = detect_car_plate_text(plate_crop)
            plate_number_list.append(plate_text)
            conf_list.append(conf)
            plate_crop_img.append(plate_crop)

    del results
    return plate_number_list, conf_list, plate_crop_img
