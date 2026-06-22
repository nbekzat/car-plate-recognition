import cv2
import pytesseract
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
pytesseract_config = (
    r"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\  --psm 8"
)


def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def read_img(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}. Check file path exists.")
        return None
    return image


def detect_with_yolo(image):

    # load pretrained YOLO
    license_plate_detector = YOLO("license_plate_detector.pt")

    license_plates = license_plate_detector(image)[0]
    license_plate_cnt = license_plates.boxes.data.tolist()

    # no number plate is detected
    if license_plate_cnt == 0:
        return None

    plate_number_list = []
    for license_plate in license_plate_cnt:
        x1, y1, x2, y2, score, class_id = license_plate

        # crop license plate
        license_plate_crop = image[int(y1) : int(y2), int(x1) : int(x2), :]

        # process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
        )

        # show_img(license_plate_crop_thresh)

        plate_number = pytesseract.image_to_string(
            license_plate_crop_thresh, config=pytesseract_config
        )

        # is text is not empty
        if plate_number.strip():
            plate_number_list.append(plate_number)

    return plate_number_list


def detect_with_edge(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    # show_img(gray)

    # step 2
    edges = cv2.Canny(gray, 20, 120)
    # show_img(edges)

    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # process top 10 largest rectagles
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour_lst = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )  # Approximates the contour to a polygon with fewer vertices.

        if len(approx) == 4:
            # plate_contour_lst.append(approx)
            x, y, w, h = cv2.boundingRect(approx)
            plate_image = gray[y : y + h, x : x + w]

            # show_img(plate_image)
            _, thresh = cv2.threshold(
                plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # show_img(thresh)

            plate_number = pytesseract.image_to_string(
                thresh, config=pytesseract_config
            )
            # print(plate_number)
            if plate_number:
                plate_contour_lst.append(plate_number)
                break

    return plate_contour_lst


def detect_with_ensemble(image: Image.Image):
    # image is a numpy array from OpenCV (already in BGR format)
    car_plate_number_text_list = detect_with_yolo(image)

    # if YOLO wasn't able to idetify the car plate then use Edge detection
    if not car_plate_number_text_list:
        print(
            "Car plate number is not detected with YOLO. Working with OCR edge detection."
        )
        car_plate_number_text_list = detect_with_edge(image)

    print("detected car plate number: ", car_plate_number_text_list)

    return car_plate_number_text_list


# image_path = "data/car2.jpg"
# detect_with_ensemble(image_path)
