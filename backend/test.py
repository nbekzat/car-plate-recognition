from car_plate_detection import read_img, detect_with_yolo


def test():
    img = read_img("data/2.jpg")
    text_list, conf_list = detect_with_yolo(img)
    print(text_list)
    print(conf_list)


test()
