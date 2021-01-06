import numpy as np
import cv2


def getContours(img):
    # 高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # 转化成灰度图
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 图像二值化
    _, thrash = cv2.threshold(imgGrey, 100, 255, cv2.THRESH_BINARY)
    # canny = cv2.Canny(imgGrey, 100, 150)
    # 图像的腐蚀和膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.erode(thrash, kernel)
    dilate = cv2.dilate(erode, kernel)

    cv2.imshow("thrash", thrash)
    # cv2.imshow("canny", canny)
    cv2.imshow("erode", erode)
    cv2.imshow("dilate", dilate)

    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 0), 3)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        elif len(approx) == 4:
            x1 ,y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
              cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            else:
              cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        elif len(approx) == 10:
            cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        else:
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    return img


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        cv2.imshow("Test", frame)
        frame = getContours(frame)
        cv2.imshow("Contour", frame)
        keys = cv2.waitKey(1)
        if keys == ord('q') or not success:
            break
    cap.release()
    cv2.destroyAllWindows()
    # img = cv2.imread('shape.jpg')
    # cv2.imshow("origin", img)
    # img = getContours(img)
    # cv2.imshow("output", img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()

