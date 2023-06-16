import cv2
import imutils
from imutils import contours
import numpy as np


def show_image(img):
    cv2.namedWindow("img")
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_ref(path_ocr_a=r'..\proccesor\digits\ocr_a_reference.png', \
            path_ocr_b=r'..\proccesor\digits\ocr_b_reference.png', \
            path_ocr_c=r'..\proccesor\digits\ocr_c_reference.png'):
    ref_a = cv2.imdecode(np.fromfile(path_ocr_a, dtype=np.uint8), cv2.IMREAD_COLOR)
    ref_b = cv2.imdecode(np.fromfile(path_ocr_b, dtype=np.uint8), cv2.IMREAD_COLOR)
    ref_c = cv2.imdecode(np.fromfile(path_ocr_c, dtype=np.uint8), cv2.IMREAD_COLOR)
    digits = {}
    for num, reference in enumerate([ref_a, ref_b, ref_c]):
        ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 100, 255, cv2.THRESH_BINARY)[1]
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        for (i, c) in enumerate(refCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            digits[num * 10 + i] = roi
    return digits


def recognize(image):
    digits = get_ref()
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    locs = []

    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if ar > 4.0 and ar < 40.0:
            if (w >= 200 and w < 290) and (h > 10 and h < 35):
                locs.append((x, y, w, h))
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    groupOutput = []
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        groupOutput = []
        y_lb = gY
        y_ub = gY + gH
        x_lb = gX
        x_ub = gX + gW
        blur = cv2.bilateralFilter(image, 7, 75, 75)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        for j in range(4):
            group = gray[y_lb - 1:y_ub + 1, x_lb + 3 + j * (gW - 3) // 4:x_ub - 3 - (3 - j) * (gW - 3) // 4]
            group = cv2.threshold(group, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            group = cv2.dilate(group, kernel, iterations=1)
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
            for c in digitCnts:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))
                scores = []
                for (digit, digitROI) in digits.items():
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                groupOutput.append(str(np.argmax(scores) % 10))
        cv2.rectangle(image, (gX, gY - 5),
                      (gX + gW, gY + gH + 5), (0, 0, 255), 2)
        text_photo = ' ' + ' '.join(
            [''.join(groupOutput[i:i + 4]) for i in range(0, len(groupOutput), 4)]) if ' ' not in groupOutput[
                                                                                                  :4] and len(
            groupOutput) > 4 else ' ' + ''.join(groupOutput)
        print(text_photo)
        cv2.putText(image, text_photo, (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        output.extend(groupOutput)

    return image, "".join(groupOutput)