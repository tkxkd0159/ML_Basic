import cv2
import matplotlib.pyplot as plt

# 엽서 이미지에서 우편 번호를 추출하는 함수


def detect_zipno(fname):

    img = cv2.imread(fname)
    img_h, img_w = img.shape[:2]

    img = img[0:img_h//2, img_w//3:] # 이미지의 오른쪽 윗부분만 추출하기

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    im2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]


    cnts = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 추출한 이미지에서 윤곽 추출하기
    result = []
    for pts in cnts:
        img_x, img_y, img_w, img_h = cv2.boundingRect(pts)

        if not 50 < img_w < 70: # 너무 크거나 너무 작은 부분 제거하기
            continue
        result.append([img_x, img_y, img_w, img_h])

    sortkey_first_element = lambda resset: resset[0]
    result = sorted(result, key=sortkey_first_element) # 추출한 윤곽을 위치에 따라 정렬하기

    result2 = []
    lastx = -100
    for img_x, img_y, img_w, img_h in result:
        if (img_x - lastx) < 10: # 추출한 윤곽이 너무 가까운 것들 제거하기
            continue
        result2.append([img_x, img_y, img_w, img_h])
        lastx = img_x

    for img_x, img_y, img_w, img_h in result2:
        cv2.rectangle(img, (img_x, img_y), (img_x+img_w, img_y+img_h), (0, 255, 0), 3)
    return result2, img


if __name__ == '__main__':

    CNTS, IMG = detect_zipno("img/hagaki1.png")

    plt.imshow(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))
    plt.savefig("detect-zip.png", dpi=200)
    plt.show()
