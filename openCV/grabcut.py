import cv2
import numpy as np

def onMouse(event, x, y, flags, param):
    global ix, iy, orig_img, img2show, mouse_pressed, drawing, value, mask
    global rect, rect_or_mask, rect_over

    if event == cv2.EVENT_RBUTTONDOWN:
        mouse_pressed = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            img2show = orig_img.copy()
            cv2.rectangle(img2show, (ix, iy), (x, y), (0, 0, 255), 2)


    elif event == cv2.EVENT_RBUTTONUP:
        mouse_pressed = False
        rect_over = True
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0
        print('a : apply')


    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_over:
            print('마우스 왼쪽 버튼을 누른채로 전경이 되는 부분을 선택하세요')
        else:
            drawing = True
            cv2.circle(img2show, (x, y), 3, value['color'], -1)
            cv2.circle(mask, (x, y), 3, value['var'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img2show, (x, y), 3, value['color'], -1)
            cv2.circle(mask, (x, y), 3, value['var'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img2show, (x, y), 3, value['color'], -1)
        cv2.circle(mask, (x, y), 3, value['var'], -1)


orig_img = cv2.imread('img/nicedog.jpg')
img2show = orig_img.copy()

mouse_pressed = False
drawing = False
rect_over = False
rect_or_mask = 100
rect = (0, 0, 1, 1)
ix = iy = 0
BG = {'color' : (0, 0, 0), 'var' : 0}
FG = {'color' : (255, 255, 255), 'var' : 1}
value = BG

mask = np.zeros(orig_img.shape[:2], dtype = np.uint8)
output = np.zeros(orig_img.shape, np.uint8)

cv2.namedWindow('input')
cv2.setMouseCallback('input', onMouse)

cv2.namedWindow('output')
cv2.moveWindow('output', orig_img.shape[1]*2, 90)

print("오른쪽 마우스 버튼을 누르고 영역을 지정한 후 a를 누르세요")


while True:
    cv2.imshow('input', img2show)
    cv2.imshow('output', output)

    k = cv2.waitKey(1)

    if k == 27:
        break

    elif k == ord('0'):
        print("왼쪽 마우스로 제거할 부분을 표시한 후 a를 누르세요")
        value = BG

    elif k == ord('1'):
        print("왼쪽 마우스로 복원할 부분을 표시한 후 a를 누르세요")
        value = FG

    elif k == ord('r'):
        print("reset")
        rect = (0, 0, 1, 1)
        drawing = False
        mouse_pressed = False
        rect_or_mask = 100
        rect_over = False
        value = BG
        img2show = orig_img.copy()
        mask = np.zeros(orig_img.shape[:2], dtype = np.uint8)
        output = np.zeros(orig_img.shape, np.uint8)
        print("0: 제거배경선택 1:복원전경선택 a:apply r:reset")

    elif k == ord('a'):
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        if rect_or_mask == 0:
            cv2.grabCut(orig_img, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1

        elif rect_or_mask == 1:
            cv2.grabCut(orig_img, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_MASK)

        print("0: 제거배경선택 1:복원전경선택 a:apply r:reset")


    # 확실한 전경(1) + 덜 확실한 전경(3) -> 1로 통일시킴. 배경은 0으로
    mask2 = np.where((mask == 3)|(mask == 1), 1, 0).astype('uint8')


    output = cv2.bitwise_and(orig_img, orig_img, mask=mask2)



cv2.destroyAllWindows()