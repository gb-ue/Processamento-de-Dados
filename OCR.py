import Levenshtein
import cv2 as cv
from PIL import Image
import easyocr

#Abrir um caminho pra pasta de imagens
#Fazer a extração de cada texto de imagem junto da sua descrição e nome
# Enviar para a llm

reader = easyocr.Reader(lang_list= "pt", gpu=True)

import numpy as np

def processing(img, max_size=1600):
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 3.0)
    new_w, new_h = int(w * scale), int(h * scale)
    gray = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return img, gray

def detect_text_boxes(gray):
    h,w = gray.shape[:2]
    img_area = h * w

    grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    grad_x = cv.convertScaleAbs(grad_x)

    _, th = cv.threshold(grad_x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


    kernel_w = max(3, int(w * 0.02))
    kernel_h = max(3, int(h * 0.01))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_w, kernel_h))
    closed = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel, iterations=1)

    closed = cv.dilate(closed, kernel, iterations=1)

    cnts, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    min_area = max(10, img_area * 1e-5)

    for c in cnts:
        x,y,ww,hh = cv.boundingRect(c)
        area = ww*hh
        if area < min_area:
            continue

        if hh < 10:
            continue
        ar = ww / float(hh + 1e-6)
        if ar < 0.2 and ar > 40:
            continue
        raw_boxes.append([x,y,x+ww,y+hh])

    boxes = merge_boxes_iou(raw_boxes, iou_thresh=0.35)

    return boxes, closed

def iou(boxA, boxB):
    xA1,yA1,xA2,yA2 = boxA
    xB1,yB1,xB2,yB2 = boxB
    ix1 = max(xA1, xB1)
    iy1 = max(yA1, yB1)
    ix2 = min(xA2, xB2)
    iy2 = min(yA2, yB2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0, (xA2-xA1) * (yA2-yA1))
    areaB = max(0, (xB2-xB1) * (yB2-yB1))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def merge_boxes_iou(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            bx = boxes[i]
            mx1,my1,mx2,my2 = bx
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                by = boxes[j]
                if iou(bx, by) > iou_thresh:
                    mx1 = min(mx1, by[0]); my1 = min(my1, by[1])
                    mx2 = max(mx2, by[2]); my2 = max(my2, by[3])
                    used[j] = True
                    merged = True
            used[i] = True
            new_boxes.append([mx1,my1,mx2,my2])
        boxes = new_boxes
    return boxes

def draw_boxes(img, boxes, color=(0,255,0), thickness=2):
    out = img.copy()
    for (x1,y1,x2,y2) in boxes:
        cv.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    return out

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def deskew_crop(img_bgr, box, binary):
    x1,y1,x2,y2 = box
    roi_bin = binary[y1:y2, x1:x2]
    cnts, _ = cv.findContours(roi_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr[y1:y2, x1:x2]
    c = max(cnts, key=cv.contourArea)
    rect = cv.minAreaRect(c)
    box_pts = cv.boxPoints(rect)
    box_pts[:,0] += x1
    box_pts[:,1] += y1
    src = order_points(box_pts)
    width = int(max(np.linalg.norm(src[0]-src[1]), np.linalg.norm(src[2]-src[3])))
    height = int(max(np.linalg.norm(src[1]-src[2]), np.linalg.norm(src[3]-src[0])))
    if width <= 0 or height <= 0:
        return img_bgr[y1:y2, x1:x2]
    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img_bgr, M, (width, height))
    return warped


def validar_result(texto, processado, tolerancia=0.7):
    texto = texto.strip()
    processado = processado.strip()
    if not processado:
        return False
    similaridade = Levenshtein.ratio(texto, processado)
    return tolerancia <= similaridade

def text_fetch(path_img, iou_merge=0.35, sim_thresh=0.85):
    im = cv.imread(path_img)
    if im is None:
        return ""
    img, gray = processing(im, max_size=1600)
    boxes, binary = detect_text_boxes(gray)
    boxes = sorted(boxes, key=lambda b: (round(b[1]/50), b[0]))

    textos_com_coords = []

    for box in boxes:
        crop = deskew_crop(img, box, binary)
        result_crop = reader.readtext(crop, detail=0, paragraph=False, contrast_ths=0.05, adjust_contrast=0.7)
        if result_crop:
            texto = ' '.join(result_crop).strip()
            if texto:
                textos_com_coords.append((box, texto))

    filtro = []
    for i, (b1, t1) in enumerate(textos_com_coords):
        duplicate = False
        for j, (b2, t2) in enumerate(filtro):
            if iou(b1, b2) > iou_merge and Levenshtein.ratio(b1, b2) > sim_thresh:
                duplicate = True
                break
        if not duplicate:
            filtro.append((b1, t1))

    filtro = sorted(filtro, key=lambda x: (x[0][1], x[0][0]))

    texto_formatado = ' '.join(t for _,t in filtro)

    print(f"Caminho: {path_img}\n")
    print(f"Texto Extraido: {texto_formatado}\n")
