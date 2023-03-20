import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un frame
    ret, frame = cap.read()

    # Convertir a escala de grises y aplicar un filtro Gaussiano
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)

    # Detección de bordes y contornos
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar contornos relevantes y dibujar un rectángulo alrededor de la placa de vehículo
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 1.5:
                # Extraer la placa de la imagen original
                plate = cv2.resize(frame[y:y+h, x:x+w], (400, 100))

                # Utilizar técnicas de procesamiento de imágenes para mejorar la visibilidad de la placa
                plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                _, plate_binary = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # Utilizar OCR para detectar y leer el texto en la placa
                text = pytesseract.image_to_string(plate_binary, lang="eng", config="--psm 7")

                # Verificar que el texto detectado es un número de matrícula válido
                if text.isalnum() and len(text) >= 6:
                    print("Plate Number:", text)

    # Mostrar el resultado
    cv2.imshow("Plate Detection", frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar la cámara y todas las ventanas
cap.release()
cv2.destroyAllWindows()