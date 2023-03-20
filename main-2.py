import cv2
import pytesseract

# Configurar el reconocimiento de caracteres
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' # Reemplace la ruta con la ubicación de Tesseract en su sistema

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro gaussiano para reducir el ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar los bordes de la placa del vehículo
    edged = cv2.Canny(gray, 50, 200)

    # Buscar contornos en la imagen
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por área descendente
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Buscar la placa del vehículo entre los contornos
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        # Recortar la imagen de la placa del vehículo
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_img = frame[y:y+h, x:x+w]

        # Aplicar OCR para extraer el número de placa
        plate_number = pytesseract.image_to_string(plate_img, lang='eng', config='--psm 11')
        if len(plate_number) == 8:
            print("PLACA: " + plate_number)
        # print("PLACA: " + plate_number)

# Dibujar un rectángulo alrededor de la placa del vehículo
        cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 2)

        # Escribir el número de placa en la imagen
        cv2.putText(frame, plate_number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow('Placa de vehiculo', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()