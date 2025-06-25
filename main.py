import cv2
import pytesseract
import imutils
import numpy as np
import mysql.connector
import re
from datetime import datetime, date
import os
from dotenv import load_dotenv
import platform

# Cargar variables de entorno
load_dotenv()

# Configurar ruta de Tesseract si es Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

# Conexión a base de datos
try:
    mydb = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    print(f"❌ Error al conectar a la base de datos: {err}")
    exit(1)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit(1)

print("🔍 Iniciando detector de placas peruanas. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo acceder a la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / h
            if 2 < ratio < 6:
                plate_contour = approx
                break

    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_img = frame[y:y + h, x:x + w]

        # Mejorar OCR: escala de grises + binarización
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Aplicar OCR
        text_raw = pytesseract.image_to_string(thresh, lang='eng', config='--psm 7')
        text_clean = re.sub(r'[^A-Z0-9]', '', text_raw.upper())

        # Buscar patrones válidos de placas peruanas
        matches = re.findall(r'\b([A-Z0-9]{3,4}[0-9]{3})\b', text_clean)

        if matches:
            plate = matches[0]
            print("🚗 PLACA DETECTADA:", plate)

            sql = "SELECT id FROM vehicles WHERE placa = %s"
            mycursor.execute(sql, (plate,))
            result = mycursor.fetchone()

            if result:
                id_vehicle = result[0]
                now = datetime.now()
                today = date.today().strftime('%Y-%m-%d')

                sql2 = "SELECT id, salida FROM controls WHERE id_vehicle = %s AND fecha = %s ORDER BY id DESC LIMIT 1"
                mycursor.execute(sql2, (id_vehicle, today))
                result2 = mycursor.fetchone()

                if result2 and result2[1] is None:
                    sql_out = "UPDATE controls SET salida = %s, updated_at = %s WHERE id = %s"
                    mycursor.execute(sql_out, (now, now, result2[0]))
                    mydb.commit()
                    print("⏺️ Salida registrada.")
                else:
                    sql_in = "INSERT INTO controls (id_vehicle, ingreso, fecha, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)"
                    mycursor.execute(sql_in, (id_vehicle, now, today, now, now))
                    mydb.commit()
                    print("✅ Ingreso registrado.")
            else:
                print("❌ La placa no está registrada en la base de datos.")

            # Dibujar y guardar imagen
            os.makedirs("placas_detectadas", exist_ok=True)
            filename = f"placas_detectadas/{plate}_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(filename, plate_img)

            # Mostrar visualmente
            cv2.putText(frame, plate, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.drawContours(frame, [plate_contour], -1, (0, 255, 0), 2)

    cv2.imshow("Detector de Placas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
mydb.close()
cv2.destroyAllWindows()