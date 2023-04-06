import cv2
import pytesseract
import mysql.connector
from datetime import datetime
from datetime import date

# Configurar el reconocimiento de caracteres
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract' # Reemplace la ruta con la ubicación de Tesseract en su sistema
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' # Reemplace la ruta con la ubicación de Tesseract en su sistema

# Configurar la conexión con la base de datos
mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="",
  database="placas"
)

mycursor = mydb.cursor()

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
            plate_number_short = plate_number[:7]
            #print("PLACA:", plate_number_short, end=" ") # utilizar end=" " para imprimir en la misma línea
            print("PLACA: ", plate_number_short)

            # Realizar la búsqueda en la base de datos
            placa_buscar = plate_number_short
            sql = "SELECT id FROM vehicles WHERE placa = %s"
            val = (placa_buscar,)
            mycursor.execute(sql, val)

            # Obtener los resultados de la búsqueda
            result = mycursor.fetchone()
            if result is not None:
                print("Se encontró la placa en la base de datos.", " El id es: " + str(result[0]))
                now = datetime.now() # obtenemos la hora y fecha acual
                today = datetime.today() # obtenemos la fecha acual

                # Realizar la búsqueda en la base de datos para ver si el vehiculo ingreso el dia de hoy
                id_vehicle = result[0]
                today_buscar = today.strftime('%Y-%m-%d')
                sql2 = "SELECT id, salida FROM controls WHERE id_vehicle = %s AND fecha = %s ORDER BY id DESC LIMIT 1"
                val2 = (id_vehicle,today_buscar)
                mycursor.execute(sql2, val2)

                # Obtener los resultados de la búsqueda
                result2 = mycursor.fetchone()
                if result2 is not None:
                    if result2[1] is None:
                        sql4 = "UPDATE controls SET salida = %s, updated_at = %s WHERE id = %s"
                        val4 = (now,now,result2[0])
                        mycursor.execute(sql4, val4)
                        mydb.commit()
                    else:
                        # Insertar el número de placa en la base de datos
                        sql3 = "INSERT INTO controls (id_vehicle,ingreso,fecha,created_at,updated_at) VALUES (%s,%s,%s,%s,%s)"
                        val3 = (id_vehicle, now, today, now, now)
                        mycursor.execute(sql3, val3)
                        mydb.commit()
                else:
                    print("No se encontraron resultados para la búsqueda")
                    # Insertar el número de placa en la base de datos
                    sql3 = "INSERT INTO controls (id_vehicle,ingreso,fecha,created_at,updated_at) VALUES (%s,%s,%s,%s,%s)"
                    val3 = (id_vehicle,now,today,now,now)
                    mycursor.execute(sql3, val3)
                    mydb.commit()
            else:
                print("No se encontró la placa en la base de datos.")
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

# Cerrar la conexión con la base de datos al final del programa
mydb.close()
# Liberar los recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()