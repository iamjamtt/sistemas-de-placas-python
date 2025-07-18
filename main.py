import cv2
import pytesseract
import numpy as np
import mysql.connector
import re
from datetime import datetime, date, timedelta
import os
from PIL import Image
from dotenv import load_dotenv
import platform
import time

# Cargar variables de entorno
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

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

def guardar_capturas(plate, carpeta_base, carpeta_destino, captura_main, captura_secondary):
    fecha_str = datetime.now().strftime("%Y-%m-%d")
    ruta = os.path.join(carpeta_base, carpeta_destino, fecha_str)
    ruta_cam1 = ruta
    ruta_cam2 = f"{ruta}_cam2"

    os.makedirs(ruta_cam1, exist_ok=True)
    os.makedirs(ruta_cam2, exist_ok=True)

    nombre_archivo = f"{plate}_{datetime.now().strftime('%H%M%S')}.png"
    ruta_bd_cam1 = os.path.join(ruta_cam1, nombre_archivo)
    ruta_bd_cam2 = os.path.join(ruta_cam2, nombre_archivo)

    if captura_main is not None:
        cv2.imwrite(ruta_bd_cam1, captura_main)
    if captura_secondary is not None:
        cv2.imwrite(ruta_bd_cam2, captura_secondary)

    return ruta_bd_cam1, ruta_bd_cam2

class PlateDetector:
    def __init__(self, cam_index_main=1, cam_index_secondary=2):
        self.cap_main = cv2.VideoCapture(cam_index_main)
        self.cap_secondary = cv2.VideoCapture(cam_index_secondary)
        if not self.cap_main.isOpened():
            raise IOError("❌ No se pudo abrir la cámara principal.")
        if not self.cap_secondary.isOpened():
            raise IOError("❌ No se pudo abrir la cámara secundaria.")

        print("🔍 Iniciando detector de placas peruanas con dos cámaras. Presiona 'q' para salir.")
        self.Ctexto = ''
        self.mensaje = ''
        self.tiempo_mensaje = 0
        self.ultima_captura_main = None
        self.ultima_captura_secondary = None
        self.placa_con_sancion = False

    def preprocess(self, frame):
        al, an, _ = frame.shape
        x1, x2 = int(an / 3), int(an * 2 / 3)
        y1, y2 = int(al / 3), int(al * 2 / 3)
        return frame[y1:y2, x1:x2], (x1, y1)

    def extract_plate_contours(self, recorte):
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return sorted(contours, key=cv2.contourArea, reverse=True), mask_blue

    def extract_text_from_plate(self, placa):
        gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pil_img = Image.fromarray(bin_img).convert("L")
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text = pytesseract.image_to_string(pil_img, config=config)

        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        return cleaned

    def save_plate_and_log(self, plate):
        carpeta_base = 'files'
        carpeta_destino = "placas_no_detectadas_bd"

        sql = "SELECT id, tieneSancion, id_tipes_sanctions FROM vehicles WHERE placa = %s"
        mycursor.execute(sql, (plate,))
        result = mycursor.fetchone()

        if result:
            id_vehicle, tieneSancion, idSancion = result
            self.placa_con_sancion = False
            if tieneSancion == 1:
                mycursor.execute("SELECT nombre FROM tipes_sanctions WHERE id = %s", (idSancion,))
                sancion = mycursor.fetchone()
                if sancion:
                    print(f"🚨 Vehículo con sanción: {sancion[0]}")
                self.placa_con_sancion = True

            now = datetime.now()
            today = date.today().strftime('%Y-%m-%d')
            two_minutes_ago = now - timedelta(minutes=2)

            mycursor.execute("SELECT id, ingreso, salida FROM controls WHERE id_vehicle = %s AND fecha = %s ORDER BY id DESC LIMIT 1", (id_vehicle, today))
            control = mycursor.fetchone()

            if control:
                control_id, ingreso_time, salida_time = control
                if salida_time is None:
                    if ingreso_time and ingreso_time >= two_minutes_ago:
                        print("⚠️ Detección ignorada (ingreso reciente).")
                        return
                    carpeta_destino = "placas_detectadas_salida"
                    ruta1, ruta2 = guardar_capturas(plate, carpeta_base, carpeta_destino, self.ultima_captura_main, self.ultima_captura_secondary)
                    mycursor.execute("UPDATE controls SET salida = %s, updated_at = %s, foto_salida = %s, foto_salida_2 = %s WHERE id = %s", (now, now, ruta1, ruta2, control_id))
                    mydb.commit()
                    print("⏺️ Salida registrada.")
                    self.mensaje = f"Placa {plate} registrada (salida)"
                else:
                    if salida_time and salida_time >= two_minutes_ago:
                        print("⚠️ Detección ignorada (salida reciente).")
                        return
                    carpeta_destino = "placas_detectadas_ingreso"
                    ruta1, ruta2 = guardar_capturas(plate, carpeta_base, carpeta_destino, self.ultima_captura_main, self.ultima_captura_secondary)
                    mycursor.execute("INSERT INTO controls (id_vehicle, ingreso, fecha, foto_ingreso, foto_ingreso_2, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                     (id_vehicle, now, today, ruta1, ruta2, now, now))
                    mydb.commit()
                    print("✅ Ingreso registrado.")
                    self.mensaje = f"Placa {plate} registrada (ingreso)"
            else:
                carpeta_destino = "placas_detectadas_ingreso"
                ruta1, ruta2 = guardar_capturas(plate, carpeta_base, carpeta_destino, self.ultima_captura_main, self.ultima_captura_secondary)
                mycursor.execute("INSERT INTO controls (id_vehicle, ingreso, fecha, foto_ingreso, foto_ingreso_2, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                 (id_vehicle, now, today, ruta1, ruta2, now, now))
                mydb.commit()
                print("✅ Ingreso registrado.")
                self.mensaje = f"Placa {plate} registrada (ingreso)"
        else:
            print("❌ La placa no está registrada.")
            self.mensaje = f"Placa {plate} no registrada"
            guardar_capturas(plate, carpeta_base, carpeta_destino, self.ultima_captura_main, self.ultima_captura_secondary)

        self.tiempo_mensaje = time.time()

    def run(self):
        while True:
            ret_main, frame_main = self.cap_main.read()
            ret_sec, frame_sec = self.cap_secondary.read()

            if not ret_main or not ret_sec:
                print("⚠️ No se pudo capturar imagen desde ambas cámaras.")
                break

            self.ultima_captura_main = frame_main.copy()
            self.ultima_captura_secondary = frame_sec.copy()

            recorte, offset = self.preprocess(frame_main)
            x_off, y_off = offset

            contours, _ = self.extract_plate_contours(recorte)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 5000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    x1, y1 = x + x_off, y + y_off
                    x2, y2 = x1 + w, y1 + h

                    cv2.rectangle(frame_main, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    placa_img = frame_main[y1:y2, x1:x2]

                    if placa_img.size > 0:
                        text = self.extract_text_from_plate(placa_img)
                        if len(text) == 6:
                            self.Ctexto = text
                            print("🚗 Placa detectada:", text)
                            self.save_plate_and_log(text)
                        break

            # Mostrar texto
            cv2.rectangle(frame_main, (870, 750), (1070, 850), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame_main, self.Ctexto, (880, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar mensaje por 5 segundos
            if self.mensaje and (time.time() - self.tiempo_mensaje < 5):
                color_mensaje = (0, 0, 255) if self.placa_con_sancion else (255, 255, 255)
                cv2.rectangle(frame_main, (40, 20), (850, 60), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame_main, self.mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color_mensaje, 2)

            cv2.imshow("Camara Principal", frame_main)
            cv2.imshow("Camara Secundaria", frame_sec)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap_main.release()
        self.cap_secondary.release()
        mydb.close()
        cv2.destroyAllWindows()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    cam1 = int(os.getenv("CAM_1", 1))
    cam2 = int(os.getenv("CAM_2", 2))
    detector = PlateDetector(cam_index_main = cam1, cam_index_secondary = cam2)
    detector.run()
