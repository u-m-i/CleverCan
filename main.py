import cv2
import numpy as np

# Crear un filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)  # 4 estados (posición x, posición y, velocidad x, velocidad y), 2 mediciones (posición x, posición y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Matriz de medición
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Matriz de transición
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03  # Covarianza del ruido del proceso

# Inicializar las variables de seguimiento
last_measurement = np.array([[2], [1]], np.float32)  # Última medición de posición
last_prediction = np.array([[2], [1]], np.float32)  # Última predicción de posición

def kalman_filter(frame):
    # Convertir la imagen a escala de grises y aplicar un filtro gaussiano para suavizarla
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de contornos
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (suponiendo que es la pelota)
    if len(contours) > 0:
        ball_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball_contour)

        # Medición
        measurement = np.array([[x], [y]], np.float32)

        # Predicción
        prediction = kalman.predict()

        # Corrección
        if radius > 10:
            estimated = kalman.correct(measurement)
        else:
            estimated = prediction

        # Dibujar la trayectoria estimada
        cv2.circle(frame, (int(estimated[0]), int(estimated[1])), 10, (0, 255, 0), 2)
        
        # Actualizar las variables de seguimiento
        last_measurement = measurement
        last_prediction = prediction

    return frame

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        frame = kalman_filter(frame)
        
        cv2.imshow('Kalman Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
