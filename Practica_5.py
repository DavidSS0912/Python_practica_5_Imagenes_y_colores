import cv2


def run():
    cap = cv2.VideoCapture(0)

    while(1):
        umbral = 100
        _, frame = cap.read()
        webCam = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)
        # Pasando la imagen de webCam a Blanco y Negro
        webCamGray = cv2.cvtColor(webCam, cv2.COLOR_BGR2GRAY)
        # Detectar bordes usando Canny
        cny = cv2.Canny(webCamGray, umbral, umbral * 2)
        # Buscar contornos
        contornos, jerarquía = cv2.findContours(
            cny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('webCam', cny)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if "__main__" == __name__:
    run()
