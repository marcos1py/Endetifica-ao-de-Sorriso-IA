import cv2

# Carrega o classificador para detecção de sorrisos
sorriso_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Inicia a captura de vídeo da câmera
camera = cv2.VideoCapture(0)

# Define um contador para o número de frames sorrindo
cont_sorrisos = 0

while True:
    # Captura um quadro do vídeo
    ret, frame = camera.read()
    
    # Converte o quadro para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta sorrisos no quadro
    detecoes = sorriso_detector.detectMultiScale(frame_cinza, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    
    # Desenha retângulos em torno das detecções de sorrisos
    for (x, y, w, h) in detecoes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Incrementa o contador de sorrisos
        cont_sorrisos += 1
    
    # Exibe o quadro com as detecções
    cv2.imshow('Detecção de Sorriso', frame)
    
    # Se detectar um sorriso por mais de 10 frames, exibe mensagem
    if cont_sorrisos > 10:
        print("Você está sorrindo!")
        cont_sorrisos = 0
    
    # Espera pela tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
