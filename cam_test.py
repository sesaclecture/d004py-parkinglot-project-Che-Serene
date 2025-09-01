import cv2
import sys
import pytesseract

cap_dev_id = 0
try:
    cap_dev_id = sys.argv[1]
    cap_dev_id = int(sys.argv[1])
except IndexError:
    print(f"Info: Using default camera 0")
    cap_dev_id = 0
except ValueError as e:
    print(f"Error: {e}")
    exit()


# Open the camera.
cap = cv2.VideoCapture(cap_dev_id)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
print("Press 'q' to quit")

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours :
        x, y, w, h = cv2.boundingRect(contour)

        if 200 < w < 350 and 50 < h < 100 :
            roi = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, lang = 'kor', config='--psm 6').strip()
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),2)

    cv2.imshow('Number Detection', frame)

    if cv2.waitKey(1) &0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()