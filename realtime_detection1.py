from ultralytics import YOLO
import cv2
import pyttsx3

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speech speed
engine.setProperty("volume", 1.0)  # Max volume

# Start webcam
cap = cv2.VideoCapture(0)

# Keep track of last spoken object to avoid repeating constantly
last_object = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.4)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Get detected object names
    names = results[0].names
    detected_classes = results[0].boxes.cls.tolist()

    if detected_classes:
        # Take first detected object (you can change logic if needed)
        obj_name = names[int(detected_classes[0])]
        
        # Only speak if object changed
        if obj_name != last_object:
            message = f"{obj_name} detected ahead"
            print(message)
            engine.say(message)
            engine.runAndWait()
            last_object = obj_name

    # Show frame
    cv2.imshow("YOLOv8n Realtime Detection with Voice", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
