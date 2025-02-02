import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 image from client
            data = await websocket.receive_json()
            image_base64 = data.get('image')
            
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Process image
            processed_img = img.copy()
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_img, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the processed image
            cv2.imwrite('processed.jpg', processed_img)

            # Encode processed image
            _, buffer = cv2.imencode('.jpg', processed_img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send processed image back to client
            await websocket.send_json({
                "processed_image": processed_image_base64,
                "face_count": len(faces)
            })

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
