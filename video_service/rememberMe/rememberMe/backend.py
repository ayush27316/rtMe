import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import os
from datetime import datetime

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

# Create a directory for stable faces if it doesn't exist
STABLE_FACES_DIR = "stable_faces"
if not os.path.exists(STABLE_FACES_DIR):
    os.makedirs(STABLE_FACES_DIR)

def save_stable_face(image):
    """Save stable face with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stable_face_{timestamp}.jpg"
    filepath = os.path.join(STABLE_FACES_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath

# Original websocket endpoint remains unchanged...

@app.websocket("/send_stable_face")
async def stable_face_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Stable face detection connection established")
    
    # Buffer for face tracking
    face_buffer = deque(maxlen=30)  # 1 second at 30 FPS
    
    try:
        while True:
            try:
                # Receive base64 image from client
                data = await websocket.receive_json()
                image_base64 = data.get('image')
                
                if not image_base64:
                    print("No image data received")
                    continue

                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                np_arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    print("Failed to decode image")
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

                stable_face_detected = False
                saved_file = None
                
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_buffer.append(largest_face)
                    
                    # Check for stable face
                    if len(face_buffer) >= 30:
                        # Count similar face positions
                        face_counts = {}
                        for face in face_buffer:
                            if face is not None:
                                x, y, w, h = face
                                # Reduce precision for stability
                                key = (x//20, y//20, w//20, h//20)
                                face_counts[key] = face_counts.get(key, 0) + 1
                        
                        # Check if any face position is stable
                        for key, count in face_counts.items():
                            if count >= 24:  # 80% of frames
                                stable_face_detected = True
                                x, y, w, h = largest_face
                                
                                # Draw green rectangle for stable face
                                cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(processed_img, 'Stable Face', (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Save stable face
                                saved_file = save_stable_face(processed_img)
                                print(f"Stable face saved: {saved_file}")
                                
                                # Clear buffer after saving
                                face_buffer.clear()
                                break
                        
                    if not stable_face_detected:
                        # Draw blue rectangle for unstable face
                        x, y, w, h = largest_face
                        cv2.rectangle(processed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(processed_img, 'Face', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    face_buffer.append(None)

                # Encode processed image
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send processed image back to client
                await websocket.send_json({
                    "processed_image": processed_image_base64,
                    "face_count": 1 if len(faces) > 0 else 0,
                    "stable_face_detected": stable_face_detected,
                    "saved_file": saved_file
                })

            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue

    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        print("Stable face detection connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
