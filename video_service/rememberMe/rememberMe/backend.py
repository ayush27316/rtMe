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

def save_cropped_face(image, face_coords, timestamp=None):
    """Crop and save face region"""
    x, y, w, h = face_coords
    # Add some padding around the face
    padding = 30
    y1 = max(y - padding, 0)
    y2 = min(y + h + padding, image.shape[0])
    x1 = max(x - padding, 0)
    x2 = min(x + w + padding, image.shape[1])
    
    face_img = image[y1:y2, x1:x2]
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_{timestamp}.jpg"
    filepath = os.path.join(STABLE_FACES_DIR, filename)
    cv2.imwrite(filepath, face_img)
    return filepath #return the path of the saved image (can be removed)

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
                                
                                # Save cropped face
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                saved_file = save_cropped_face(img, largest_face, timestamp)
                                print(f"Cropped face saved: {saved_file}")
                                
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

import face_recognition
import numpy as np
from PIL import Image

def compare_faces(image1_path, image2_path):
    """
    Compare two face images using face_recognition library
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
    Returns:
        float: Distance between faces (lower means more similar)
    """
    # Load images
    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)
    
    # Get face encodings
    face_encodings1 = face_recognition.face_encodings(image1)
    face_encodings2 = face_recognition.face_encodings(image2)
    
    if not face_encodings1 or not face_encodings2:
        raise ValueError("No faces found in one or both images")
    
    # Calculate distance between faces
    distance = face_recognition.face_distance([face_encodings1[0]], face_encodings2[0])[0]
    
    # Convert distance to similarity score (0-1)
    similarity = 1 - distance
    
    if similarity > 0.6:
        return True
    else:
        return False



