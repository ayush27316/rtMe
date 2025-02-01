import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ImageRequest(BaseModel):
    image: str  # Base64-encoded string

@app.post("/receive_image")
async def receive_image(request: ImageRequest):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(request.image)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Decoded image is None")

        # Create a copy for processing
        processed_img = img.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Optional: Add text label above the rectangle
            cv2.putText(processed_img, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save both original and processed images for debugging
        cv2.imwrite('received_frame.jpg', img)  # Save original image
        cv2.imwrite('processed_frame.jpg', processed_img)  # Save processed image

        # Encode the processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Save base64 string to a text file for inspection
        with open('processed_image_base64.txt', 'w') as f:
            f.write(processed_image_base64)

        return {
            "message": f"Image processed successfully. Found {len(faces)} faces.",
            "processed_image": processed_image_base64,
            "face_count": len(faces)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
