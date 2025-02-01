
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64-encoded string

@app.post("/receive_image")
async def receive_image(request: ImageRequest):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(request.image)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Ensure image is not empty
        if img is None:
            raise ValueError("Decoded image is None")

        # Convert to grayscale (test processing step)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {"message": "Image processed successfully", "gray_shape": gray.shape}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
