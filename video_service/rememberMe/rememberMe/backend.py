import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure PyTorch for faster inference
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("CUDA is available. Using GPU.")
else:
    logger.info("CUDA is not available. Using CPU.")

# Load YOLO model
try:
    model = YOLO('yolov8n-face.pt')  # or your specific model path
    model.conf = 0.5  # confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    logger.info("YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    raise

@app.get("/")
async def root():
    return {"message": "Face Detection Server is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive base64 image from client
            try:
                data = await websocket.receive_json()
                image_base64 = data.get('image')
                if not image_base64:
                    continue
                
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                np_arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    logger.warning("Received invalid image")
                    continue

                # Resize image for faster processing
                scale_factor = 0.5
                height, width = img.shape[:2]
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                small_frame = cv2.resize(img, (new_width, new_height))
                
                # Run YOLOv8 inference
                results = model(small_frame, conf=0.7, iou=0.45)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = float(box.conf[0])
                        
                        # Only process high-confidence detections
                        if conf < 0.5:
                            continue
                            
                        # Scale back the coordinates
                        x1, y1, x2, y2 = [int(coord/scale_factor) for coord in [x1, y1, x2, y2]]
                        
                        # Draw rectangle with thickness based on confidence
                        thickness = max(1, int(conf * 3))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                        
                        # Add confidence score with better formatting
                        label = f'{conf:.2%}'
                        (label_width, label_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1-20), (x1+label_width, y1),
                                    (0, 255, 0), -1)
                        cv2.putText(img, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Encode with lower quality for faster transmission
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, buffer = cv2.imencode('.jpg', img, encode_param)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send processed image back to client
                await websocket.send_json({
                    "processed_image": processed_image_base64,
                    "face_count": len(results[0].boxes)
                })

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
