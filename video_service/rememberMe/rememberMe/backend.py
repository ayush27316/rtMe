# database.py
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # or your preferred YOLO model


#from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import faiss
import numpy as np

# Get environment variables
MONGO_URI = "mongodb+srv://MubeenMohammed:<db_password>@cluster0.eydtb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_PASSWORD = "Mohdaslan@123"
# Global variable to store the last stable face image
latest_stable_face_image_base64 = None
object_names=[]
person_names=[]


# Encode password safely
encoded_password = quote_plus(MONGO_PASSWORD)
from pydantic import BaseModel

class FaceEntry(BaseModel):
    name: str
    face_encoding: str  # Assuming you store face encoding as a base64 string or similar
    created_at: str
# Replace the placeholder in MONGO_URI with the encoded password
MONGO_URI = MONGO_URI.replace("<db_password>", encoded_password)

# Initialize MongoDB connection

client = AsyncIOMotorClient(MONGO_URI)
db = client["Context"]

# main.py (your FastAPI application)
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from collections import deque
import base64
import os
from datetime import datetime
from database import get_database
from concurrent.futures import ThreadPoolExecutor
app = FastAPI()
face_buffer = deque(maxlen=30)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import aiohttp
import asyncio
from bson import ObjectId
from datetime import datetime


last_checked_name = None
is_background_task_running = False


async def check_face_count():
    """Check if the face count is exactly 1 from the WebSocket endpoint."""
    url = "https://158b-132-205-229-214.ngrok-free.app/ws-stable"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response_json = await response.json()
                face_count = response_json.get("face_count", 0)
                return face_count == 1
        except Exception as e:
            print(f"Error checking face count: {str(e)}")
            return False
        
async def get_latest_name(db) -> str:
    """Fetch the most recent name from face_name collection"""
    try:
        # Sort by _id in descending order and get the first document
        result = await db["face_name"].find_one(
            sort=[("_id", -1)]  # Sort by _id in descending order
        )
        
        if result and 'name' in result:
            return result['name']
        return None
    except Exception as e:
        print(f"Error fetching latest name: {str(e)}")
        return None
async def count_name_occurrences(name: str, db) -> int:
    """Count how many times a name appears in the faces collection"""
    try:
        count = await db["faces"].count_documents({"name": name})
        return count
    except Exception as e:
        print(f"Error counting name occurrences: {str(e)}")
        return 0
    
async def background_face_checker():
    global last_checked_name, is_background_task_running

    is_background_task_running = True

    while is_background_task_running:
        try:
            # Fetch the latest name
            current_name = await get_latest_name(db)
            occurrence_count = await count_name_occurrences(current_name, db)
            print(f"Name '{current_name}' occurs {occurrence_count} times in database")
            
            if occurrence_count < 5:
                if latest_stable_face_image_base64:
                    # Check if face_count == 1
                    face_is_valid = await check_face_count()

                    if face_is_valid:
                        try:
                            # Make HTTP POST request to create_user endpoint
                            async with aiohttp.ClientSession() as session:
                                url = f"https://158b-132-205-229-214.ngrok-free.app/create_user/{current_name}"
                                async with session.post(url) as response:
                                    result = await response.json()
                                    if "success" in result:
                                        print(f"Successfully registered new face for {current_name}")
                                    else:
                                        print(f"Failed to register face: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            print(f"Error calling create_user endpoint: {str(e)}")
                    else:
                        print("Face count is not 1, skipping registration")
                else:
                    print("No stable face image available")
                
                last_checked_name = current_name
            
            # Wait for 3 seconds before next check
            await asyncio.sleep(3)
        
        except Exception as e:
            print(f"Error in background face checker: {str(e)}")
            await asyncio.sleep(3)




async def background_item_adder():
    """Background task to add the latest items from global arrays to database every 3 seconds"""
    global is_background_task_running, object_names, person_names
    
    while is_background_task_running:
        try:
            # Only proceed if there are items to add
            if object_names or person_names:
                timestamp = datetime.now()
                
                # Get the latest data from arrays
                latest_objects = object_names[-1] if len(object_names[-1])>0 else None  # Copy current objects
                latest_person = person_names[-1] if person_names else None  # Get last person name
                if not (latest_person == 'Face' and not len(latest_objects)==0):

                
                    # Prepare item data
                    item_data = {
                        "timestamp": timestamp,
                        "person_name": latest_person,
                        "detected_objects": latest_objects
                    }
                    
                    # Add to database
                    await db["items"].insert_one(item_data)
                    
                    print(f"Added to items collection: Person: {latest_person}, Objects: {latest_objects}")
                    
                    # Clear the arrays after adding to database
                    object_names.clear()
                    person_names.clear()
                
            # Wait for 3 seconds
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error in background item adder: {str(e)}")
          
            await asyncio.sleep(3)


# Initialize database connection on startup
@app.on_event("startup")
async def startup_db_client():
    app.mongodb = client
    asyncio.create_task(background_face_checker())
    asyncio.create_task(background_item_adder())

@app.on_event("shutdown")
async def shutdown_db_client():
    global is_background_task_running
    is_background_task_running = False
    client.close()

# Your existing face detection setup
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

            # Keep only the largest detected face
            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Find the largest rectangle
                x, y, w, h = largest_face

                # Draw a rectangle around the largest face
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
                "face_count": 1 if len(faces) > 0 else 0
            })

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
    
@app.websocket("/ws-stable")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    face_buffer = []

    try:
        while True:
            data = await websocket.receive_json()
            image_base64 = data.get('image')

            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Create a single image for all detections
            processed_img = img.copy()
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

            # YOLO object detection
            results = model(processed_img)
            
            # Process YOLO results
            CONFIDENCE_THRESHOLD = 0.5

            # Get class names directly from model
            names = model.model.names
            obj=[]

            # Draw object detection results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                    
                        label = names[cls]
                        obj.append(label)
                        
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Draw object detection (in blue)
                        cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(processed_img, f'{label} {conf:.2f}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Face detection
            global object_names
            object_names.append(label)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            recognized_name = 'Face'

            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                face_buffer.append(largest_face)

                if len(face_buffer) > 30:
                    face_buffer.pop(0)

                if len(face_buffer) >= 30:
                    face_counts = {}
                    for face in face_buffer:
                        if face is not None:
                            x, y, w, h = face
                            key = (x//20, y//20, w//20, h//20)
                            face_counts[key] = face_counts.get(key, 0) + 1

                    for key, count in face_counts.items():
                        if count >= 1:
                            x, y, w, h = largest_face
                            recognition_result = await recognize_face_endpoint()
                            if "recognized_face" in recognition_result:
                                recognized_name = recognition_result["recognized_face"]
                            break

                # Draw face detection (in green)
                x, y, w, h = largest_face
                cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                global person_names
                person_names.append(recognized_name)
                cv2.putText(processed_img, recognized_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode final processed image
            _, buffer = cv2.imencode('.jpg', processed_img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            global latest_stable_face_image_base64
            latest_stable_face_image_base64 = processed_image_base64

            # Simplified response JSON
            response_json = {
                "processed_image": processed_image_base64,
                "face_count": len(faces)
            }

            await websocket.send_json(response_json)
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        await websocket.close()





def crop_face_from_rect_overlay(image):
    # Ensure the input image is valid
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image provided. Ensure the input is a valid OpenCV image.")

    # Convert to grayscale to detect the rectangle
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours (assuming the rectangle is the only prominent shape)
    edged = cv2.Canny(gray, 50, 150)  # Edge detection
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangle (assumed to be the face box)
    max_area = 0
    best_rect = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    # If no rectangle is found, return None
    if best_rect is None:
        return None

    # Crop the detected rectangle
    x, y, w, h = best_rect
    face_crop = image[y:y+h, x:x+w]

    return face_crop  # Return the raw cropped face image






from bson import Binary
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity




from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Assuming facenet_model is initialized globally
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
from torchvision import transforms
import torch
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Make sure FaceNet model is on the correct device
facenet_model = facenet_model.to(device)

async def get_all_face_encodings(db) -> list[dict]:
    try:
        face_entries = await db["faces"].find().to_list(length=None)
        print(f"Found {len(face_entries)} entries in database")
        
        processed_entries = []
        for entry in face_entries:
            # Convert the list back to numpy array and ensure correct shape
            face_encoding = np.array(entry["face_encoding"], dtype=np.float32)
            # Print shape for debugging
            print(f"Shape of encoding for {entry['name']}: {face_encoding.shape}")
            processed_entries.append({
                "name": entry["name"],
                "face_encoding": face_encoding,
            })
        
        print(f"Processed {len(processed_entries)} entries")
        return processed_entries
    except Exception as e:
        print(f"Error retrieving face encodings: {str(e)}")
        traceback.print_exc()
        return []

async def recognize_face(current_face_encoding: np.ndarray, db):
    try:
        # Get all face encodings from the database
        face_entries = await get_all_face_encodings(db)
        
        if not face_entries:
            return "No faces in database"
        
        best_match = None
        max_similarity = -1
        
        # Print shape for debugging
        print(f"Current encoding shape: {current_face_encoding.shape}")
        
        # Ensure current_face_encoding is 2D
        if len(current_face_encoding.shape) == 1:
            current_encoding = current_face_encoding.reshape(1, -1)
        else:
            current_encoding = current_face_encoding
            
        print(f"Reshaped current encoding shape: {current_encoding.shape}")
        
        # Compare with each face encoding
        for entry in face_entries:
            db_encoding = entry["face_encoding"]
            
            # Ensure database encoding is 2D
            if len(db_encoding.shape) == 1:
                db_encoding = db_encoding.reshape(1, -1)
                
            print(f"DB encoding shape for {entry['name']}: {db_encoding.shape}")
            
            try:
                # Calculate cosine similarity
                similarity = cosine_similarity(current_encoding, db_encoding)[0][0]
                print(f"Similarity with {entry['name']}: {similarity}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = entry["name"]
            except Exception as e:
                print(f"Error calculating similarity for {entry['name']}: {str(e)}")
                continue
        
        # If the best match similarity is too low, return unknown
        threshold = 0.8  # Adjust this threshold as needed
        if max_similarity < threshold:
            print(f"Best match similarity ({max_similarity}) is below threshold ({threshold})")
            return "Unknown"
        
        print(f"Found match: {best_match} with similarity {max_similarity}")
        return best_match

    except Exception as e:
        print(f"Error in recognize_face: {str(e)}")
    
        return "Error during recognition"


@app.post("/recognize_face")
async def recognize_face_endpoint():
    global latest_stable_face_image_base64
    
    if not latest_stable_face_image_base64:
        return {"error": "No stable face image found. Please try again later."}
    
    try:
        # Decode the base64 image
        image_data = base64.b64decode(latest_stable_face_image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image"}

        # Crop the face using the rectangle overlay
        cropped_face = crop_face_from_rect_overlay(img)
        if cropped_face is None:
            return {"error": "Failed to crop face from image"}

        # Convert BGR to RGB and to PIL Image
        face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Generate the face encoding
        current_face_encoding = generate_facenet_encoding(pil_image)
        
        # Get the best match from the database
        recognized_name = await recognize_face(current_face_encoding, db)
        
        return {"recognized_face": recognized_name}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}




# Function to create a new face entry in MongoDB
async def create_face_entry(name: str, face_encoding: np.ndarray, image_path: str, db):
    try:
        # Convert numpy array to Binary for MongoDB storage
        #encoding_binary = Binary(pickle.dumps(face_encoding))
        face_encoding_list = face_encoding.tolist()
        face_data = {
            "name": name,
            "face_encoding": face_encoding_list,
            "image_path": image_path,  # You can save image path or the image itself here
            "created_at": datetime.now()
        }
        print(face_data)
        
        result = await db["faces"].insert_one(face_data)
        print(result)
        return 1
    except Exception as e:
        print(f"Error creating face entry: {e}")
        return None

@app.post("/create_user/{name}")
async def create_user(name: str):
    """
    This endpoint will receive a stable face image and save the corresponding face encoding into the database.
    """
    global latest_stable_face_image_base64

    # Check if the stable face image exists
    if not latest_stable_face_image_base64:
        return {"error": "No stable face image found. Please try again later."}

    try:
        # Decode the base64 image
        image_data = base64.b64decode(latest_stable_face_image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image"}

        # Crop the face using the rectangle overlay
        cropped_face = crop_face_from_rect_overlay(img)
        if cropped_face is None:
            return {"error": "Failed to crop face from image"}

        # Convert BGR to RGB and to PIL Image
        face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Generate the face encoding
        face_encoding = generate_facenet_encoding(pil_image)

        # Save to the database
        result = await create_face_entry(name, face_encoding, "", db)
        print(result)
        
        if result:
            return {"success": True, "message": "Face entry created successfully.", "id": result}
        else:
            return {"error": "Failed to create face entry."}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

def crop_face_from_rect_overlay(image):
    """Crop the face from the image using the rectangle overlay"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate the rectangle
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (should be the rectangle)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small margin (e.g., 10 pixels) to the crop
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        # Crop the face region
        face_crop = image[y:y+h, x:x+w]
        
        return face_crop

    except Exception as e:
        print(f"Error in crop_face_from_rect_overlay: {str(e)}")
        return None

def generate_facenet_encoding(pil_image):
    """Generate Face Encoding using FaceNet from a PIL Image"""
    try:
        # Resize the image to 160x160 (FaceNet's expected input size)
        pil_image = pil_image.resize((160, 160))
        
        # Convert PIL image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])
        face_tensor = transform(pil_image)
        
        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)
        
        # Move to the same device as the model
        face_tensor = face_tensor.to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = facenet_model(face_tensor)

        # Return the embedding as numpy array
        return embedding[0].cpu().numpy()

    except Exception as e:
        raise ValueError(f"Error generating face encoding: {str(e)}")


    




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

