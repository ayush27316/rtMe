# database.py
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus

# Get environment variables
MONGO_URI = "mongodb+srv://MubeenMohammed:<db_password>@cluster0.eydtb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_PASSWORD = "Mohdaslan@123"
# Global variable to store the last stable face image
latest_stable_face_image_base64 = None

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

async def get_database():
    client = AsyncIOMotorClient(MONGO_URI)
    return client["Context"]
db=get_database()


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

# Initialize database connection on startup
@app.on_event("startup")
async def startup_db_client():
    app.mongodb = await get_database()

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb.client.close()

# Your existing face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

            stable_face_detected = True
            saved_file = None

            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # Find the largest face
                face_buffer.append(largest_face)

                if len(face_buffer) >= 30:  # Check stability after 30 frames
                    face_counts = {}
                    for face in face_buffer:
                        if face is not None:
                            x, y, w, h = face
                            key = (x//20, y//20, w//20, h//20)  # Group faces by position and size
                            face_counts[key] = face_counts.get(key, 0) + 1
                    

                    # If a face appears consistently in the same area (e.g., 24 out of 30 frames)
                    for key, count in face_counts.items():
                        if count >= 1:
                            stable_face_detected = True
                            x, y, w, h = largest_face
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            saved_file = save_cropped_face(img, largest_face, timestamp)
                            global latest_stable_face_image_base64

                            # You can store or return the stable face image for further processing
                            latest_stable_face_image_base64 = base64.b64encode(buffer).decode('utf-8')
                            break
                #global latest_stable_face_image_base64

                # You can store or return the stable face image for further processing
                

                # Draw a rectangle around the largest detected face
                x, y, w, h = largest_face
                cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_img, 'Face', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode processed image to base64
            _, buffer = cv2.imencode('.jpg', processed_img)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            latest_stable_face_image_base64 = processed_image_base64

            # Prepare the response JSON
            response_json = {
                "processed_image": processed_image_base64,
                "face_count": 1 if len(faces) > 0 else 0,
                "stable_face_detected": stable_face_detected,
                "saved_file": saved_file
            }

            # Print the final JSON being returned
            print("Returning the following JSON:")
            #print(response_json)
            #print(latest_stable_face_image_base64)

            # Send the response to the client
            await websocket.send_json(response_json)
    except Exception as e:
            print(f"Error: {e}")
    finally:
        await websocket.close()


# Modified stable face endpoint to store the latest stable face
@app.websocket("/send_stable_face")
async def stable_face_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Stable face detection connection established")
    
    face_buffer = deque(maxlen=30)
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                image_base64 = data.get('image')
                
                if not image_base64:
                    continue

                # Decode image
                image_data = base64.b64decode(image_base64)
                np_arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                processed_img = img.copy()
                gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
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
                    
                    if len(face_buffer) >= 30:
                        face_counts = {}
                        for face in face_buffer:
                            if face is not None:
                                x, y, w, h = face
                                key = (x//20, y//20, w//20, h//20)
                                face_counts[key] = face_counts.get(key, 0) + 1
                        
                        for key, count in face_counts.items():
                            if count >= 24:
                                stable_face_detected = True
                                x, y, w, h = largest_face
                                
                                # Save cropped face
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                saved_file = save_cropped_face(img, largest_face, timestamp)
                                
                                # Store the stable face in shared state
                                shared_state.latest_stable_face = largest_face
                                shared_state.latest_stable_face_image = img
                                
                                face_buffer.clear()
                                break

                # Encode processed image to base64
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send processed image and other details back to client
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
        await websocket.close()



def generate_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #face_encodings = face_recognition.face_encodings(rgb_image)
    return 123



# Function to convert base64 image to a numpy array and then to face encoding
def convert_image_to_encoding(base64_image: str) -> np.ndarray:
    image_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Assuming you are using some facial recognition library like face_recognition to get face encoding
    # You can replace this with the actual face encoding extraction process you use
    #import face_recognition
    #face_encodings = face_recognition.face_encodings(img)

    #if len(face_encodings) == 0:
       # raise ValueError("No faces found in the image.")

    # Return the first face encoding
    return 12

# Function to create a new face entry in MongoDB
async def create_face_entry(name: str, face_encoding: np.ndarray, image_path: str, db):
    try:
        # Convert numpy array to Binary for MongoDB storage
        #encoding_binary = Binary(pickle.dumps(face_encoding))
        
        face_data = {
            "name": name,
            "face_encoding": 12,
            "image_path": image_path,  # You can save image path or the image itself here
            "created_at": datetime.now()
        }
        print(face_data)
        
        result = await db["faces"].insert_one(face_data)
        return 1
    except Exception as e:
        print(f"Error creating face entry: {e}")
        return None


# Modified create_user endpoint that uses the stable face
@app.post("/create_user/{name}")
async def create_user(name: str):
    """
    This endpoint will receive a stable face image and save the corresponding face encoding into the database.
    """
    global latest_stable_face_image_base64

    # Check if the stable face image exists
    if not latest_stable_face_image_base64:
        return {"error": "No stable face image found. Please try again later."}

    # Decode the base64 image
    image_data = base64.b64decode(latest_stable_face_image_base64)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert image to encoding
    try:
        face_encoding = convert_image_to_encoding(img)  # Assuming face_recognition is used
    except ValueError as e:
        return {"error": f"Error encoding face: {str(e)}"}

    # Save to the database
    result = await create_face_entry(name, face_encoding, latest_stable_face_image_base64, db)
    
    if result:
        return {"success": True, "message": "Face entry created successfully.", "id": result}
    else:
        return {"error": "Failed to create face entry."}





# Add these helper functions if not already present:
async def create_user(user_data: dict, db):
    """Create a new user in the database"""
    try:
        result = await db["users"].insert_one(user_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating user: {e}")
        return None
    




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''
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



'''