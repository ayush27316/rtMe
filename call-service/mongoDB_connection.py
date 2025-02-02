from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId  # Import ObjectId

load_dotenv()



# --------------------------
# CRUD Helper Functions
# --------------------------


async def create_user(user_data: dict, db):
    """
    Insert a new document into the 'user' collection.
    """
    try:
        result = await db["users"].insert_one(user_data)
        return str(result.inserted_id)  # Convert ObjectId to string
    except Exception as e:
        print(f"An error occurred: {e}")

async def create_conversation_context(conversation_data: dict, db):
    """
    Insert a new document into the 'conversation' collection.
    """
    try:
        result = await db["conversation_context"].insert_one({
            "name": name,
            "context": conversation_data
        })
        return str(result.inserted_id)
    except Exception as e:
        print(f"An error occurred: {e}")

async def update_user(user_id: str, data: dict, db):
    """
    Update the context of a user.
    """
    try:
        result = await db["user"].update_one({"_id": ObjectId(user_id)}, {"$set": data})
        return result.modified_counts
    except Exception as e:
        print(f"An error occurred: {e}")

async def get_user(name: str, db):
    """
    Retrieve the context of a user.
    """
    try:
        result = await db["user"].find_one({"name": name})
        return result
    except Exception as e:
        print(f"An error occurred: {e}")

async def get_conversation_context(conversation_id: str, db):
    """
    Retrieve the context of a conversation.
    """
    try:
        result = await db["conversation_context"].find_one({"_id": ObjectId(conversation_id)})
        return result
    except Exception as e:
        print(f"An error occurred: {e}")

async def update_conversation_context(conversation_id: str, context: dict, db):
    """
    Update the context of a conversation.
    """
    try:
        result = await db["conversation_context"].update_one({"_id": ObjectId(conversation_id)}, {"$set": {"context": context}})
        return result.modified_count
    except Exception as e:
        print(f"An error occurred: {e}")

def close_db_connection(client):
    """
    Close the connection to the database.
    """
    client.close()


from typing import Optional, List
import numpy as np
from bson.binary import Binary
import pickle

# Add these CRUD functions for face operations

async def create_face_entry(name: str, face_encoding: np.ndarray, image_path: str, db):
    """
    Insert a new face encoding into the 'faces' collection.
    """
    try:
        # Convert numpy array to Binary for MongoDB storage
        encoding_binary = Binary(pickle.dumps(face_encoding))
        
        face_data = {
            "name": name,
            "face_encoding": encoding_binary,
            "created_at": datetime.now()
        }
        
        result = await db["faces"].insert_one(face_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error creating face entry: {e}")
        return None

async def get_all_face_encodings(db) -> List[dict]:
    """
    Retrieve all face encodings from the database.
    Returns list of dictionaries containing name and encoding.
    """
    try:
        face_entries = await db["faces"].find().to_list(length=None)
        
        # Convert Binary back to numpy arrays
        processed_entries = []
        for entry in face_entries:
            processed_entries.append({
                "name": entry["name"],
                "face_encoding": pickle.loads(entry["face_encoding"]),
            })
        
        return processed_entries
    except Exception as e:
        print(f"Error retrieving face encodings: {e}")
        return []



async def delete_face(name: str, db):
    """
    Delete a face entry from the database.
    """
    try:
        result = await db["faces"].delete_one({"name": name})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error deleting face: {e}")
        return False

async def update_face_encoding(name: str, new_face_encoding: np.ndarray, new_image_path: str, db):
    """
    Update the face encoding for an existing name.
    """
    try:
        encoding_binary = Binary(pickle.dumps(new_face_encoding))
        
        result = await db["faces"].update_one(
            {"name": name},
            {
                "$set": {
                    "face_encoding": encoding_binary,
                    "updated_at": datetime.now()
                }
            }
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating face encoding: {e}")
        return False