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
        result = await db["conversation_context"].insert_one(conversation_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"An error occurred: {e}")

async def update_user_context(user_id: str, context: dict, db):
    """
    Update the context of a user.
    """
    try:
        result = await db["user"].update_one({"_id": ObjectId(user_id)}, {"$set": {"context": context}})
        return result.modified_count
    except Exception as e:
        print(f"An error occurred: {e}")

async def get_user_context(user_id: str, db):
    """
    Retrieve the context of a user.
    """
    try:
        result = await db["user"].find_one({"_id": ObjectId(user_id)})
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
