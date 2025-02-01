from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import openai
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI.replace("<db_password>", os.getenv('MONGO_PASSWORD')))
db = client["userContextDB"]
collection = db["userContext"]


# Database and Collection
db = client["context"]


# --------------------------
# CRUD Helper Functions
# --------------------------


async def create_user(user_data: dict):
    """
    Insert a new document into the 'user' collection.
    """
    try:
        result = await db["user"].insert_one(user_data)
        return result.inserted_id
    except Exception as e:
        print(f"An error occurred: {e}")


async def create_conversation_context(conversation_data: dict):
    """
    Insert a new document into the 'conversation' collection.
    """
    try:
        result = await db["conversation_context"].insert_one(conversation_data)
        return result.inserted_id
    except Exception as e:
        print(f"An error occurred: {e}")

async def update_user_context(user_id: str, context: dict):
    """
    Update the context of a user.
    """
    try:
        result = await db["user"].update_one({"_id": ObjectId(user_id)}, {"$set": {"context": context}})
        return result.modified_count
    except Exception as e:
        print(f"An error occurred: {e}")

async def get_user_context(user_id: str):
    """
    Retrieve the context of a user.
    """
    try:
        result = await db["user"].find_one({"_id": ObjectId(user_id)})
        return result
    except Exception as e:
        print(f"An error occurred: {e}")

async def get_conversation_context(conversation_id: str):
    """
    Retrieve the context of a conversation.
    """
    try:
        result = await db["conversation_context"].find_one({"_id": ObjectId(conversation_id)})
        return result
    except Exception as e:
        print(f"An error occurred: {e}")

async def update_conversation_context(conversation_id: str, context: dict):
    """
    Update the context of a conversation.
    """
    try:
        result = await db["conversation_context"].update_one({"_id": ObjectId(conversation_id)}, {"$set": {"context": context}})
        return result.modified_count
    except Exception as e:
        print(f"An error occurred: {e}")

def close_db_connection():
    """
    Close the connection to the database.
    """
    client.close()