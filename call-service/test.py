import asyncio
import os
from bson.objectid import ObjectId
from mongoDB_connection import (
    update_user
)
from urllib.parse import quote_plus
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
encoded_password = quote_plus(os.getenv("MONGO_PASSWORD"))
# Replace the placeholder with the encoded password.
client = AsyncIOMotorClient(MONGO_URI.replace("<db_password>", encoded_password))

# Use your database name (adjust as needed)
db = client["Context"]

def print_separator():
    print("\n" + "-" * 50 + "\n")

async def test_crud_operations():
    print("Starting CRUD tests...")
    print_separator()

    # Update user context
    updated_count = await update_user("679f0e88fe70e4ec01ca2cf2", {"mood": "happy"}, db)
    print(f"User Context Updated: {updated_count} document(s) modified")
    print_separator()

    # Close the database connection
    client.close()
    print("Database connection closed.")
    print_separator()

asyncio.run(test_crud_operations())