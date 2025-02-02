import asyncio
from bson.objectid import ObjectId
from mongoDB_connection import (
    create_user, create_conversation_context, update_user_context,
    get_user_context, get_conversation_context,
    close_db_connection
)

def print_separator():
    print("\n" + "-" * 50 + "\n")

async def test_crud_operations():
    print("Starting CRUD tests...")
    print_separator()

    # Create a new user
    user_data = {"name": "Alice", "email": "alice@example.com", "context": {}}
    user_id = await create_user(user_data)
    print(f"User Created: {user_id}")
    print_separator()

    # Retrieve user context
    user = await get_user_context(user_id)
    print(f"User Retrieved: {user}")
    print_separator()

    # Update user context
    updated_count = await update_user_context(user_id, {"mood": "happy"})
    print(f"User Context Updated: {updated_count} document(s) modified")
    print_separator()

    # Retrieve updated user context
    updated_user = await get_user_context(user_id)
    print(f"Updated User: {updated_user}")
    print_separator()

    # Create a new conversation
    conversation_data = {"user_id": user_id, "messages": []}
    conversation_id = await create_conversation_context(conversation_data)
    print(f"Conversation Created: {conversation_id}")
    print_separator()

    # Retrieve conversation context
    conversation = await get_conversation_context(conversation_id)
    print(f"Conversation Retrieved: {conversation}")
    print_separator()

    # Retrieve updated conversation context
    updated_conversation = await get_conversation_context(conversation_id)
    print(f"Updated Conversation: {updated_conversation}")
    print_separator()

    # Close the database connection
    close_db_connection()
    print("Database connection closed.")
    print_separator()

asyncio.run(test_crud_operations())