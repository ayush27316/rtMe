import os
import json
import base64
import asyncio
import websockets
import wave
import audioop
import time

from urllib.parse import quote_plus
import traceback
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from mongoDB_connection import (
    create_user,
    close_db_connection,
    get_user,
    create_conversation_context,  # Add this line
    get_conversation_context
) 

import assemblyai as aai
aai.settings.api_key = "6897ea66a618490eace0f7e38c25fc15"
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
encoded_password = quote_plus(os.getenv("MONGO_PASSWORD"))
# Replace the placeholder with the encoded password.
client = AsyncIOMotorClient(MONGO_URI.replace("<db_password>", encoded_password))

# Use your database name (adjust as needed)
db = client["Context"]

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))

def build_system_message(user_data):
    return = ("""
        You are a MemoryAid agent assisting patients with early-stage Alzheimer's and memory loss. Your goal is to help them retrieve, save, modify, and delete information in the database while maintaining clarity, accuracy, and a caring approach.  
        Any input that you receive can be assumed to be coming from the patient. But once you receive a response starting with [ADMIN], you should treat it as an instruction from the system.
        Rules for Handling Requests:  
        1. Saving Information (Create Operation)  
        Trigger: When the patient asks to save or store new information.  
        Response Format: You must respond with a JSON object in this exact format:  
        {"operation": "create", "data": {}}  
        How to Fill It:  
        - Place the provided information inside the data field exactly as given by the patient.  
        - Do not modify, rephrase, or exclude any details.  
        Example:  
        User Request: "Remember that my doctor's appointment is on Monday at 3 PM."  
        Correct Response: {"operation": "create", "data": {"appointment": "Doctor on Monday at 3 PM"}}  
        2. Retrieving Information (Search Operation)  
        Trigger: When the patient asks "What" or "Where" questions and you don't have necessary infromation to answer it then you must query the admin using following response .  
        Response Format: You must respond with a JSON object in this exact format:  
        {"operation": "search", "name": "name_of_the_person", "data": {}}  
        How to Fill It:  
        - Place the requested details inside the data field.  
        Example:  
        User Request: "What is my doctor's appointment?"  
        Correct Response: {"operation": "search","name": "name of the person about whom the patient is querying" "data": {"appointment": "Doctor on Monday at 3 PM"}}  

        3. Modifying Information (Update Operation)  
        Trigger: When the user asks to change, update, or modify existing information.  
        Response Format: You must respond with a JSON object in this exact format:  
        {'operation': "update", 'data': {}}  
        How to Fill It:  
        - Place both the existing information and the new details inside the data field.  
        Example:  
        User Request: "Change my doctor's appointment to Tuesday at 4 PM."  
        Correct Response: {'operation': "update", 'data': {'appointment': {'old': 'Doctor on Monday at 3 PM', 'new': 'Doctor on Tuesday at 4 PM'}}}  

        Additional Rules to Follow:  
        Be direct and precise and provide only the requested information without adding extra text.  
        Be caring and patient to ensure your tone remains friendly and supportive.  
        Strictly follow JSON formatting without adding extra characters, words, or explanations.  
        Do not make assumptions. If details are unclear, ask the user for clarification instead of assuming.  
        """
        f"Here is some context about the patient: {user_data}"
    )

def get_user_data():
    users = {}          #use the data_base to get the information
    return f"{users}"

SYSTEM_MESSAGE = build_system_message(get_user_data())

VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]

def ulaw_to_wav(input_file, output_file, channels=1, sampwidth=2, framerate=8000):
    """
    Convert a μ-law encoded audio file to WAV format.
    
    Parameters:
    input_file (str): Path to input .ulaw file
    output_file (str): Path to output .wav file
    channels (int): Number of audio channels (default: 1 for mono)
    sampwidth (int): Sample width in bytes (default: 2)
    framerate (int): Frame rate in Hz (default: 8000)
    """
    # Read the ulaw file
    with open(input_file, 'rb') as ulaw_file:
        ulaw_data = ulaw_file.read()
    
    # Convert ulaw to linear PCM
    linear_data = audioop.ulaw2lin(ulaw_data, sampwidth)
    
    # Create and write WAV file
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(linear_data)


def is_file_open(filepath):
    try:
        with open(filepath, "r+"):  # Try opening in read+write mode
            return False  # File is not open by another process
    except IOError:
        return True 


audio_file = open("call_audio.ulaw", "wb")

SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("Hey")
    response.pause(length=1)
    response.say("O.K. you can start talking!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def inject_data(data):
            initial_conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "context": data,
                            "text": "Answer the user question with the following context?"
                        }
                    ]
                }
            }
            
            await openai_ws.send(json.dumps(initial_conversation_item))
            await openai_ws.send(json.dumps({"type": "response.create"}))

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    if response.get('type') == 'response.done':
                        print("Response done.")
                        if response.get("response"):
                            output = response["response"].get("output", [])
                            if output:  # Check if output list is not empty
                                content = output[0].get("content", "")
                                if content:  # Check if content list is not empty
                                    # Extract JSON string from transcript between the curly brackets
                                    transcript = content[0].get("transcript", "")
                                    print(f"Transcript: {transcript}")
                                    start_index = transcript.find('{')
                                    end_index = transcript.rfind('}') + 1
                                    if start_index != -1 and end_index != -1:
                                        json_string = transcript[start_index:end_index]
                                        try:
                                            transcript_data = json.loads(json_string)
                                            # Process the transcript data
                                            if isinstance(transcript_data, dict):
                                                if "operation" in transcript_data and "data" in transcript_data:
                                                    if transcript_data["operation"] == "create":
                                                        if "name" in transcript_data["data"]:
                                                            user = await create_user(transcript_data["data"], db)
                                                            print(f"Successfully processed user information: {transcript_data['operation']}")
                                                        else:
                                                            await create_conversation_context(transcript_data["data"], db)
                                                    if transcript_data["operation"] == "search":
                                                        print(f"Searching for user information: {transcript_data['data']}")
                                                        user = await get_user(transcript_data["data"]["name"], db)
                                                        await inject_data({"user": user})
                                                        print(f"Successfully processed database operation: {transcript_data['operation']}")
                                                else:
                                                    print(f"Missing required fields in transcript data: {transcript_data}")
                                            else:
                                                print(f"Transcript data is not a dictionary: {transcript_data}")
                                                
                                        except json.JSONDecodeError as e:
                                            print(f"JSON parsing error: {e}")
                                            print(f"Transcript content: {json_string}")
                                        except Exception as e:
                                            print(f"Error processing transcript: {str(e)}")
                                            print(f"Transcript content: {json_string}")
                                    else:
                                        print("No JSON data found in the transcript.")
                                else:
                                    print("No content found in the output.")
                            else:
                                print("No output found in the response.")

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                    if response.get('type') == 'input_audio_buffer.speech_started':
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
                traceback.print_exc()


        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
    close_db_connection(client)
