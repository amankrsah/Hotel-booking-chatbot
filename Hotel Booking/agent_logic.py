import os
import json
import torch
import whisper
import re
import numpy as np
from gtts import gTTS
from datetime import datetime
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import textwrap

# --- NEW: Database Imports ---
import chromadb
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# --- FORCE OFFLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ==========================================
# 1. RELATIONAL DATABASE SETUP (SQLAlchemy)
# ==========================================
print("⚡ Initializing SQLite Relational Database...")
# connect_args ensures SQLite handles multiple threads from FastAPI safely
engine = create_engine('sqlite:///bookings.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the strict SQL Schema for Bookings
class BookingRecord(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(String, index=True)
    guest_name = Column(String)
    email = Column(String)
    phone = Column(String)
    adults = Column(Integer)
    children = Column(Integer)
    hotel_name = Column(String, index=True)
    room_type = Column(String, index=True)
    check_in = Column(String)
    check_out = Column(String)
    status = Column(String, default="Booked")
    timestamp = Column(String)

# Create the database file and tables if they don't exist
Base.metadata.create_all(bind=engine)
print("✅ Relational Database Ready!")

# ==========================================
# 2. AI & GPU MODEL SETUP
# ==========================================
model_path = "qwen2.5-3b-instruct-q4_k_m.gguf"
if not os.path.exists(model_path):
    print(f"🚨 CRITICAL: Cannot find {model_path}.")

print("⚡ Loading GGUF Model into RTX 5050 VRAM...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  
    n_ctx=4096,       
    n_batch=4096,      
    flash_attn=True,   
    verbose=False     
)
print("✅ LLM Loaded Instantly!")

asr_model = whisper.load_model("base", device="cuda")

print("⚡ Loading RAG Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

# ==========================================
# 3. VECTOR DATABASE SETUP (ChromaDB)
# ==========================================
def load_data(file_path):
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

resorts_db = load_data('resorts.json')

print("⚡ Initializing ChromaDB Persistent Vector Store...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="resorts")

# Auto-populate ChromaDB only if it's empty (Saves huge boot time!)
if collection.count() == 0 and resorts_db:
    print("⚡ First boot detected: Indexing vectors into ChromaDB...")
    docs, metadatas, ids, embs = [], [], [], []
    
    for i, r in enumerate(resorts_db):
        text = f"Hotel: {r.get('hotel_name')}. Location: {r.get('location')}. Description: {r.get('description')}. Rooms: {json.dumps(r.get('room_types'))}"
        docs.append(text)
        metadatas.append({"hotel_name": r.get('hotel_name')})
        
        # Use existing ID or fallback
        resort_id = r.get('id', f"resort_{i}")
        ids.append(resort_id)
        
        # Process vector on GPU
        embs.append(embedder.encode(text).tolist())

    collection.add(embeddings=embs, documents=docs, metadatas=metadatas, ids=ids)
    print(f"✅ Indexed {len(resorts_db)} resorts into persistent storage.")
else:
    print(f"✅ ChromaDB loaded from disk. Contains {collection.count()} indexed resorts.")


def retrieve_relevant_resorts(query: str, k=10):
    """Fetches Top-K resorts from persistent ChromaDB based on semantic similarity."""
    if collection.count() == 0 or not resorts_db:
        return []
    
    discovery_keywords = ["show", "list", "hotels", "resorts", "options", "available"]
    if any(kw == query.lower().strip() for kw in discovery_keywords):
        return resorts_db[:k]
        
    query_emb = embedder.encode([query]).tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(k, collection.count())
    )
    
    retrieved = []
    if results['ids'] and len(results['ids']) > 0:
        for res_id in results['ids'][0]:
            # Map the vector ID back to the full JSON object for the UI
            for r in resorts_db:
                if r.get('id', '') == res_id or r.get('hotel_name') == res_id:
                    retrieved.append(r)
                    break
    return retrieved

def extract_dates(text):
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    dates = re.findall(date_pattern, text)
    if len(dates) >= 2:
        return dates[0], dates[1]
    return None, None

# ==========================================
# 4. INTENT CLASSIFICATION & LLM LOGIC
# ==========================================
def get_agent_response(user_input: str):
    user_input_lower = user_input.lower()
    discovery_keywords = ["show", "list", "hotels", "resorts", "options", "available", "stay", "accommodation", "booking"]
    is_asking_for_resorts = any(kw in user_input_lower for kw in discovery_keywords)
    
    relevant_resorts = retrieve_relevant_resorts(user_input, k=10) if is_asking_for_resorts else []
    
    check_in, check_out = extract_dates(user_input)
    internal_avail_msg = ""
    if check_in and check_out:
        is_free, msg = check_room_availability(None, None, check_in, check_out)
        if not is_free:
            internal_avail_msg = f"System Note: The requested dates {check_in} to {check_out} conflict with existing bookings."

    context_str = json.dumps(relevant_resorts, separators=(',', ':')) if relevant_resorts else "[]"

    system_message = textwrap.dedent(f"""\
    You are the Barnawapara Wildlife Sanctuary Booking Agent.
    CRITICAL RULE: Respond ONLY in valid JSON. No markdown, no plain text.
    INTENTS: DISCOVERY (show options), BOOKING (room selected).
    KNOWLEDGE: {context_str}
    REQUIRED JSON: {{"intent":"DISCOVERY","reply":"...","data":[]}}
    {internal_avail_msg}""").strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]

    output = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.3)
    raw_content = output['choices'][0]['message']['content']

    try:
        clean_json = raw_content.replace('```json', '').replace('```', '').strip()
        structured_data = json.loads(clean_json)
        
        if is_asking_for_resorts and structured_data.get('intent') != 'BOOKING':
            structured_data['intent'] = 'DISCOVERY'
        if "reply" not in structured_data:
            structured_data["reply"] = "I found these options for you."
            
        if structured_data.get('intent') == 'DISCOVERY':
            structured_data['data'] = relevant_resorts if relevant_resorts else load_data('resorts.json')
            
        return structured_data
    except Exception as e:
        fallback_data = relevant_resorts if relevant_resorts else load_data('resorts.json')
        return {
            "intent": "DISCOVERY" if is_asking_for_resorts else "GENERAL",
            "reply": "I found several resorts for you." if is_asking_for_resorts else raw_content,
            "data": fallback_data if is_asking_for_resorts else []
        }

# ==========================================
# 5. RELATIONAL BOOKING & DATE LOGIC (SQL)
# ==========================================
def get_booked_dates_for_room(hotel_name, room_type):
    """Queries SQL Database for disabled dates."""
    db = SessionLocal()
    try:
        bookings = db.query(BookingRecord).filter(
            BookingRecord.hotel_name == hotel_name,
            BookingRecord.room_type == room_type,
            BookingRecord.status == 'Booked'
        ).all()
        
        booked_dates = [{"check_in": b.check_in, "check_out": b.check_out} for b in bookings]
        return booked_dates
    finally:
        db.close()

def is_overlapping(req_in, req_out, existing_in, existing_out):
    fmt = "%Y-%m-%d"
    try:
        req_in_dt = datetime.strptime(req_in, fmt)
        req_out_dt = datetime.strptime(req_out, fmt)
        exist_in_dt = datetime.strptime(existing_in, fmt)
        exist_out_dt = datetime.strptime(existing_out, fmt)
        return max(req_in_dt, exist_in_dt) < min(req_out_dt, exist_out_dt)
    except:
        return False

def check_room_availability(hotel_name, room_type, check_in, check_out):
    """Verifies concurrency directly against SQL Database."""
    db = SessionLocal()
    try:
        bookings = db.query(BookingRecord).filter(BookingRecord.status == 'Booked').all()
        for b in bookings:
            if is_overlapping(check_in, check_out, b.check_in, b.check_out):
                return False, "Dates are booked."
        return True, "Available"
    finally:
        db.close()

def save_confirmed_booking(booking_id, guest_name, check_in=None, check_out=None, hotel_name=None, room_type=None, email=None, phone=None, adults=1, children=0):
    """Commits final verified transaction to SQL Database."""
    db = SessionLocal()
    try:
        new_booking = BookingRecord(
            booking_id=str(booking_id),
            guest_name=guest_name,
            email=email,
            phone=phone,
            adults=adults,
            children=children,
            hotel_name=hotel_name,
            room_type=room_type,
            check_in=check_in,
            check_out=check_out,
            status="Booked",
            timestamp=datetime.now().isoformat()
        )
        db.add(new_booking)
        db.commit()
        db.refresh(new_booking)
        return True, f"Reservation confirmed for {guest_name}!"
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        return False, "Failed to save booking."
    finally:
        db.close()

# ==========================================
# 6. VOICE PROCESSING
# ==========================================
def speech_to_text(audio_path: str):
    return asr_model.transcribe(audio_path)["text"]

def text_to_speech(text_payload, output_path: str = "response.mp3"):
    speech_text = text_payload.get('reply', "Processing complete.") if isinstance(text_payload, dict) else str(text_payload)
    clean_speech = speech_text.replace('#', '').replace('*', '').replace('---', '')
    tts = gTTS(text=clean_speech, lang='en')
    tts.save(output_path)
    return output_path