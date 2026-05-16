import os
import shutil
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi import Request
from agent_logic import SessionLocal, BookingRecord
import secrets
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from agent_logic import (
    get_agent_response, 
    speech_to_text, 
    text_to_speech, 
    save_confirmed_booking,
    get_booked_dates_for_room,
    load_data
)

app = FastAPI(title="Barnawapara Integrated Agentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Agent-Reply"] 
)

if not os.path.exists("templates"):
    os.makedirs("templates")

app.mount("/static", StaticFiles(directory="templates", html=True), name="static")

class ChatRequest(BaseModel):
    user_input: str

class BookedDatesRequest(BaseModel):
    hotel_name: str
    room_type: str

class PaymentInitiationRequest(BaseModel):
    booking_id: int
    amount: int
    currency: str = "INR"

class PaymentConfirmationRequest(BaseModel):
    booking_id: str
    guest_name: str
    email: str
    phone: str
    adults: int
    children: int
    razorpay_payment_id: str
    razorpay_order_id: str
    check_in: str
    check_out: str
    hotel_name: str
    room_type: str

def cleanup_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            pass

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        return get_agent_response(request.user_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Agent processing failed.")
    
# --- ADMIN AUTHENTICATION SETUP ---
security = HTTPBasic()

def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    # Hardcode your admin credentials here
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "barnawapara123")
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- ADMIN DASHBOARD ROUTES (Now Protected) ---

@app.get("/admin")
async def admin_dashboard(request: Request, username: str = Depends(get_current_admin)):
    """Serves the Admin HTML Interface. Protected by Basic Auth."""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/bookings")
def get_all_bookings(username: str = Depends(get_current_admin)):
    """Fetches all bookings from the SQLite Database. Protected by Basic Auth."""
    # Note: Import SessionLocal and BookingRecord from agent_logic if not already done
    from agent_logic import SessionLocal, BookingRecord 
    
    db = SessionLocal()
    try:
        # Fetch all records, ordered by newest first
        bookings = db.query(BookingRecord).order_by(BookingRecord.id.desc()).all()
        return bookings
    finally:
        db.close()

@app.post("/voice-chat")
async def voice_chat_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    request_id = uuid.uuid4()
    temp_input = f"temp_{request_id}.wav"
    output_audio_path = f"res_{request_id}.mp3"
    
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        user_text = speech_to_text(temp_input)
        agent_payload = get_agent_response(user_text)
        text_to_speech(agent_payload, output_audio_path)
        
        background_tasks.add_task(cleanup_file, temp_input)
        background_tasks.add_task(cleanup_file, output_audio_path)
            
        return FileResponse(
            output_audio_path, media_type="audio/mpeg", 
            headers={"X-Agent-Reply": json.dumps(agent_payload)}
        )
    except Exception as e:
        cleanup_file(temp_input)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-booked-dates")
async def get_booked_dates_endpoint(request: BookedDatesRequest):
    """Returns exact dates that are unavailable for a specific room."""
    dates = get_booked_dates_for_room(request.hotel_name, request.room_type)
    return {"booked_dates": dates}

@app.post("/create-payment-order")
async def create_payment_order(request: PaymentInitiationRequest):
    dummy_order_id = f"order_{uuid.uuid4().hex[:12]}"
    return {
        "status": "created",
        "order_id": dummy_order_id,
        "amount": request.amount,
        "currency": request.currency,
        "key_id": "rzp_test_dummy_key"
    }

@app.post("/verify-and-confirm")
async def verify_and_confirm(request: PaymentConfirmationRequest):
    success, message = save_confirmed_booking(
        booking_id=request.booking_id, 
        guest_name=request.guest_name,
        email=request.email,
        phone=request.phone,
        adults=request.adults,
        children=request.children,
        check_in=request.check_in,
        check_out=request.check_out,
        hotel_name=request.hotel_name,
        room_type=request.room_type
    )
    if not success:
        raise HTTPException(status_code=500, detail="Ledger update failed.")
    return {"status": "confirmed", "receipt": request.razorpay_payment_id, "message": message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)