"""
FastAPI Backend for Waste Classification
Classifies waste into: ORGANIC, RECYCLABLE, HAZARDOUS, GENERAL
Uses YOLOv8 model (primary) + Gemini AI for generating awareness tips
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
print(f"📁 Loaded .env from: {env_path}")

# Import custom modules
from utils import (
    get_dustbin_color, 
    get_dustbin_icon, 
    normalize_class_name,
    validate_image_format
)
from gemini_service import generate_awareness_tip, generate_safety_warning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="Personal waste-segregation assistant powered by YOLOv8 and Gemini AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - use garbage_detect.pt (89.7% accuracy, 8 classes mapped to 4 categories)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), '..', 'models', 'garbage_detect.pt'))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"

# Global model variable
model: Optional[YOLO] = None


@app.on_event("startup")
async def startup_event():
    """Load YOLOv8 model on application startup"""
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


@app.get("/")
async def root():
    """Serve frontend HTML"""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {
            "message": "Waste Classification API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "classify": "/api/classify",
                "categories": "/api/categories"
            }
        }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/classify")
async def classify_waste(file: UploadFile = File(...)):
    """
    Main classification endpoint
    
    Accepts image file and returns:
    - Waste category (ORGANIC/RECYCLABLE/HAZARDOUS)
    - Dustbin color and icon
    - AI-generated awareness tip
    - Confidence score
    """
    # Validate model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_IMAGE_SIZE/1024/1024}MB"
        )
    
    # Validate file type - check both filename and content-type
    valid_content_types = {
        'image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 
        'image/webp', 'image/gif', 'image/tiff'
    }
    is_valid_by_name = validate_image_format(file.filename)
    is_valid_by_type = file.content_type and file.content_type.lower() in valid_content_types
    
    if not is_valid_by_name and not is_valid_by_type:
        logger.warning(f"Invalid file: name={file.filename}, type={file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: JPG, PNG, JPEG, BMP, WEBP, GIF, TIFF"
        )
    
    try:
        # Process image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # ===== PRIMARY: YOLOv8 garbage detection model =====
        # Model detects: Bottle, Cup, Glass, Hazardous, Metal, Organic, Plastic, Plastik
        # We map these 8 classes to 4 categories: RECYCLABLE, ORGANIC, HAZARDOUS, GENERAL
        logger.info("🔍 Running YOLOv8 detection...")
        results = model(image, verbose=False, conf=0.25)
        
        # Process results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            class_names = results[0].names
            
            # Collect all detections and map to categories
            category_scores = {"RECYCLABLE": 0.0, "ORGANIC": 0.0, "HAZARDOUS": 0.0, "GENERAL": 0.0}
            best_item_name = None
            best_conf = 0.0
            
            for cls_id, conf in zip(classes, confidences):
                item_name = class_names[int(cls_id)]
                category = normalize_class_name(item_name)
                conf_val = float(conf)
                if conf_val > category_scores[category]:
                    category_scores[category] = conf_val
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_item_name = item_name
            
            # Pick the category with the highest confidence
            category = max(category_scores, key=category_scores.get)
            confidence = category_scores[category]
            yolo_class_name = best_item_name
            
            logger.info(f"Detected: {best_item_name} -> {category} ({confidence:.2f})")
            logger.info(f"Category scores: {category_scores}")
            
            is_safe_classification = confidence >= CONFIDENCE_THRESHOLD
            
            # Get dustbin info
            dustbin_color = get_dustbin_color(category)
            dustbin_icon = get_dustbin_icon(category)
            
            # Generate awareness tip using Gemini
            logger.info("Generating awareness tip...")
            awareness_tip = generate_awareness_tip(yolo_class_name, category, confidence)
            
            # Generate safety warning if needed
            safety_warning = generate_safety_warning(confidence)
            
            # Build response
            response = {
                "success": True,
                "category": category,
                "confidence": round(confidence, 4),
                "dustbin_color": dustbin_color,
                "dustbin_icon": dustbin_icon,
                "explanation": awareness_tip,
                "safety_warning": safety_warning,
                "is_safe_classification": is_safe_classification,
                "detected_item": yolo_class_name,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": "YOLOv8 Custom Trained"
            }
            
            logger.info(f"✅ Classification successful: {category} (confidence: {confidence:.2f})")
            return response
        
        else:
            # No waste detected
            logger.warning("No waste detected in image")
            return {
                "success": False,
                "category": "HAZARDOUS",  # Safety default
                "confidence": 0.0,
                "dustbin_color": "red",
                "dustbin_icon": "warning",
                "explanation": "No recognizable waste item detected. For safety, treat unknown items as hazardous waste.",
                "safety_warning": "⚠️ Unable to identify item - dispose as HAZARDOUS for safety",
                "is_safe_classification": False,
                "detected_item": None,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    except Exception as e:
        logger.error(f"❌ Classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/categories")
async def get_categories():
    """Get available waste categories and their properties"""
    return {
        "categories": [
            {
                "name": "ORGANIC",
                "dustbin_color": "green",
                "icon": "leaf",
                "description": "Organic waste that decomposes naturally. Examples: food scraps, garden waste."
            },
            {
                "name": "RECYCLABLE",
                "dustbin_color": "blue",
                "icon": "recycle",
                "description": "Materials that can be reprocessed. Examples: plastic, paper, glass, metal."
            },
            {
                "name": "HAZARDOUS",
                "dustbin_color": "red",
                "icon": "warning",
                "description": "Waste that poses risks to health or environment. Examples: batteries, chemicals, e-waste."
            }
        ],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


# ============================================================
# DASHBOARD ROUTES (merged from Flask dashboard)
# ============================================================

import csv

EVIDENCE_FOLDER = Path(__file__).parent.parent / "evidence"
IMAGES_FOLDER = EVIDENCE_FOLDER / "images"
VIDEOS_FOLDER = EVIDENCE_FOLDER / "videos"
LOGS_FILE = EVIDENCE_FOLDER / "logs" / "events.csv"
DASHBOARD_TEMPLATES = Path(__file__).parent.parent / "dashboard" / "templates"


@app.get("/dashboard")
async def dashboard():
    """Serve littering detection dashboard."""
    html_path = DASHBOARD_TEMPLATES / "dashboard.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "Dashboard template not found"}


@app.get("/incidents")
async def incidents_page():
    """Serve incidents page."""
    html_path = DASHBOARD_TEMPLATES / "incidents.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "Incidents template not found"}


@app.get("/incident_detail")
async def incident_detail_page():
    """Serve incident detail page."""
    html_path = DASHBOARD_TEMPLATES / "incident_detail.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "Incident detail template not found"}


@app.get("/api/incidents")
async def get_incidents():
    """Get all littering incidents."""
    incidents = []
    
    if LOGS_FILE.exists():
        try:
            with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('image_path'):
                        row['image_path'] = os.path.basename(row['image_path'])
                    if row.get('video_path'):
                        row['video_path'] = os.path.basename(row['video_path'])
                    incidents.append(row)
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
    
    if IMAGES_FOLDER.exists():
        for filename in os.listdir(IMAGES_FOLDER):
            if filename.endswith(('.jpg', '.png')):
                try:
                    parts = filename.replace('litter_', '').replace('.jpg', '').replace('.png', '')
                    timestamp = parts.replace('_', ' ').replace('-', ':', 2)
                    exists = any(i.get('image_path', '').endswith(filename) for i in incidents)
                    if not exists:
                        incidents.append({
                            'timestamp': timestamp,
                            'event_type': 'LITTERING',
                            'image_path': filename,
                            'video_path': '',
                            'garbage_class': 'Unknown',
                            'garbage_confidence': '0.0',
                            'person_confidence': '0.0'
                        })
                except Exception:
                    pass
    
    incidents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return incidents


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    incidents = []
    if LOGS_FILE.exists():
        try:
            with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                incidents = list(reader)
        except Exception:
            pass
    
    image_count = 0
    if IMAGES_FOLDER.exists():
        image_count = len([f for f in os.listdir(IMAGES_FOLDER) if f.endswith(('.jpg', '.png'))])
    
    video_count = 0
    if VIDEOS_FOLDER.exists():
        video_count = len([f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi'))])
    
    return {
        'total_incidents': max(len(incidents), image_count),
        'total_images': image_count,
        'total_videos': video_count,
        'today_incidents': sum(1 for i in incidents if datetime.now().strftime('%Y-%m-%d') in i.get('timestamp', ''))
    }


@app.get("/evidence/images/{filename}")
async def serve_evidence_image(filename: str):
    """Serve evidence images."""
    file_path = IMAGES_FOLDER / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/evidence/videos/{filename}")
async def serve_evidence_video(filename: str):
    """Serve evidence videos."""
    file_path = VIDEOS_FOLDER / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Video not found")


# Mount static files for frontend (must be AFTER all route definitions)
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
