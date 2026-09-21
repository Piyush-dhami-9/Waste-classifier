"""
Gemini API Integration for Waste Classification
Uses Gemini AI for generating educational awareness tips about waste disposal
Classification is done by our custom-trained YOLOv8 model
"""

import os
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

from utils import get_fallback_awareness_tip

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"

# Initialize Gemini for text generation
model = None
if GEMINI_API_KEY and ENABLE_GEMINI and GENAI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("\u2705 Gemini API configured for awareness tip generation (gemini-1.5-flash)")
    except Exception as e:
        print(f"\u26a0\ufe0f Gemini API configuration failed: {e}")
        model = None
else:
    if not GENAI_AVAILABLE:
        print("\u26a0\ufe0f Gemini API library not available - using fallback tips")
    else:
        print("\u26a0\ufe0f Gemini API disabled or no API key - using fallback tips")


def generate_awareness_tip(item_name: str, category: str, confidence: float) -> str:
    """
    Generate educational awareness tip using Gemini AI
    
    Args:
        item_name: Detected item name (from YOLO)
        category: Classification (ORGANIC/RECYCLABLE/HAZARDOUS)
        confidence: Model confidence score
    
    Returns:
        str: Awareness tip (Gemini-generated or fallback)
    """
    # Use fallback if Gemini is disabled or not configured
    if not model or not ENABLE_GEMINI:
        return get_fallback_awareness_tip(category)
    
    try:
        # Generate contextual prompt for Gemini
        prompt = f"""You are a friendly waste management expert helping people sort their garbage correctly.

ITEM DETECTED: {item_name}
CATEGORY: {category} waste

Write a SHORT, CLEAR awareness tip (2-3 sentences) that includes:
1. Why this belongs in {category} category
2. The correct disposal method
3. ONE environmental impact fact

Rules:
- Be direct and helpful
- Use simple language anyone can understand
- Include a practical disposal tip
- DO NOT use emojis
- Keep under 200 characters"""

        # Call Gemini API
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.6,
                'max_output_tokens': 150,
            }
        )
        
        # Extract and clean response
        if response and response.text:
            tip = response.text.strip()
            # Ensure it's not too long
            if len(tip) > 250:
                tip = tip[:247] + "..."
            return tip
        else:
            return get_fallback_awareness_tip(category)
    
    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        # Always return fallback on error
        return get_fallback_awareness_tip(category)


def generate_safety_warning(confidence: float) -> str:
    """
    Generate safety warning for low-confidence predictions
    
    Args:
        confidence: Model confidence score
    
    Returns:
        str: Safety warning message
    """
    if confidence < 0.5:
        return "⚠️ Low confidence detection. When unsure, treat as HAZARDOUS waste for safety."
    elif confidence < 0.7:
        return "⚠️ Uncertain classification. Please verify or consult local waste guidelines."
    else:
        return ""
