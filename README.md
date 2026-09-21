<div align="center">

<!-- TOP BANNER -->
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:000000,30:001a0d,70:001a1a,100:000000&height=220&text=SMARTWASTE%20AI&fontSize=62&fontColor=00ff88&fontAlignY=45&desc=◈%20WASTE%20CLASSIFICATION%20%7C%20LITTERING%20DETECTION%20◈&descAlignY=68&descSize=14&animation=blinking&stroke=00ff88&strokeWidth=2" width="100%"/>

<br/>

<!-- FUTURISTIC STAT CARDS -->
<table border="0" cellspacing="0" cellpadding="0">
<tr>
  <td align="center" width="140">
    <img src="https://img.shields.io/badge/─────────────────────────────────────────────────────────────────────-000000?style=for-the-badge" width="0"/>
    <img src="https://img.shields.io/badge/SYSTEM-ONLINE-00ff88?style=for-the-badge&labelColor=001a0d&logo=statuspage&logoColor=00ff88"/>
  </td>
  <td align="center" width="140">
    <img src="https://img.shields.io/badge/MODELS-5%20ACTIVE-00f5ff?style=for-the-badge&labelColor=001a1a&logo=pytorch&logoColor=00f5ff"/>
  </td>
  <td align="center" width="140">
    <img src="https://img.shields.io/badge/ACCURACY-99.4%25_PEAK-ffd700?style=for-the-badge&labelColor=1a1500&logo=target&logoColor=ffd700"/>
  </td>
  <td align="center" width="140">
    <img src="https://img.shields.io/badge/MODULES-2%20RUNNING-a855f7?style=for-the-badge&labelColor=0f0020&logo=buffer&logoColor=a855f7"/>
  </td>
  <td align="center" width="140">
    <img src="https://img.shields.io/badge/GPU-RTX%203050-ff006e?style=for-the-badge&labelColor=1a0010&logo=nvidia&logoColor=ff006e"/>
  </td>
</tr>
</table>

<br/>

<!-- TECH STACK BADGES ROW 1 -->
<img src="https://img.shields.io/badge/Python-3.13-00ff88?style=for-the-badge&logo=python&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/FastAPI-0.109-00f5ff?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/Flask-3.0-a855f7?style=for-the-badge&logo=flask&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/PyTorch-Latest-ff006e?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=0a0a0a"/>

<!-- TECH STACK BADGES ROW 2 -->

<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-ffd700?style=for-the-badge&logo=opencv&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/OpenCV-Vision-00f5ff?style=for-the-badge&logo=opencv&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/Gemini-1.5_Flash-ff006e?style=for-the-badge&logo=google&logoColor=white&labelColor=0a0a0a"/>
<img src="https://img.shields.io/badge/License-Custom-00ff88?style=for-the-badge&labelColor=0a0a0a"/>

<br/><br/>

<!-- TAGLINE -->

```
█▀ █▀▄▀█ ▄▀█ █▀█ ▀█▀ █ █ █ ▄▀█ █▀ ▀█▀ █▀▀   ▄▀█ █
▄█ █ ▀ █ █▀█ █▀▄  █  ▀▄▀▄▀ █▀█ ▄█  █  ██▄   █▀█ █
```

> 🌍 **AI-powered waste classification & real-time littering detection**
> Powered by **YOLOv8 · Computer Vision · Google Gemini AI · Python 3.13**

<br/>

<!-- NAV LINKS -->
[![Quick Start](https://img.shields.io/badge/🚀_QUICK_START-000?style=for-the-badge&labelColor=001a0d&color=00ff88)](#-quick-start)
[![Architecture](https://img.shields.io/badge/🏗_ARCHITECTURE-000?style=for-the-badge&labelColor=001a1a&color=00f5ff)](#️-architecture)
[![Models](https://img.shields.io/badge/🤖_MODELS-000?style=for-the-badge&labelColor=1a1500&color=ffd700)](#-model-performance)
[![Structure](https://img.shields.io/badge/📁_STRUCTURE-000?style=for-the-badge&labelColor=0f0020&color=a855f7)](#-project-structure)
[![Contribute](https://img.shields.io/badge/🤝_CONTRIBUTE-000?style=for-the-badge&labelColor=1a0010&color=ff006e)](#-contributing)

</div>

---

## 🎯 Two Powerful Modules

<table>
<tr>
<td width="50%" align="center">

### 🗑️ Module 1 — Waste Classifier
**FastAPI + YOLOv8 + Gemini AI**

Upload a photo → AI classifies waste type → Recommends correct dustbin color → Generates eco-tips

`Port 8000` · `4 Waste Categories` · `Gemini Tips`

</td>
<td width="50%" align="center">

### 📹 Module 2 — Littering Detector
**OpenCV + 4 YOLOv8 Models**

Real-time camera monitoring → Detects littering behavior → 10-second grace period → Auto-captures evidence

`Port 5000` · `Live Detection` · `Evidence Saved`

</td>
</tr>
</table>

---

## 🏗️ Architecture

### 🗑️ Module 1 — Waste Classifier · Port 8000

```mermaid
graph TB
    A([📸 User Uploads Waste Image])
    A --> B[🌐 Frontend\nindex.html · script.js · style.css]
    B --> C[⚙️ FastAPI Server\nbackend/app.py · Port 8000]
    C --> D[🤖 YOLOv8 Classification Model\nwaste_classify.pt · 17.6 MB · 53.6% mAP50]
    D --> E{🗂️ 4 Waste Categories}

    E --> E1[♻️ RECYCLABLE\nBottles · Paper · Cardboard · Metal]
    E --> E2[🌱 ORGANIC\nFood · Leaves · Flowers · Fruit]
    E --> E3[⚠️ HAZARDOUS\nBatteries · Chemicals · Medical]
    E --> E4[⬛ GENERAL\nMixed · Non-Recyclable · Other]

    E1 --> BIN1[🔵 Blue Bin]
    E2 --> BIN2[🟢 Green Bin]
    E3 --> BIN3[🔴 Red Bin]
    E4 --> BIN4[⬛ Black Bin]

    BIN1 & BIN2 & BIN3 & BIN4 --> G[🧠 Gemini 1.5 Flash AI\ngemini_service.py\nEco Tips · Disposal Info · Fun Facts]
    G --> H([🖥️ Display Result\nBin Color · Category · Confidence % · AI Tips])

    style A fill:#001a0d,stroke:#00ff88,color:#00ff88
    style C fill:#001a1a,stroke:#00f5ff,color:#00f5ff
    style D fill:#1a1500,stroke:#ffd700,color:#ffd700
    style E fill:#0f0020,stroke:#a855f7,color:#a855f7
    style G fill:#1a0010,stroke:#ff006e,color:#ff006e
    style H fill:#001a0d,stroke:#00ff88,color:#00ff88
    style E1 fill:#001a33,stroke:#00f5ff,color:#00f5ff
    style E2 fill:#001a0d,stroke:#00ff88,color:#00ff88
    style E3 fill:#1a0000,stroke:#ff006e,color:#ff006e
    style E4 fill:#111111,stroke:#888888,color:#aaaaaa
    style BIN1 fill:#001a33,stroke:#00f5ff,color:#00f5ff
    style BIN2 fill:#001a0d,stroke:#00ff88,color:#00ff88
    style BIN3 fill:#1a0000,stroke:#ff006e,color:#ff006e
    style BIN4 fill:#111111,stroke:#888888,color:#aaaaaa
```

---

### 📹 Module 2 — Littering Detector · Port 5000

```mermaid
graph TB
    CAM([🎥 Webcam Feed Stream\nReal-Time Video Input])

    CAM --> P1[👤 Person Detection\nYOLOv8n COCO · 6.3 MB]
    P1 --> P2[✋ Hand Tracking\nhand_detect.pt · 6.0 MB · 99.4% mAP50]
    P2 --> P3[🗑️ Garbage Detection\ngarbage_detect.pt · 5.9 MB · 89.7% mAP50]
    P3 --> P4[🪣 Dustbin Detection\ndustbin_detect.pt · 6.0 MB · 96.6% mAP50]
    P4 --> P5[🧠 Littering Decision Logic\nlittering_decision.py]

    P5 --> GP{⏱️ Grace Period\n10 Seconds}

    GP -->|✅ Disposed in Dustbin| OK([✅ No Violation\nReset Detection])

    GP -->|❌ Littering Detected!| EV[📁 Evidence Manager\nevidence_manager.py]

    EV --> SS[📸 Screenshot\nevidence/images/]
    EV --> VID[🎬 Video Clip\nevidence/videos/]
    EV --> LOG[📋 Event Log\nevidence/logs/]

    SS & VID & LOG --> DASH([🖥️ Flask Dashboard\ndashboard/app.py · Port 5000\nCyberpunk Evidence UI])

    style CAM fill:#001a1a,stroke:#00f5ff,color:#00f5ff
    style P1 fill:#001a0d,stroke:#00ff88,color:#00ff88
    style P2 fill:#001a0d,stroke:#00ff88,color:#00ff88
    style P3 fill:#1a1500,stroke:#ffd700,color:#ffd700
    style P4 fill:#1a1500,stroke:#ffd700,color:#ffd700
    style P5 fill:#0f0020,stroke:#a855f7,color:#a855f7
    style GP fill:#1a0a00,stroke:#ff8800,color:#ff8800
    style OK fill:#001a0d,stroke:#00ff88,color:#00ff88
    style EV fill:#1a0010,stroke:#ff006e,color:#ff006e
    style SS fill:#1a0010,stroke:#ff006e,color:#ff006e
    style VID fill:#1a0010,stroke:#ff006e,color:#ff006e
    style LOG fill:#1a0010,stroke:#ff006e,color:#ff006e
    style DASH fill:#001a1a,stroke:#00f5ff,color:#00f5ff
```

---

## 🔄 Request Flow

```mermaid
sequenceDiagram
    actor User
    participant FE as 🌐 Frontend
    participant API as ⚙️ FastAPI :8000
    participant YOLO as 🤖 YOLOv8
    participant GEM as 🧠 Gemini AI

    User->>FE: Upload Waste Image
    FE->>API: POST /classify
    API->>YOLO: Run Inference
    YOLO-->>API: Category + Confidence %

    alt ✅ High Confidence
        API->>GEM: Generate Eco Tips
        GEM-->>API: Tips + Fun Facts
        API-->>FE: JSON Result
        FE-->>User: Bin Color + Category + Tips
    else ⚠️ Low Confidence
        API-->>FE: Low Confidence Warning
        FE-->>User: Please retake photo
    end
```

---

## 📹 Littering Detection Flow

```mermaid
sequenceDiagram
    participant CAM as 🎥 Webcam
    participant DET as 🔍 Multi-Model Pipeline
    participant LOG as ⏱️ Grace Period Logic
    participant EV as 📁 Evidence Manager
    participant DASH as 🖥️ Flask Dashboard :5000

    loop Every Frame
        CAM->>DET: Video Frame
        DET->>DET: Detect Person
        DET->>DET: Track Hand
        DET->>DET: Detect Garbage
        DET->>DET: Locate Dustbin
    end

    DET->>LOG: Garbage in Hand + No Dustbin Disposal
    LOG->>LOG: Start 10-sec Grace Period

    alt ✅ Disposed in Time
        LOG-->>DET: Reset — No Violation
    else ❌ Littered
        LOG->>EV: Capture Screenshot + Video Clip
        EV->>DASH: Save Evidence + Log
        DASH-->>EV: Stored at /evidence
    end
```

---

## ⚡ Data Pipeline

```mermaid
flowchart LR
    A([📷 Input\nImage/Camera]) --> B{Source?}
    B -->|Upload| C[FastAPI\nPort 8000]
    B -->|Webcam| D[OpenCV\nStream]

    C --> E[YOLOv8\nClassifier]
    D --> F[4-Model\nPipeline]

    E --> G{Category}
    G --> G1[♻️ Recyclable]
    G --> G2[🌱 Organic]
    G --> G3[⚠️ Hazardous]
    G --> G4[⬛ General]

    F --> H[Grace Period\nLogic]
    H -->|Violation| I[Evidence\nCapture]
    H -->|OK| J([✅ Clear])

    G1 & G2 & G3 & G4 --> K[Gemini AI\nTips]
    K --> L([🖥️ Web UI\nResult + Tips])
    I --> M([📊 Dashboard\nEvidence])

    style A fill:#001a0d,stroke:#00ff88,color:#00ff88
    style L fill:#001a1a,stroke:#00f5ff,color:#00f5ff
    style M fill:#1a0010,stroke:#ff006e,color:#ff006e
    style J fill:#001a0d,stroke:#00ff88,color:#00ff88
```

---

## 🤖 Model Performance

| Model | Task | Accuracy | Size | Status |
|-------|------|----------|------|--------|
| 🗑️ **Waste Classifier** | 4-Class Segregation | `53.6% mAP50` | 17.6 MB | ✅ Live |
| ✋ **Hand Detection** | Object Tracking | `99.4% mAP50` | 6.0 MB | ✅ Live |
| 🗑️ **Garbage Detection** | Garbage Objects | `89.7% mAP50` | 5.9 MB | ✅ Live |
| 🪣 **Dustbin Detection** | Dustbin Location | `96.6% mAP50` | 6.0 MB | ✅ Live |
| 👤 **Person Detection** | YOLOv8n COCO | High | 6.3 MB | ✅ Live |

```mermaid
xychart-beta
    title "Model Accuracy (mAP50 %)"
    x-axis ["Waste Classifier", "Hand Detection", "Garbage Detect", "Dustbin Detect"]
    y-axis "mAP50 (%)" 0 --> 100
    bar [53.6, 99.4, 89.7, 96.6]
```

---

## 🎨 Waste Categories

```mermaid
graph LR
    W([🗑️ Waste\nInput]) --> R
    W --> O
    W --> H
    W --> G

    R["♻️ RECYCLABLE\n🔵 Blue Bin\n─────────────\nBottles · Paper\nCardboard · Metal\nPlastic · Glass"]
    O["🌱 ORGANIC\n🟢 Green Bin\n─────────────\nFood · Leaves\nFlowers · Fruit\nVegetables"]
    H["⚠️ HAZARDOUS\n🔴 Red Bin\n─────────────\nBatteries · Chemicals\nMedical Waste\nElectronics"]
    G["⬛ GENERAL\n⬛ Black Bin\n─────────────\nMixed Waste\nNon-Recyclable\nOther"]

    style R fill:#001a33,stroke:#00f5ff,color:#00f5ff
    style O fill:#001a0d,stroke:#00ff88,color:#00ff88
    style H fill:#1a0000,stroke:#ff006e,color:#ff006e
    style G fill:#111111,stroke:#888888,color:#aaaaaa
    style W fill:#1a1500,stroke:#ffd700,color:#ffd700
```

---

## 🛠️ Tech Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| 🎨 **Frontend** | HTML5 · CSS3 · JavaScript | — | Web UI for classifier |
| ⚙️ **Backend** | FastAPI | `0.109` | Waste classification API |
| 📊 **Dashboard** | Flask | `3.0` | Evidence review portal |
| 🤖 **Detection** | YOLOv8 · Ultralytics | latest | All vision models |
| 👁️ **Vision** | OpenCV | latest | Camera stream processing |
| 🧠 **Gen AI** | Google Gemini 1.5 Flash | — | Eco tips generation |
| 🔥 **ML Framework** | PyTorch | latest | Model inference |
| 💻 **Language** | Python | `3.13` | Full stack |
| 🖥️ **Hardware** | NVIDIA RTX 3050 | 4GB VRAM | GPU inference |

### Code Composition

```mermaid
pie title Codebase Language Distribution
    "Python 🐍" : 51.9
    "HTML 🌐" : 37.3
    "CSS 🎨" : 5.7
    "JavaScript ⚡" : 5.1
```

---

## 📁 Project Structure

```
smartwaste-ai/
│
├── 🎯 main.py                        # Littering Detection Entry Point
├── 📦 requirements.txt               # All Dependencies
│
├── 📂 backend/                       # Waste Classification API · Port 8000
│   ├── app.py                        # FastAPI Server
│   ├── gemini_service.py             # Gemini AI Integration
│   └── utils.py                      # Helper Functions
│
├── 📂 frontend/                      # Web UI
│   ├── index.html                    # Main Interface
│   ├── script.js                     # Frontend Logic
│   └── style.css                     # Styling
│
├── 📂 detection/                     # Littering Detection Module
│   ├── person_detection.py           # Person Detection
│   ├── hand_detection.py             # Hand Tracking · 99.4% mAP50
│   ├── garbage_detection.py          # Garbage Detection · 89.7% mAP50
│   ├── dustbin_detection.py          # Dustbin Detection · 96.6% mAP50
│   ├── garbage_classification.py     # Waste Classification
│   ├── 📂 logic/
│   │   └── littering_decision.py     # Grace Period & Warning Logic
│   └── 📂 utils/
│       ├── evidence_manager.py       # Screenshot & Video Capture
│       └── video_utils.py            # Video Processing
│
├── 📂 dashboard/                     # Evidence Dashboard · Port 5000
│   ├── app.py                        # Flask Server
│   └── 📂 templates/                 # Cyberpunk UI
│
├── 📂 models/                        # YOLOv8 Model Weights
│   ├── waste_classify.pt             # Waste Classification · 17.6MB
│   ├── hand_detect.pt                # Hand Detection · 6.0MB
│   ├── garbage_detect.pt             # Garbage Detection · 5.9MB
│   ├── dustbin_detect.pt             # Dustbin Detection · 6.0MB
│   └── person_detect.pt              # Person Detection · 6.3MB
│
├── 📂 evidence/                      # Auto-Captured Evidence
│   ├── 📂 images/                    # Screenshots
│   ├── 📂 videos/                    # Recorded Evidence
│   └── 📂 logs/                      # Event Logs
│
└── 📂 training/                      # Model Training Scripts
    ├── train.py
    ├── train_v2.py
    └── waste_data.yaml
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** `3.13+`
- **NVIDIA GPU** recommended (RTX 3050+ · 4GB VRAM)
- **Webcam** for littering detection module
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com)

### Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smartwaste-ai.git
cd smartwaste-ai

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > backend/.env
```

```bash
# ── MODULE 1: Waste Classifier ──────────────────────────
cd backend
uvicorn app:app --reload --port 8000
# Then open: frontend/index.html in your browser
```

```bash
# ── MODULE 2: Littering Detector ────────────────────────
python main.py
# Controls: q = quit | r = reset
```

```bash
# ── MODULE 3: Evidence Dashboard ────────────────────────
cd dashboard
python app.py
# Open: http://localhost:5000
```

### Port Map

| Service | Port | URL |
|---------|------|-----|
| 🗑️ Waste Classification API | `8000` | `http://localhost:8000` |
| 📊 Evidence Dashboard | `5000` | `http://localhost:5000` |
| 🌐 Frontend UI | — | Open `frontend/index.html` |

---

## 🔐 Privacy & Evidence

```mermaid
graph LR
    CAM([🎥 Camera]) --> DET[Detection\nPipeline]
    DET -->|No Violation| CLR([✅ No Storage])
    DET -->|Littering Detected| EV[Evidence\nManager]
    EV --> SS[📸 Screenshot\nevidence/images/]
    EV --> VID[🎥 Video Clip\nevidence/videos/]
    EV --> LOG[📋 Event Log\nevidence/logs/]
    SS & VID & LOG --> DASH([🖥️ Flask Dashboard\nPort 5000])

    style CAM fill:#001a1a,stroke:#00f5ff,color:#00f5ff
    style CLR fill:#001a0d,stroke:#00ff88,color:#00ff88
    style DASH fill:#1a0010,stroke:#ff006e,color:#ff006e
```

---

## 📈 Roadmap

```mermaid
gantt
    title SmartWaste AI — Development Timeline
    dateFormat  YYYY-MM
    section Module 1 ✅
    YOLOv8 Waste Classifier          :done, m1a, 2025-08, 2025-10
    Gemini AI Integration            :done, m1b, 2025-10, 2025-11
    FastAPI Backend                  :done, m1c, 2025-11, 2025-12
    Web Frontend UI                  :done, m1d, 2025-12, 2026-01
    section Module 2 ✅
    Person + Hand Detection          :done, m2a, 2026-01, 2026-02
    Garbage + Dustbin Detection      :done, m2b, 2026-02, 2026-03
    Grace Period Logic               :done, m2c, 2026-03, 2026-03
    Evidence Manager + Dashboard     :done, m2d, 2026-03, 2026-04
    section Upcoming 🚧
    Mobile App Integration           :active, m3a, 2026-05, 2026-08
    Cloud Evidence Storage           :       m3b, 2026-07, 2026-09
    Multi-Camera Support             :       m3c, 2026-08, 2026-10
    City-Scale Deployment API        :       m3d, 2026-10, 2027-01
```

---

## 🤝 Contributing

```bash
# 1. Fork the repo, then clone
git clone https://github.com/YOUR_USERNAME/smartwaste-ai.git

# 2. Create a feature branch
git checkout -b feature/amazing-feature

# 3. Commit your changes
git commit -m 'feat: add amazing feature'

# 4. Push & open a Pull Request
git push origin feature/amazing-feature
```

### Areas We Need Help With

- 🐛 Bug fixes & model accuracy improvements
- 📱 Mobile app integration
- 🌐 Additional language support for Gemini tips
- 🧪 Test coverage & CI/CD pipeline
- 📊 Analytics dashboard enhancements
- 🗺️ Multi-city deployment infrastructure

---

## 📄 License

```
Copyright (c) 2026 Piyush Dhami. All rights reserved.

This repository is provided for educational and reference purposes only.

PERMITTED:
  * You may view and download the code for personal learning.
  * You may fork this repository on GitHub.
  * You may submit issues, suggestions, and pull requests.

RESTRICTIONS:
  * You may NOT use this code in any personal, academic, or commercial project.
  * You may NOT publish, distribute, or re-upload this code (in whole or in part).
  * You may NOT use this code to build or showcase your own applications.
  * You may NOT claim this code as your own work.

ATTRIBUTION:
  If you are explicitly permitted to share any part of this code, you must
  give clear credit at the beginning:
  "Original code by Piyush Dhami (2026)"

NO LICENSE:
  Only the permissions listed above are allowed.
  Any other use is strictly prohibited.

ENFORCEMENT:
  If you violate these terms, the copyright holder may take
  appropriate legal action.
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:a855f7,50:00f5ff,100:00ff88&height=120&section=footer&animation=fadeIn" width="100%"/>

**Made with 💚 for a cleaner planet**

⭐ Star this repo · 🐛 Report issues · 🤝 Contribute · 📢 Share

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=smartwaste-ai&style=for-the-badge)

</div>
