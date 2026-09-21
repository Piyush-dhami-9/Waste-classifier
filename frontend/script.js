// ========================================
// Smart Waste Classifier - Frontend Logic
// ========================================

// Configuration - Use relative URL for Vercel (frontend and API on same domain)
const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB (Vercel limit)

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const classifyBtn = document.getElementById('classifyBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Result Elements
const dustbinVisual = document.getElementById('dustbinVisual');
const dustbinIcon = document.getElementById('dustbinIcon');
const categoryTitle = document.getElementById('categoryTitle');
const dustbinColorText = document.getElementById('dustbinColorText');
const colorIndicator = document.getElementById('colorIndicator');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const safetyWarning = document.getElementById('safetyWarning');
const safetyWarningText = document.getElementById('safetyWarningText');
const awarenessTip = document.getElementById('awarenessTip');
const tryAnotherBtn = document.getElementById('tryAnotherBtn');

// State
let selectedFile = null;

// ========================================
// Event Listeners
// ========================================

// Upload area click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File input change
imageInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Change image button
changeImageBtn.addEventListener('click', () => {
    resetUpload();
});

// Classify button
classifyBtn.addEventListener('click', () => {
    classifyWaste();
});

// Try another button
tryAnotherBtn.addEventListener('click', () => {
    resetAll();
});

// ========================================
// File Handling
// ========================================

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, etc.)');
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showError(`File too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB`);
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        classifyBtn.classList.remove('hidden');
        errorMessage.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    uploadArea.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    classifyBtn.classList.add('hidden');
    errorMessage.classList.add('hidden');
}

function resetAll() {
    resetUpload();
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
}

// ========================================
// API Communication
// ========================================

async function classifyWaste() {
    if (!selectedFile) return;

    // Show loading state
    setLoading(true);
    errorMessage.classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_URL}/api/classify`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Classification failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Classification error:', error);
        showError(error.message || 'Failed to classify image. Please try again.');
        setLoading(false);
    }
}

// ========================================
// Results Display
// ========================================

function displayResults(result) {
    // Hide upload section, show results
    uploadSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    setLoading(false);

    // Extract data
    const { category, confidence, dustbin_color, dustbin_icon, explanation, safety_warning } = result;

    // Set category
    categoryTitle.textContent = category;
    categoryTitle.className = `category-title ${category.toLowerCase()}`;

    // Set dustbin visual
    dustbinVisual.className = `dustbin ${dustbin_color}`;
    dustbinIcon.innerHTML = getIconSVG(dustbin_icon);

    // Set dustbin label
    const colorName = dustbin_color.charAt(0).toUpperCase() + dustbin_color.slice(1);
    dustbinColorText.textContent = `${colorName} Dustbin`;
    colorIndicator.className = `color-indicator ${dustbin_color}`;

    // Set confidence
    const confidencePercent = Math.round(confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${confidencePercent}%`;

    // Update progress bar color based on confidence
    if (confidencePercent >= 70) {
        confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
    } else if (confidencePercent >= 50) {
        confidenceFill.style.background = 'linear-gradient(90deg, #fbbf24, #f59e0b)';
    } else {
        confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
    }

    // Set safety warning
    if (safety_warning && safety_warning.trim() !== '') {
        safetyWarning.classList.remove('hidden');
        safetyWarningText.textContent = safety_warning;
    } else {
        safetyWarning.classList.add('hidden');
    }

    // Set awareness tip
    awarenessTip.textContent = explanation;

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// ========================================
// Icon SVGs
// ========================================

function getIconSVG(iconName) {
    const icons = {
        leaf: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M3 21c3-6 6-9 9-9s6 3 9 9M9 12c0-1.5.5-3 1.5-4.5S13 5 15 5c0 3-1 5-2.5 6.5S9 13.5 9 12z"/>
            </svg>
        `,
        recycle: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
        `,
        warning: `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
            </svg>
        `
    };

    return icons[iconName] || icons.warning;
}

// ========================================
// UI Helpers
// ========================================

function setLoading(isLoading) {
    if (isLoading) {
        classifyBtn.disabled = true;
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
    } else {
        classifyBtn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

// ========================================
// Health Check on Load
// ========================================

async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend is healthy');
        } else {
            console.warn('⚠️ Backend health check failed');
        }
    } catch (error) {
        console.warn('⚠️ Backend not reachable:', error.message);
        showError('Backend server not available. Please start the server first.');
    }
}

// Check backend on page load
window.addEventListener('load', () => {
    checkBackendHealth();
});
