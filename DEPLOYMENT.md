# Deployment Guide: Classroom Face Recognition Web Application

This document provides clear, step-by-step instructions to deploy your Face Recognition & Attendance Web Application.

---

## 📁 Clean Project Structure

Your project is grouped neatly into modular layers:

```text
face-recognition/
├── frontend/                  # Web Interface (HTML, CSS, JS)
│   ├── index.html             # Clean white-theme UI layout
│   ├── style.css              # Modern CSS design system
│   └── app.js                 # Drag & drop, API integration, copy actions
│
├── src/                       # Core AI Backend Pipeline (UNTOUCHED)
│   ├── __init__.py
│   ├── detector.py            # MTCNN face detector & alignment
│   ├── embedder.py            # FaceNet InceptionResnetV1 (512-d)
│   ├── recognizer.py          # Cosine similarity template matching
│   ├── preprocessing.py       # Quality checks & affine alignment
│   └── utils.py               # Bounding box & annotation drawing
│
├── scripts/                   # CLI Tools (UNTOUCHED)
│   ├── enroll.py              # Enrolls student folders into embeddings DB
│   ├── recognize.py           # CLI runner for test images
│   └── evaluate.py            # Accuracy & threshold evaluation
│
├── embeddings/                # Pre-calculated Knowledge Base
│   └── face_embeddings.pkl    # 528 enrolled student face embeddings
│
├── test/                      # Sample classroom photos for testing
│   ├── test-6.jpeg
│   ├── test-2.jpeg
│   └── ...
│
├── app.py                     # FastAPI application serving UI + AI API
├── requirements.txt           # Local dependencies
├── requirements-prod.txt      # Cloud Linux production dependencies (CPU-optimized)
├── render.yaml                # Render Blueprint (Infrastructure as code)
├── Dockerfile                 # Production Docker container definition
├── Procfile                   # Process command for PaaS platforms
├── vercel.json                # Vercel configuration
├── .gitignore                 # Excludes 1GB+ local venv, caches, and temp files
└── .dockerignore              # Keeps Docker build context minimal (<10MB)
```

---

## 🚀 Recommended Deployment Platform: **Render**

### Why Render?
- **PyTorch Support**: FaceNet and MTCNN require deep learning libraries (`torch`, `torchvision`, `facenet-pytorch`) that exceed **700 MB** uncompressed.
- **Serverless Limits (Vercel)**: Vercel has a hard **250 MB total package limit** for serverless functions, making full PyTorch models fail with `FUNCTION_PAYLOAD_TOO_LARGE`.
- **In-Memory Models**: Render keeps the web service running continuously, so models remain in memory. Detection takes **1–2 seconds**, avoiding cold-start model reloads.
- **Free Tier Available**: Render offers a free tier for Web Services.

---

## 📋 Method 1: Deploy on Render (Recommended, ~3 Minutes)

### Step 1: Initialize Git and Push to GitHub

Open PowerShell in the project directory:

```powershell
cd c:\Users\kodal\OneDrive\Desktop\PROJECTS\face-recognition

# Initialize git repository
git init

# Add all files (the .gitignore automatically excludes the 1GB local venv)
git add .

# Create initial commit
git commit -m "Initial commit with clean frontend and cloud deployment configuration"

# Add your GitHub remote repository and push
git remote add origin https://github.com/YOUR_USERNAME/face-recognition.git
git branch -M main
git push -u origin main
```

> **Note:** Because of the `.gitignore`, your upload is fast and lightweight (under 35 MB including test images).

---

### Step 2: Create Web Service on Render

1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → Select **Web Service**.
3. Choose **Build and deploy from a Git repository**.
4. Select your `face-recognition` GitHub repository.

---

### Step 3: Configure Settings

Render will detect the project. Fill in the following settings:

| Setting | Value |
|---|---|
| **Name** | `classroom-face-recognition` (or any name you choose) |
| **Language / Runtime** | `Python` |
| **Region** | Closest to your users (e.g., `Oregon (US West)` or `Frankfurt`) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements-prod.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` (or Starter for higher RAM) |

#### Environment Variables (under Advanced):
- Add `PYTHON_VERSION`: `3.11.9`

Click **Create Web Service**.

---

### Step 4: Done!
Render will:
1. Download CPU-optimized PyTorch wheels quickly (using `requirements-prod.txt`).
2. Install headless OpenCV (`opencv-python-headless` so no GUI library errors occur).
3. Start the server and provide a public URL like `https://classroom-face-recognition.onrender.com`.

---

## 🐳 Method 2: Deploy on Render via Docker (Alternative)

If you prefer containerized deployment:

1. In Render, select **New +** → **Web Service**.
2. Connect your repository.
3. For **Runtime**, choose **Docker**.
4. Render will automatically detect the [`Dockerfile`](Dockerfile) in the root directory.
5. Click **Create Web Service**.

---

## ⚡ Method 3: Deploying on Vercel

If you want the **frontend** hosted on Vercel:

1. Deploy the backend on **Render** (as explained in Method 1) to get your backend URL, for example:
   `https://classroom-face-recognition.onrender.com`
2. In `frontend/app.js`, change the API base URL from relative `/api/...` to your Render URL:
   ```javascript
   const API_BASE_URL = "https://classroom-face-recognition.onrender.com";
   ```
3. Deploy to Vercel via Vercel CLI or Git:
   ```bash
   vercel
   ```

*(For single-URL simplicity with zero CORS setup, deploying everything together on Render is recommended.)*

---

## 🛠️ Testing Locally Before Deploying

To run and verify the app locally at any time:

```bash
python app.py
```
Or simply double-click [`run_app.bat`](run_app.bat).

Open your browser at:
👉 **`http://127.0.0.1:8000`**
