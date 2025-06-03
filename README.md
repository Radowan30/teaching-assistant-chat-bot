# Teaching Assistant - A FastAPI Chat Application with Video Generation

This document provides instructions for setting up and running the FastAPI chat application with video generation capabilities.

---

## ✨ Features

- **Real-time Chat Interface**: WebSocket-based chat with AI teaching assistant  
- **Web Search Integration**: Uses Brave Search API to answer questions  
- **Educational Animation Generation**: Creates educational videos using Manim and LangGraph (The version with LangGraph is in the 'teaching-assistant-with-langgraph' branch)
- **Responsive UI**: React frontend with Tailwind CSS styling  
- [**Watch the Introduction Video**](https://youtu.be/jnNEZmpi4tA)

---

## 🔧 Prerequisites

### Backend Requirements
- Python 3.11 
- Cairo (2D graphics library)  
- FFmpeg (video processing)  
- LaTeX distribution (MikTeX recommended)  
- OpenAI API key  
- Brave Search API key (for web search functionality)  

### Frontend Requirements
- Node.js and npm  

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd <repo-directory>
```

---

### 2. Backend Setup

### First change directory into the backend folder
```bash
cd .\backend\
```
#### Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Python dependencies:

```bash
pip install -r requirements.txt
```

#### Create a `.env` file in the backend directory with the following:

```env
OPENAI_API_KEY=your_openai_api_key
BRAVE_SEARCH_API_KEY=your_optional_brave_api_key
ALLOWED_ORIGINS=http://localhost:3000
```

#### Create a `public\vidoes\2160p60` directory in the backend folder:

```bash
mkdir public
mkdir public\videos
mkdir public\videos\2160p60
```

---

### 3. Frontend Setup

#### Navigate to the frontend directory:

```bash
cd .. #going back to the root folder incase you are in the backend directory
cd frontend
```

#### Install Node.js dependencies:

```bash
npm install
```

---

## 🚀 Running the Application

### Start the Backend Server

From the backend directory with the virtual environment activated:

```bash
uvicorn chat_app:app --reload
```

The backend will be available at:  
📍 **http://127.0.0.1:8000**

---

### Start the Frontend Development Server

From the frontend directory:

```bash
npm run dev
```

The frontend will be available at:  
📍 **http://localhost:3000**

---

## 💬 Using the Application

1. Open your browser and navigate to [http://localhost:3000](http://localhost:3000)  
2. Type your message in the input box at the bottom of the screen and press **Send**  
3. For general questions, simply type your query  
4. To generate educational videos, start your prompt with the `@video` keyword  
5. To search the web, ask a question that requires current information  


## 🧩 Troubleshooting

- **Manim Issues**: Ensure that **Cairo**, **FFmpeg**, and **LaTeX** are properly installed and available in your system's `PATH`.  
- **Python Compatibility**: If you encounter Python dependency issues, use **Python 3.11** as noted in the `note.txt`.  
- **Video Generation Fails**: Check the **backend console** for specific error messages.  
- **CORS Issues**: Verify that your frontend origin is listed in the **allowed origins** in `chat_app.py`.  

---

## 📌 Additional Notes

- Chat history is stored in a **SQLite database** in the backend directory  
- Generated videos are accessible via the `/public` endpoint on the backend  
- The project uses **TailwindCSS** for styling; any CSS changes should be made through the appropriate Tailwind utility classes  
- The Manim code generation technique is inspired by the [Generative Manim](https://github.com/marcelo-earth/generative-manim/tree/main/api) GitHub repository 
