# FastAPI Chat Application for teaching

A chat application built with FastAPI that integrates a web search tool via the Brave Search API. This project uses modern Python features, asynchronous programming, and integrates with [logfire](https://logfire.dev) for logging and instrumentation.

## Features

- **Real-time Chat Interface:** Communicate with the bot over WebSockets.
- **Web Search Integration:** Query the Brave Search API for web results.
- **Logging & Instrumentation:** Uses logfire to capture API calls and performance metrics.
- **Environment Driven:** Configure your API keys and model preferences via a `.env` file.
- **Asynchronous HTTP Client:** Utilizes `httpx.AsyncClient` for HTTP requests.

## Requirements


- Python >= 3.12
- Environment variables set in a `.env` file (see below).

## Installation

### 1. Clone the repository:

```sh
git clone <repository-url>
cd fastapi-chat-application
```

### 2. Set up a virtual environment:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies:

```sh
pip install -r requirements.txt
```

Alternatively, if using `pyproject.toml`, run:

```sh
pip install .
```

## Configuration

Create a `.env` file in the project root with the following contents:

```
BRAVE_API_KEY=your_api_key_here
LOGFIRE_API_KEY=your_logfire_key_here
```

Ensure that your `.env` file is not checked into version control.

## Project Structure

```
fastapi-chat-application/
│── main.py           # FastAPI application, WebSocket endpoint, and Brave Search API integration
│── main2.py          # Sample script demonstrating Pydantic-AI Agent usage
│── .logfire/         # Contains logfire credentials and configuration
│── pyproject.toml    # Project configuration including dependencies
│── .env.example      # Example environment variables file
```

## Running the Application

### Start the FastAPI Server with Uvicorn:

```sh
uvicorn main:app --reload
```

### Access the Chat Interface:

Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

## Chat Flow:

1. The web page includes a chat interface.
2. Type your message in the input box and hit the "Send" button.
3. The WebSocket connection established at the `/ws` endpoint handles sending and receiving messages.

## Code Overview

### Web Search Integration:

The function `search_web` (decorated with `@web_search_agent.tool`) handles requests to the Brave Search API. It uses the `Deps` dataclass for dependency injection, which includes the HTTP client and Brave API key.

### WebSocket Endpoint:

The endpoint at `/ws` creates a WebSocket connection and processes incoming messages by running the web search agent.

## Logging & Instrumentation

The project uses logfire for logging API calls and tracing requests. Logfire is configured both globally in the application and for specific FastAPI routes.

