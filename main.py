from __future__ import annotations as _annotations
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import random
import logfire
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')

model = OpenAIModel(llm)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

web_search_agent = Agent(
    model,
    system_prompt=f'You are an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
    deps_type=Deps,
    retries=2
)

@web_search_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    with logfire.span('calling Brave search API', query=web_query) as span:
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': web_query,
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."


app = FastAPI()

logfire.configure(service_name="chatbot")
logfire.instrument_fastapi(app)

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Chatbot</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="flex flex-col items-center justify-center h-screen bg-gray-100">
    <h2 class="text-2xl font-bold mb-4">Chatbot</h2>
    <div id="chat" class="w-96 h-80 overflow-y-auto bg-white p-4 rounded-lg shadow-md mb-4"></div>
    <div class="flex space-x-2">
        <input type="text" id="messageInput" autocomplete="off" class="border p-2 rounded-lg w-72"/>
        <button onclick="sendMessage()" class="bg-blue-500 text-white px-4 py-2 rounded-lg">Send</button>
    </div>
    <script>
        var ws = new WebSocket("ws://localhost:8000/ws");
        
        ws.onopen = function(event) {
            console.log("Connection established");
        };

        ws.onmessage = function(event) {
            var chat = document.getElementById('chat');
            var message = document.createElement('p');
            console.log(event.data);
            message.textContent = "Bot: " + event.data;
            message.classList.add("bg-gray-200", "p-2", "rounded-lg", "mb-2");
            chat.appendChild(message);
            chat.scrollTop = chat.scrollHeight;
        };
        
        function sendMessage() {
            var input = document.getElementById("messageInput");
            var message = document.createElement('p');
            message.textContent = "You: " + input.value;
            message.classList.add("bg-blue-500", "text-white", "p-2", "rounded-lg", "mb-2", "self-end");
            document.getElementById('chat').appendChild(message);
            ws.send(input.value);
            input.value = "";
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get():
    return html


# agent = Agent(  
#     'openai:gpt-4o-mini',
#     system_prompt='You are an open source developer, and you did a lot of stuff yesterday.',  
# )

# @agent.tool_plain
# def what_i_did_yesterday():
#     stuff_done = [
#         "I wrote a new blog post",
#         "I fixed a bug in my project",
#         "I learned a new programming language",
#         "I released a new version of my project",
#     ]
#     return random.choice(stuff_done)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with AsyncClient() as client:
        brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = Deps(client=client, brave_api_key=brave_api_key)
        async for data in websocket.iter_text():
            result = await web_search_agent.run(user_prompt=data, message_history=[], deps=deps)
            await websocket.send_text(result.data)


# async def main():
#     async with AsyncClient() as client:
#         brave_api_key = os.getenv('BRAVE_API_KEY', None)
#         deps = Deps(client=client, brave_api_key=brave_api_key)

#         result = await web_search_agent.run(
#             'Give me some articles talking about the new release of React 19.', deps=deps
#         )
        
#         debug(result)
#         print('Response:', result.data)

