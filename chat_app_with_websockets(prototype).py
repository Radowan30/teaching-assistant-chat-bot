from __future__ import annotations as _annotations
import fastapi
from fastapi import FastAPI, WebSocket
from fastapi import Depends, Request
from fastapi.responses import HTMLResponse
import random
import logfire
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from httpx import AsyncClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext
import sqlite3
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Annotated, Any, Callable, Literal, TypeVar
from typing_extensions import LiteralString, ParamSpec, TypedDict
from functools import partial
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from fastapi.logger import logger as fastapi_logger
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelRequest
from typing import List



load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')

model = OpenAIModel(llm)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
# logfire.configure(send_to_logfire='if-token-present')

logfire.configure(
    service_name="chatbot"    
)




@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

web_search_agent = Agent(
    model,
    system_prompt=f'You are a helpful assistant, and when needed you are also an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
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




agent = Agent('openai:gpt-4o')
THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)


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


async def get_db(request: Request) -> Database:
    return request.app.state.db



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, database: Database = Depends(get_db)):
    await websocket.accept()

    async with AsyncClient() as client:
        brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = Deps(client=client, brave_api_key=brave_api_key)
        async for data in websocket.iter_text():
            # get the chat history so far to pass as context to the agent
            messages = await database.get_messages()

            result = await web_search_agent.run(user_prompt=data, message_history=messages, deps=deps)                

            # add new messages (e.g. the user prompt and the agent response in this case) to the database
            await database.add_messages(result.new_messages_json())
            
            await websocket.send_text(result.data)
                
        


P = ParamSpec('P')
R = TypeVar('R')
#Dataclass for the database
@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file), check_same_thread=False)
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )