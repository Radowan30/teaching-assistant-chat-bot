from __future__ import annotations as _annotations
import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import fastapi
import logfire
from fastapi import Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
import os
from datetime import datetime
from httpx import AsyncClient
from dotenv import load_dotenv
from manimDocs import manimDocs
from typing import Any
from agents import agent, Deps


# Load environment variables from .env file
load_dotenv()

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        _app.state.db = db
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
)

# To enable static files, we need to mount a directory to a specific path.
# This will allow the browser to access the files directly.
app.mount("/public", StaticFiles(directory=str(THIS_DIR / "public")), name="public")


@app.websocket('/async_chat')
async def async_chat(websocket: WebSocket):
    await websocket.accept()
    # Get the DB instance from the app state
    database: Database = websocket.app.state.db

    # Send previous chat messages to the client.
    previous_msgs = await database.get_messages()
    for m in previous_msgs:
        await websocket.send_text(json.dumps(to_chat_message(m)))
        
    while True:
        try:
            # Wait for a new prompt from the client.
            prompt = await websocket.receive_text()
            
            #Immediately send back the user prompt.
            await websocket.send_text(
                json.dumps({
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                })
            )
            
            async with AsyncClient() as client:
                messages = await database.get_messages()
                brave_api_key = os.getenv('BRAVE_API_KEY', None)
                deps = Deps(client=client, brave_api_key=brave_api_key)

                if prompt.strip().startswith("@video"):
                    result = await agent.run(prompt, message_history=messages, deps=deps)

                    content = str(result.data)

                    m = ModelResponse(parts=[TextPart(content)], timestamp=datetime.now(tz=timezone.utc))
                    await websocket.send_text(
                        json.dumps(to_chat_message(m))
                    )
                    await database.add_messages(result.new_messages_json())
                else:
                    # Stream messages from the agent.
                    async with agent.run_stream(prompt, message_history=messages, deps=deps) as result:
                        
                        async for text in result.stream(debounce_by=0.01):

                            m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                            await websocket.send_text(
                                json.dumps(to_chat_message(m))
                            )

                        await database.add_messages(result.new_messages_json())
    
        except Exception as e:
            error_msg = {
                'role': 'model',
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'content': f"Error: {str(e)}"
            }
            await websocket.send_text(json.dumps(error_msg))
            break

    await websocket.close()



async def get_db(request: Request) -> Database:
    return request.state.db

class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model', 'video']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            content = first_part.content
            # Try to parse content as JSON.
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "video_url" in data:
                    # Verify video file exists and is accessible
                    video_path = data["video_url"].split("/")[-1]
                    full_path = os.path.join(THIS_DIR, "public", "videos", "720p30", video_path)

                    if data["video_url"] and os.path.exists(full_path):  # Only if URL is not empty
                        return {
                            'role': 'video',
                            'timestamp': m.timestamp.isoformat(),
                            'content': data["video_url"],
                        }
                    elif "error" in data:  # Display error as text
                        return {
                            'role': 'model',
                            'timestamp': m.timestamp.isoformat(),
                            'content': f"Video generation failed: {data['error']}",
                        }
            except Exception:
                pass

            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': content,
            }
 

P = ParamSpec('P')
R = TypeVar('R')


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
        con = sqlite3.connect(str(file))
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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )