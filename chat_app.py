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
from typing import Annotated, Any, Callable, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
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
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
import uuid
import subprocess
from fastapi.staticfiles import StaticFiles
from manimDocs import manimDocs
from typing import Any, Dict, Union
import importlib.util
from manim import config

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')

model = OpenAIModel(llm)

thinking_model = OpenAIModel('o3-mini')

BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:8000')

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


#Creating the main agent
@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None


agent: Agent[None, Union[Dict[str, Any], str]] = Agent( model,
    system_prompt=f"""You are a helpful teaching assistant.

    When needed you can be an expert at researching the web to answer user questions using the search_web() tool. 

    You can also generate animation videos using the generate_manim_video() tool when the user passes a prompt that contains the keyword @video (do not invoke the generate_manim_video() tool unless this keyword is provided by the user), and when this tool is invoked you must return a Dict that only contains the key "video_url" with the URL to the generated video. Do not include any extra text with the Dict. It should be exactly of this format: 
    
    "video_url": "URL" 
    
    and no other text should be included in the response.  

    If the user wants a video or animation to be made and they do not mention the @video keyword, then tell them that they should pass a message that starts with the keyword, followed by the animation description. Example: @video make a circle going from left to right. 
    
    The current date is: {datetime.now().strftime("%Y-%m-%d")}""",
    deps_type=Deps,
    result_type=Union[Dict[str, Any], str],
    retries=4)


#Creating Manim Agent
class CodeResult(BaseModel):
    code: str = Field(
    description="The generated Manim code for the video."
    )

manim_agent = Agent(
    thinking_model,
    deps_type=str,
    result_type=CodeResult,
    retries=1,
    system_prompt="""You are an experienced teacher who is able to explain very well. You also know about Manim, which is a mathematical animation engine that is used to create videos programmatically using the python programming langauge. Your job is to generate Manim code that can be used to create an animation video that explains and teaches a concept very well to the audience.

The following is an example of the code:
\`\`\`
from manim import *
from math import *

class GenScene(Scene):
#your code here

\`\`\`

# Rules
1. Always use GenScene as the class name, otherwise, the code will not work.
2. Always use self.play() to play the animation, otherwise, the code will not work.
3. Do not use text to explain the code, only the code.
4. Do not explain the code, only the code.
5. The LaTex distribution used is MikTeX.
6. Make sure to create code so that it fits in a 16:9 aspect ratio.
7. Make sure your code is lengthy so that the animation that will be generated from your Manim code explains the concept very well to the audience.
8. Make sure there is text in the animation as if you are speaking to the audience. The text should fit with in the 16:9 aspect ratio and not overlap with other text.
9. Make sure to only use the Manim library for the code.
10. Think deeply about the animation and how it can explain the concept the best way possible.
11. Code should be compatible with manim==0.18.0 and manim-physics==0.4.0.

# Manim Library
{manimDocs}

"""
        )


#Web search tool
@agent.tool
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

#Manim video generation tool
@agent.tool
async def generate_manim_video(ctx: RunContext, prompt: str) -> Dict[str, Any]:

    try:
            

        manim_agent_result = await manim_agent.run(prompt)

        manim_code = manim_agent_result.data.code #The 'code' field is accessible from the data object

        file_name = "scene_temp"
        file_class = "GenScene"

        user_id = str(uuid.uuid4())
        project_name = "default_project"
        iteration = 1

        video_storage_file_name = f"video-{user_id}-{project_name}-{iteration}.mp4"

        frame_size, frame_width = (3840, 2160), 14.22 #16:9 aspect ratio

        modified_code = f"""
from manim import *
from math import *
config.frame_size = {frame_size}
config.frame_width = {frame_width}

{manim_code}
    """
        
        # Create a unique file name
        file_name = f"scene_{os.urandom(2).hex()}.py"

        # Adjust the path to point to /public/
        root_dir = os.path.dirname(__file__)  # Go up one level
        public_dir = os.path.join(root_dir, "public")
        os.makedirs(public_dir, exist_ok=True)  # Ensure the public directory exists
        file_path = os.path.join(public_dir, file_name)

        # Write the code to the file
        with open(file_path, "w") as f:
            f.write(modified_code)

    
        # video_file_path = os.path.join(public_dir, video_storage_file_name)

        # Video generation logic using dynamic import and programmatic rendering
        spec = importlib.util.spec_from_file_location("temp_scene", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SceneClass = getattr(module, file_class)
        
        # Ensure Manim outputs to the public directory with the desired filename
        config.output_file = video_storage_file_name
        config.media_dir = "public"
        config.custom_folders = False  # Disable folder nesting so the video goes directly into "public"
        
        scene = SceneClass()
        scene.render()

        video_url = f"{BASE_URL.rstrip('/')}/public/videos/2160p60/{video_storage_file_name}"

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")

        return {"video_url": video_url}

        # try:
        #     command_list = [
        #         "python",
        #         "-m",
        #         "manim",
        #         file_path,  # Use the full path to the file
        #         file_class,
        #         "--format=mp4",
        #         "--output_file", video_storage_file_name,
        #         "--media_dir",
        #         "public",
        #         "--custom_folders",
        #     ]

        #     process = subprocess.Popen(
        #         command_list,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         cwd=os.path.dirname(os.path.realpath(__file__)),
        #         text=True,
        #         bufsize=1,  # Ensure the output is in text mode and line-buffered
        #     )

        #     stdout, stderr = process.communicate()

        #     current_animation = -1
        #     current_percentage = 0
        #     error_output = []
        #     in_error = False


        #     if process.returncode == 0:
               
                
        #         video_url = f"{BASE_URL.rstrip('/')}/public/{video_storage_file_name}"

        #         return {"video_url": video_url}

        #     else:
        #         return {"video_url": "", "error": f"Error generating video: {stderr}"}
            
        # except Exception as e:
        #     return {"video_url": "", "error": str(e)}
        
        # finally:
        #     # Remove the temporary Python file
        #     try:
        #         if os.path.exists(file_path) and os.path.exists(video_file_path):
        #             os.remove(file_path)
        #             print(f"Removed temporary file: {file_path}")
        #     except Exception as e:
        #         print(f"Error removing temporary file {file_path}: {e}")


    except Exception as e:
        return {"video_url": "", "error": str(e)}
        


THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)

# To enable static files, we need to mount a directory to a specific path.
# This will allow the browser to access the files directly.
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model', 'video', 'progress']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    if isinstance(m, ModelRequest):
        # Look for a user prompt in the parts.
        for part in m.parts:
            if isinstance(part, UserPromptPart):
                return {
                    'role': 'user',
                    'timestamp': part.timestamp.isoformat(),
                    'content': part.content,
                }
        # Fallback if no explicit user prompt is found.
        return {
            'role': 'user',
            'timestamp': m.parts[0].timestamp.isoformat(),
            'content': m.parts[0].content,
        }
    elif isinstance(m, ModelResponse):
        first_part = m.parts[0]
        if isinstance(first_part, TextPart):
            content = first_part.content.strip()
            # Try to parse content as JSON.
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "video_url" in data:
                    if data["video_url"]:  # Only if URL is not empty
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

            # Check if the content is a plain video URL.
            # Use lower() and strip() to catch extra spaces or different cases.
            if content.lower().startswith("http") and content.lower().endswith(".mp4"):
                return {
                    'role': 'video',
                    'timestamp': m.timestamp.isoformat(),
                    'content': content,
                }
            
            # Otherwise, treat as a normal text response.
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        # Stream the user prompt immediately.
        yield (
            json.dumps({
                'role': 'user',
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'content': prompt,
            }).encode('utf-8')
            + b'\n'
        )

        async with AsyncClient() as client:
            messages = await database.get_messages()
            brave_api_key = os.getenv('BRAVE_API_KEY', None)
            deps = Deps(client=client, brave_api_key=brave_api_key)


            # Stream messages normally.
            async with agent.run_stream(prompt, message_history=messages, deps=deps) as result:
                async for text in result.stream(debounce_by=0.01):
                    m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                    yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

            
            await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


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