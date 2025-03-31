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
from fastapi import Depends, Request, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
import os
from datetime import datetime
from httpx import AsyncClient
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
import uuid
from manimDocs import manimDocs
from typing import Any, Dict, Union
import importlib.util
from manim import config
import shutil

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

    Important note: Give all your answers in markdown format, EXCEPT for the video URL, which should be returned as a Dict.
    
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
    retries=2,
    system_prompt=r"""You are an experienced teacher who is able to explain very well. You also know about Manim, which is a mathematical animation engine that is used to create videos programmatically using the python programming langauge. Your job is to generate Manim code that can be used to create an animation video that explains and teaches a concept very well to the audience.

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


def cleanup_manim_files():
    """
    Cleanup function that:
    1. Deletes all files starting with 'scene' in the public folder
    2. Deletes all folders in public except 'videos'
    3. Deletes the 'partial_movie_files' folder in videos/2160p60/
    """
    # Get the parent folder path
    root_dir = os.path.dirname(__file__)
    public_dir = os.path.join(root_dir, "public")
    
    if not os.path.exists(public_dir):
        print(f"Public directory does not exist: {public_dir}")
        return
    
    # Step 1: Delete files starting with 'scene'
    for item in os.listdir(public_dir):
        item_path = os.path.join(public_dir, item)
        if os.path.isfile(item_path) and item.startswith('scene'):
            try:
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            except Exception as e:
                print(f"Error deleting file {item_path}: {e}")
    
    # Step 2: Delete all folders except 'videos'
    for item in os.listdir(public_dir):
        item_path = os.path.join(public_dir, item)
        if os.path.isdir(item_path) and item != 'videos':
            try:
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
            except Exception as e:
                print(f"Error deleting directory {item_path}: {e}")
    
    # Step 3: Delete partial_movie_files in videos/2160p60/
    videos_2160p60_dir = os.path.join(public_dir, "videos", "2160p60")
    partial_movie_files_dir = os.path.join(videos_2160p60_dir, "partial_movie_files")
    
    if os.path.exists(partial_movie_files_dir):
        try:
            shutil.rmtree(partial_movie_files_dir)
            print(f"Deleted partial_movie_files directory: {partial_movie_files_dir}")
        except Exception as e:
            print(f"Error deleting partial_movie_files directory: {e}")


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

        # try:
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #         print(f"Removed temporary file: {file_path}")
        # except Exception as e:
        #     print(f"Error removing temporary file {file_path}: {e}")

        cleanup_manim_files()
        
        return {"video_url": video_url}
    
    except Exception as e:
        return {"video_url": "", "error": str(e)}
        


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
            
            # Immediately send back the user prompt.
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
    
                # Stream messages from the agent.
                async with agent.run_stream(prompt, message_history=messages, deps=deps) as result:

                    video_json_accumulator = ""
                    collecting_video_json = False
                    already_sent_video = False 
                    
                    async for text in result.stream(debounce_by=0.01):
                        # Start collecting JSON when we detect potential video JSON
                        if text.strip().startswith("{") and not collecting_video_json:
                            collecting_video_json = True
                            video_json_accumulator = text
                            continue
                        
                        # Keep accumulating JSON parts
                        if collecting_video_json:
                            video_json_accumulator = text
                            
                            # If we have complete JSON (ending with closing brace)
                            if "}" in text:
                                try:
                                    # Try to parse the complete JSON
                                    video_data = json.loads(video_json_accumulator)
                                    
                                    # Check if it's a video URL
                                    if isinstance(video_data, dict) and "video_url" in video_data:
                                        # Verify video file exists and is accessible
                                        video_path = video_data["video_url"].split("/")[-1]
                                        full_path = os.path.join(THIS_DIR, "public", "videos", "2160p60", video_path)
                                        
                                        # Check if file exists
                                        if os.path.exists(full_path):
                                            # Send video message to client
                                            await websocket.send_text(
                                                json.dumps({
                                                    'role': 'video',
                                                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                                                    'content': video_data["video_url"],
                                                })
                                            )
                                            already_sent_video = True
                                            collecting_video_json = False
                                            video_json_accumulator = ""
                                            print("Sent a json with role video", flush=True)
                                            continue
                                
                                except json.JSONDecodeError:
                                    # Not valid JSON yet, continue collecting
                                    pass
                        
                        # For regular text content (not part of video JSON)
                        if not collecting_video_json:
                            m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                            await websocket.send_text(
                                json.dumps(to_chat_message(m))
                            )
                            print("Sent a message with role model", flush=True)
                    

                    # Handle any remaining collected JSON that wasn't processed
                    if collecting_video_json and video_json_accumulator:
                        try:
                            video_data = json.loads(video_json_accumulator)
                            if isinstance(video_data, dict) and "video_url" in video_data:
                                video_path = video_data["video_url"].split("/")[-1]
                                full_path = os.path.join(THIS_DIR, "public", "videos", "2160p60", video_path)
                                
                                if os.path.exists(full_path) and not already_sent_video:
                                    await websocket.send_text(
                                        json.dumps({
                                            'role': 'video',
                                            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                                            'content': video_data["video_url"],
                                        })
                                    )
                                    print("Sent a json with role video in the second branch", flush=True)
                        except:
                            # If we can't parse it as video JSON, send as regular text
                            m = ModelResponse(parts=[TextPart(video_json_accumulator)], timestamp=result.timestamp())
                            await websocket.send_text(
                                json.dumps(to_chat_message(m))
                            )

                    await database.add_messages(result.new_messages_json())



                    # fullVideoJson = '' #to accumulate the video json response so that it is not sent in chunks
                    # async for text in result.stream(debounce_by=0.01):

                        # if text.strip().startswith("{") and (".mp4}" not in text): 
                        #     # Just accumulate potential video JSON
                        #     # fullVideoJson += text
                        #     continue
                            
                        # elif ".mp4}" in text:
                        #     # Complete video JSON found
                        #     # fullVideoJson += text
                        #     fullVideoJson = text
                            
                        #     try:
                        #         # Parse the complete JSON to ensure it's valid
                        #         video_data = json.loads(fullVideoJson)
                        #         if isinstance(video_data, dict) and "video_url" in video_data:
                        #             # Send direct message with video role
                        #             await websocket.send_text(
                        #                 json.dumps({
                        #                     'role': 'video',
                        #                     'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                        #                     'content': video_data["video_url"],
                        #                 })
                        #             )
                        #             # Reset for next potential video
                        #             fullVideoJson = ''
                        #             continue
                        #     except json.JSONDecodeError:
                        #         # If not valid JSON, treat as regular text
                        #         pass
                                
                        #     # If we get here, reset and handle as regular text
                        #     fullVideoJson = ''

                        #     # Handle regular text responses
                        #     m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                        #     await websocket.send_text(
                        #         json.dumps(to_chat_message(m))
                        #     )


                        # m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())

                        # if text.strip().startswith("{") and (".mp4}" not in text): #Keep getting 'text' until .mp4} substring is found
                        #     continue

                        # elif ".mp4}" in text:
                        #     fullVideoJson = text

                        #     # video_message = ModelResponse(parts=[TextPart(fullVideoJson)], timestamp=result.timestamp())
                        #     # await websocket.send_text(
                        #     #     json.dumps(to_chat_message(video_message))
                        #     # )

                        #     try:
                        #         video_data = json.loads(fullVideoJson)
                        #         if isinstance(video_data, dict) and "video_url" in video_data:
                        #             await websocket.send_text(
                        #                 json.dumps(
                        #                     {
                        #                         'role': 'video',
                        #                         'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                        #                         'content': video_data['video_url'],
                        #                     }
                        #                 )
                        #             ) 


                        #         fullVideoJson = ''

                        #         break

                        #     except json.JSONDecodeError:
                        #         pass
                        # else:
                        #     await websocket.send_text(
                        #         json.dumps(to_chat_message(m))
                        #     )



                        # new_chunk = text[len(fullVideoJson):] if text.startswith(fullVideoJson) else text #Get the new part of text and put it in new_chunk
                        # fullVideoJson += new_chunk

                        # if fullVideoJson.strip().startswith("{") and (".mp4}" not in fullVideoJson): #Keep getting 'text' until .mp4} substring is found
                        #     continue
                        
                        # elif ".mp4}" in fullVideoJson:
                        #     video_message = ModelResponse(parts=[TextPart(fullVideoJson)], timestamp=result.timestamp())
                        #     await websocket.send_text(
                        #         json.dumps(to_chat_message(video_message))
                        #     )

                        #     fullVideoJson = ''
                            
                        # else:
                        #     m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())

                        #     await websocket.send_text(
                        #         json.dumps(to_chat_message(m))
                        #     )
    
                await database.add_messages(result.new_messages_json())
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break

    await websocket.close()



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

    role: Literal['user', 'model', 'video']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        # # Look for a user prompt in the parts.
        # for part in m.parts:
        #     if isinstance(part, UserPromptPart):
        #         return {
        #             'role': 'user',
        #             'timestamp': part.timestamp.isoformat(),
        #             'content': part.content,
        #         }
        # # Fallback if no explicit user prompt is found.
        # return {
        #     'role': 'user',
        #     'timestamp': m.parts[0].timestamp.isoformat(),
        #     'content': m.parts[0].content,
        # }
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
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