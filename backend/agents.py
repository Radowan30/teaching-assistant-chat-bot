from __future__ import annotations as _annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire

from pydantic_ai import Agent, RunContext

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
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')

model = OpenAIModel(llm)

thinking_model = OpenAIModel('o4-mini')

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

        cleanup_manim_files()
        
        return {"video_url": video_url}
    
    except Exception as e:
        return {"video_url": "", "error": str(e)}