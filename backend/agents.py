from __future__ import annotations as _annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Annotated, Literal
import asyncio
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

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

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

    You can also generate animation videos using the generate_manim_video() tool when the user passes a prompt that contains the keyword @video (always invoke the generate_manim_video() tool when this keyword is provided by the user), and when this tool is invoked you must return a Dict that only contains the key "video_url" with the URL to the generated video. Do not include any extra text with the Dict. It should be exactly of this format: 
    
    "video_url": "URL" 
    
    and no other text should be included in the response.  

    If the user wants a video or animation to be made and they do not mention the @video keyword, then tell them that they should pass a message that starts with the keyword, followed by the animation description. Example: @video explain photosynthesis like I'm 5. 

    Important note: Give all your answers in markdown format, EXCEPT for the video URL, which should be returned as a Dict.
    
    The current date is: {datetime.now().strftime("%Y-%m-%d")}""",
    deps_type=Deps,
    result_type=Union[Dict[str, Any], str],
    retries=4)


# Creating the Video Script Agent
script_agent = Agent[str](
    thinking_model,
    result_type= str,
    retries=2,
    system_prompt=r"""You are an experienced teacher who is able to explain very well and intuitively. Your job is to generate a script based on the user's prompt that can be used to create python code which generates an animation video. This video should explain and teach the concept very well to the audience. 

    Rules:
    1. You must make the script into logical scenes, all of which should come together to give a very intuitive understanding of the concept to the audience.
    2. Make sure there is narration text in the script as if you are speaking to the audience while explaining the concepts.
    3. The video to be generated doesn't have any audio, so do not mention anything related to audio in the script.
    4. Make sure you describe what visuals such as shapes or illustrations that should be present in each scene and how they must move, morph or animate, and how they are colored.
    5. Make sure you mention exactly what text should be displayed in each scene of the video.
    6. Make sure there isn't too much text in each scene, so that the audience can easily read and understand it.
    7. Remember that the video can't have any images, only shapes, illustrations, designs and text made from the python code.
    8. The video only has a black background, so do not mention anything related to background color in the script and make the color of everything else on the lighter side so that they are visible.
    """
)

#Creating the Coding Agent
class CodeResult(BaseModel):
    code: str = Field(
    description="The generated Manim code for the video."
    )

coding_agent = Agent[CodeResult](
    model,
    result_type=CodeResult,
    retries=1,
    system_prompt=r"""You are an experienced teacher who is able to explain very well and intuitively. You also know about Manim, which is a mathematical animation engine that is used to create videos programmatically using the python programming langauge. 
    
    Your job is to generate Manim code based a video script given to you. The code that you will generate can be used to create an animation video that explains and teaches a concept very well to the audience.

    You can also edit an existing code that is provided to you based on feedback and the video script (which was followed to make the code) to make the code better.

The following is an example of the code:
\`\`\`
from manim import *
from math import *

class GenScene(Scene):
#your code here

\`\`\`

Rules:
1. Always use GenScene as the class name, otherwise, the code will not work.
2. Always use self.play() to play the animation, otherwise, the code will not work.
3. Do not use text to explain the code, only the code.
4. Do not explain the code, only the code.
5. The LaTex distribution used is MikTeX.
6. Make sure to create code so that the generated content (everything, including text) fits in a 16:9 aspect ratio and 720p resolution (You DO NOT have to set the frame or camera to 16:9 aspect ratio and 720p resolution, that part will be added later by another person).
7. Make sure there is text in the animation as if you are speaking to the audience.
8. Always fade out previous narration text before showing the next text. 
9. Non-narration words must be far apart from each other.
10. Make sure to only use the Manim library for the code.
11. Code MUST be compatible with manim version 0.18.0 and manim-physics version 0.4.0, and all parameters, methods etc must be from manim version 0.18.0 and manim-physics version 0.4.0. This is a strict rule, so do not break it. Code using newer version of manim will not work. 
12. Do not make use of any pictures or svgs of any kind, only use shapes and designs that can be made using Manim.
13. Make sure to use raw string r"" for LaTeX expressions. Example: Do MathTex("F", "=", "m", r"\times", "a", ...) instead of MathTex("F", "=", "m", \times, "a", ...)
14. Valid Manim color constants include: RED, GREEN, BLUE, WHITE, BLACK, GRAY, YELLOW, GOLD, PINK, 
PURPLE, ORANGE, TEAL, LIGHT_BROWN. If you need other colors like BROWN, you must define them 
explicitly using hex codes (e.g., BROWN = "#8B4513") at the top of the file.
15. For chemical formulas or any mathematical notation with subscripts/superscripts, always use MathTex instead of Tex. Example: MathTex(r"H_2O") instead of Tex(r"H_2O")


# Manim Library
{manimDocs}

"""
        )


# creating the Code Reviewer Agent
@dataclass
class CodeReviewDeps:
    code: str
    video_script: str
    is_code_OK: bool 

class CodeReviewResult(BaseModel):
    feedback: str | None = Field("The constructive feedback on the code given to you.")
    is_code_OK: bool = Field("Whether the code is good or not. True if the code is good and meets the specified conditions, False otherwise.")

code_reviewer_agent = Agent[CodeReviewResult](
    model,
    deps_type=CodeReviewDeps,
    result_type=CodeReviewResult,
    # retries=0,
    system_prompt="""You are an experienced teacher who is able to explain very well and intuitively. You are also an expert in reviewing code and providing constructive feedback. A video is to be made from the Manim code provided to you. Your job is to review the Manim code and provide feedback on it based on the the code acceptance criteria mentioned below. If the code meets all the acceptance criteria, then set is_code_OK to True and there is no need to provide feedback (feedback is set to None). Otherwise, if the code does not meet all the criteria, set is_code_OK to False and provide a detailed feedback on exactly what needs to be changed in the code to meet all the criteria.
    
Code Acceptance Criteria:
5. Code properly follows the video script provided to you.
6. Code doesn't have too much text in a scene, so that the screen doesn't look crowded with text.
7. Code doesn't have text that overlaps with other text in the animation.
8. The text and animations in the video are logical and follows what the script mentions.
9. The shapes, and animations are formed how the script specifies.
10. Code doesn't have any syntax errors or runtime errors when executed.
11. Make sure raw string r"" is used for LaTeX expressions. For example: it is like MathTex("F", "=", "m", r"\times", "a", ...) instead of MathTex("F", "=", "m", \times, "a", ...).
11. Code is very likely to compile correctly and generate a video.
12. Code MUST be compatible with manim version 0.18.0 and manim-physics version 0.4.0, and all parameters, methods etc must be from manim version 0.18.0 and manim-physics version 0.4.0. This is a strict rule, that should not be broken. Code using newer version of manim will not work. 
13. Valid Manim color constants include: RED, GREEN, BLUE, WHITE, BLACK, GRAY, YELLOW, GOLD, PINK, 
PURPLE, ORANGE, TEAL, LIGHT_BROWN. If the code needs other colors like BROWN, it must define them 
explicitly using hex codes (e.g., BROWN = "#8B4513") at the top of the file.
14. For chemical formulas or any mathematical notation with subscripts/superscripts, always use MathTex instead of Tex. Example: MathTex(r"H_2O") instead of Tex(r"H_2O")
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
    videos_720p30_dir = os.path.join(public_dir, "videos", "720p30")
    partial_movie_files_dir = os.path.join(videos_720p30_dir, "partial_movie_files")
    
    if os.path.exists(partial_movie_files_dir):
        try:
            shutil.rmtree(partial_movie_files_dir)
            print(f"Deleted partial_movie_files directory: {partial_movie_files_dir}")
        except Exception as e:
            print(f"Error deleting partial_movie_files directory: {e}")



 # LangGraph graph for code Generation and Review 
def create_graph():    
    class State(TypedDict):
        video_script: str
        code_for_video: str | None
        feedback: str | None
        is_code_OK: bool
        counter: int
        next_node: str | None

    async def generate_code(state: State):
        video_script = state.get("video_script")
        previous_code = state.get("code_for_video")
        feedback = state.get("feedback")

        if previous_code == None or feedback == None:
            coding_agent_result = await coding_agent.run(
                f"Generate Manim code for the video script provided: {video_script}",
                )
        else:
            coding_agent_result = await coding_agent.run(
                f"""Edit the Manim code provided based on the feedback and the video script that was used. 
                
                Manim code: {previous_code},
                Feedback: {feedback},
                Video script: {video_script}""",
                )
        
        return {"code_for_video": coding_agent_result.data.code}

    async def review_code(state: State):
        codeReviewDeps = CodeReviewDeps(
            code = state.get("code_for_video"),
            video_script = state.get("video_script"),
            is_code_OK = state.get("is_code_OK")
        )
        code_reviewer_result = await code_reviewer_agent.run(
            "Review the Manim code for the video script",
            deps=codeReviewDeps
        )
        incremented_counter = state.get("counter") + 1
        return {"feedback": code_reviewer_result.data.feedback, "is_code_OK": code_reviewer_result.data.is_code_OK, "counter": incremented_counter}


    def router(state: State)-> Command[Literal["coding_agent", END]]:
        is_code_OK = state.get("is_code_OK")
        counter = state.get("counter")
        if is_code_OK==False and counter <= 2:
            goto = "coding_agent"
        else:
            goto = END
        return Command(
            goto=goto,
        )

    graph_builder = StateGraph(State)

    graph_builder.add_node("coding_agent", generate_code)
    graph_builder.add_node("code_reviewer_agent", review_code)
    graph_builder.add_node("router", router)

    graph_builder.add_edge(START, "coding_agent")
    graph_builder.add_edge("coding_agent", "code_reviewer_agent")
    graph_builder.add_edge("code_reviewer_agent", "router")

    return graph_builder.compile()


#Video generation tool
@agent.tool
async def generate_manim_video(ctx: RunContext, prompt: str) -> Dict[str, Any]:

    try:
            
        # Generate the script for the video
        print(f"Generating video script...", flush=True)
        script_agent_result = await script_agent.run(prompt)
        video_script = script_agent_result.data

        print(f"Video script has been generated.", flush=True) 

        print("Creating LangGraph graph...", flush=True)
        graph = create_graph()

        # Pass the Video script to the langGraph graph for code generation and review
        initial_state = {"video_script": video_script, "code_for_video": None, "feedback": None, "is_code_OK": False, "counter": 0, "next_node": None}

        final_state = await graph.ainvoke(initial_state)
        print("Graph created and invoked.", flush=True)

        print("\nManim code generated.", flush=True)
        manim_code = final_state.get("code_for_video")

        print(f"Manim code: {manim_code}", flush=True)
        
        if not manim_code:
            return {"video_url": "", "error": "Failed to generate valid code"}

        file_name = "scene_temp"
        file_class = "GenScene"

        user_id = str(uuid.uuid4())
        project_name = "default_project"
        iteration = 1

        video_storage_file_name = f"video-{user_id}-{project_name}-{iteration}.mp4"

        frame_size = (1280, 720)  # 720p resolution
        frame_width = 14.22 #16:9 aspect ratio

        modified_code = f"""
from manim import *
from math import *
config.pixel_width = {frame_size[0]}
config.pixel_height = {frame_size[1]}
config.frame_width = {frame_width}

{manim_code}
    """
        
        # Create a unique file name
        file_name = f"scene_{os.urandom(2).hex()}.py"

        # Adjust the path to point to /public/
        root_dir = os.path.dirname(__file__)  # Go up one level
        public_dir = os.path.join(root_dir, "public")
        os.makedirs(public_dir, exist_ok=True)  # Ensure the public directory exists
        
        # Make sure the output directory structure exists
        videos_dir = os.path.join(public_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        videos_1080p_dir = os.path.join(videos_dir, "1080p60")
        os.makedirs(videos_1080p_dir, exist_ok=True)
        
        file_path = os.path.join(public_dir, file_name)

        # Write the code to the file
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(modified_code)

        # Give the filesystem a moment to sync
        await asyncio.sleep(0.1)

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
        config.preview = False
        config.quality = "medium_quality"  # Use medium quality
        
        try:
            scene = SceneClass()
            print("\nRendering video (this may take time)...", flush=True)
            scene.render()
            print("Video render completed.", flush=True)
        except Exception as e:
            print(f"Error during rendering: {str(e)}", flush=True)
            return {"video_url": "", "error": f"Video render error: {str(e)}"}

         # Check if video exists
        video_path = os.path.join(public_dir, "videos", "720p30", video_storage_file_name)
        if not os.path.exists(video_path):
            print(f"Video file was not created at expected path: {video_path}", flush=True)
            # Try to find it elsewhere
            for root, _, files in os.walk(public_dir):
                for file in files:
                    if file == video_storage_file_name:
                        video_path = os.path.join(root, file)
                        print(f"Found video at: {video_path}", flush=True)
                        break

        video_url = f"{BASE_URL.rstrip('/')}/public/videos/720p30/{video_storage_file_name}"

        cleanup_manim_files()
        
        return {"video_url": video_url}
    
    except Exception as e:
        return {"video_url": "", "error": str(e)}