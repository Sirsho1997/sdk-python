import json
import os
from dotenv import load_dotenv
from runware import Runware, IPromptEnhance, RunwareAPIError, IImageInference
from mcp.server.fastmcp import FastMCP
from dataclasses import fields
import openai
import re


load_dotenv()
RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")
runware = Runware(api_key=RUNWARE_API_KEY)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize FastMCP server
mcp = FastMCP("runware_server")


@mcp.tool()
async def prompt_enhance(prompt: str, promptVersions: int = 1, promptMaxLength: int = 64, includeCost: bool = False):
    """
    Enhance a prompt using Runware's promptEnhance API.
    Args:
        prompt (str): The prompt to enhance.
        promptVersions (int): Number of prompt versions to generate.
        promptMaxLength (int): Maximum length of the enhanced prompt (in tokens).
        includeCost (bool): Whether to include cost in the response.
    Returns:
        List of enhanced prompt texts or an error message.
    """
    await runware.connect()
    prompt_enhancer = IPromptEnhance(
        prompt=prompt,
        promptVersions=promptVersions,
        promptMaxLength=promptMaxLength,
        includeCost=includeCost,
    )
    try:
        enhanced_prompts = await runware.promptEnhance(promptEnhancer=prompt_enhancer)
        return [ep.text for ep in enhanced_prompts]
    except RunwareAPIError as e:
        return {"error": str(e)}

@mcp.tool()
async def image_inference(params: dict):
    """
    Perform image inference using Runware's imageInference API.
    Args:
        params (dict): Dictionary of parameters matching IImageInference attributes.
    Returns:
        List of image URLs or an error message.
    """
    await runware.connect()
    # Filter params to only those accepted by IImageInference
    valid_fields = {f.name for f in fields(IImageInference)}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    try:
        request = IImageInference(**filtered)
        images = await runware.imageInference(requestImage=request)
        return [img.imageURL for img in images if hasattr(img, 'imageURL')]
    except RunwareAPIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def parse_image_inference_prompt(prompt: str) -> dict:
    """
    Parse a natural language prompt into IImageInference parameters using OpenAI GPT.
    The schema is extracted directly from the IImageInference docstring.
    """
    # Extract the schema from the docstring
    doc = IImageInference.__doc__
    schema_lines = []
    in_attr = False
    for line in doc.splitlines():
        if 'Attributes:' in line:
            in_attr = True
            continue
        if in_attr:
            if line.strip() == '' or not line.startswith(' ' * 8):
                break
            schema_lines.append(line.strip())
    # Join and clean up
    schema_str = '\n'.join(schema_lines)
    system_prompt = (
        f"You are an API parameter extraction assistant. "
        f"Given a user request, extract the following parameters for the image generation API:\n"
        f"{schema_str}\n"
        f"Return a JSON dictionary with only the relevant parameters."
    )
    user_prompt = f'User request: "{prompt}"\nOutput (as JSON):'

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = response['choices'][0]['message']['content']
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            params = json.loads(match.group(0))
            return params
        except Exception as e:
            return {"error": f"Failed to parse JSON: {e}", "raw": content}
    else:
        return {"error": "No JSON found in LLM response", "raw": content}

@mcp.tool()
async def parse_prompt_enhance(prompt: str) -> dict:
    """
    Parse a natural language prompt into IPromptEnhance parameters using OpenAI GPT.
    The schema is extracted directly from the IPromptEnhance docstring.
    """
    # Extract the schema from the docstring
    doc = IPromptEnhance.__doc__
    schema_lines = []
    in_attr = False
    for line in doc.splitlines():
        if 'Attributes:' in line:
            in_attr = True
            continue
        if in_attr:
            if line.strip() == '' or not line.startswith(' ' * 8):
                break
            schema_lines.append(line.strip())
    schema_str = '\n'.join(schema_lines)
    system_prompt = (
        f"You are an API parameter extraction assistant. "
        f"Given a user request, extract the following parameters for the prompt enhancement API:\n"
        f"{schema_str}\n"
        f"Return a JSON dictionary with only the relevant parameters."
    )
    user_prompt = f'User request: "{prompt}"\nOutput (as JSON):'

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=256,
    )
    content = response['choices'][0]['message']['content']
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            params = json.loads(match.group(0))
            return params
        except Exception as e:
            return {"error": f"Failed to parse JSON: {e}", "raw": content}
    else:
        return {"error": "No JSON found in LLM response", "raw": content}

# Starting the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
