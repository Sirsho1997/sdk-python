import json
import os
from dotenv import load_dotenv
from runware import Runware, IPromptEnhance, RunwareAPIError, IImageInference, IImageBackgroundRemoval
from mcp.server.fastmcp import FastMCP
from dataclasses import fields
import re
import logging
import os
import traceback
from openai import OpenAI

load_dotenv()
RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")
runware = Runware(api_key=RUNWARE_API_KEY)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize FastMCP server
mcp = FastMCP("runware_server")

# Configure logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug.log")

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create new handlers
file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler()

# Set formatter
formatter = logging.Formatter('%(asctime)s [SERVER] %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to root logger
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)

logging.info(f"Server started. Log file: {log_file}")


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
        List of image URLs or an error message, and the request body for debugging.
    """
    logging.info(f"image_inference called with params: {params}")
    await runware.connect()
    valid_fields = {f.name for f in fields(IImageInference)}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    logging.info(f"Filtered params: {filtered}")
    try:
        request = IImageInference(**filtered)
        logging.info(f"Request: {request}")
        logging.info(f"Request details: positivePrompt='{request.positivePrompt}', model='{request.model}', height={request.height}, width={request.width}")
        images = await runware.imageInference(requestImage=request)
        result = [img.imageURL for img in images if hasattr(img, 'imageURL')]
        logging.info(f"Result: {result}")
        logging.info(f"Number of images returned: {len(images)}")
        for i, img in enumerate(images):
            logging.info(f"Image {i}: {img}")
        return {
            "request": str(request),
            "result": result
        }
    except RunwareAPIError as e:
        logging.error(f"RunwareAPIError: {e} | Request: {request}")
        return {"error": str(e), "request": str(request)}
    except TypeError as e:
        logging.error(f"TypeError creating IImageInference: {e}")
        logging.error(f"TypeError details: {type(e).__name__}: {str(e)}")
        logging.error(f"Filtered params: {filtered}")
        logging.error(f"Valid fields: {valid_fields}")
        
        logging.error(f"TypeError traceback: {traceback.format_exc()}")
        return {"error": f"TypeError: {e}", "params": filtered}
    except Exception as e:
        logging.error(f"Unexpected exception: {type(e).__name__}: {e}")
        logging.error(f"Request: {request if 'request' in locals() else 'Not created'}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "request": str(request) if 'request' in locals() else "Not created"}

@mcp.tool()
async def image_background_removal(params: dict):
    """
    Remove background from an image using Runware's imageBackgroundRemoval API.
    Args:
        params (dict): Dictionary of parameters matching IImageBackgroundRemoval attributes.
    Returns:
        List of image URLs or an error message, and the request body for debugging.
    """
    logging.info(f"image_background_removal called with params: {params}")
    await runware.connect()
    valid_fields = {f.name for f in fields(IImageBackgroundRemoval)}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    logging.info(f"Filtered params: {filtered}")
    try:
        request = IImageBackgroundRemoval(**filtered)
        logging.info(f"Request: {request}")
        logging.info(f"Request details: inputImage='{request.inputImage}', model='{request.model}', outputType='{request.outputType}'")
        images = await runware.imageBackgroundRemoval(removeImageBackgroundPayload=request)
        result = [img.imageURL for img in images if hasattr(img, 'imageURL')]
        logging.info(f"Result: {result}")
        logging.info(f"Number of images returned: {len(images)}")
        for i, img in enumerate(images):
            logging.info(f"Image {i}: {img}")
        return {
            "request": str(request),
            "result": result
        }
    except TypeError as e:
        logging.error(f"TypeError creating IImageBackgroundRemoval: {e}")
        logging.error(f"TypeError details: {type(e).__name__}: {str(e)}")
        logging.error(f"Filtered params: {filtered}")
        logging.error(f"Valid fields: {valid_fields}")
        import traceback
        logging.error(f"TypeError traceback: {traceback.format_exc()}")
        return {"error": f"TypeError: {e}", "params": filtered}
    except RunwareAPIError as e:
        logging.error(f"RunwareAPIError: {e} | Request: {request}")
        return {"error": str(e), "request": str(request)}
    except Exception as e:
        logging.error(f"Unexpected exception: {type(e).__name__}: {e}")
        logging.error(f"Request: {request if 'request' in locals() else 'Not created'}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "request": str(request) if 'request' in locals() else "Not created"}

@mcp.tool()
async def parse_image_inference_prompt(prompt: str) -> dict:
    """
    Parse a natural language prompt into IImageInference parameters using OpenAI GPT.
    The schema is extracted directly from the IImageInference docstring.
    """
    logging.info(f"parse_image_inference_prompt called with prompt: {prompt}")
    # Force flush
    for handler in logging.root.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
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
        f"IMPORTANT: You MUST include these REQUIRED parameters:\n"
        f"- positivePrompt: The text description of what to generate\n"
        f"- model: Use 'runware:1@1' as the default model\n"
        f"- height: Use 512 as default height\n"
        f"- width: Use 512 as default width\n"
        f"Return a JSON dictionary with the required parameters and any other relevant optional parameters."
    )
    user_prompt = f'User request: "{prompt}"\nOutput (as JSON):'


    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    content = response.choices[0].message.content
    logging.info(f"OpenAI response: {content}")
    logging.info(f"OpenAI response length: {len(content)} characters")
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            params = json.loads(match.group(0))
            logging.info(f"Parsed parameters: {params}")
            logging.info(f"Parameter types: {[(k, type(v)) for k, v in params.items()]}")
            return params
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
            logging.error(f"Failed content: {match.group(0)}")
            return {"error": f"JSON decode error: {e}", "raw": content}
        except Exception as e:
            logging.error(f"Unexpected error parsing JSON: {type(e).__name__}: {e}")
            logging.error(f"Failed content: {match.group(0)}")
            return {"error": f"Failed to parse JSON: {e}", "raw": content}
    else:
        logging.error(f"No JSON found in response: {content}")
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


    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content
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
async def parse_background_removal_prompt(prompt: str) -> dict:
    """
    Parse a natural language prompt into IImageBackgroundRemoval parameters using OpenAI GPT.
    The schema is extracted directly from the IImageBackgroundRemoval docstring.
    """
    logging.info(f"parse_background_removal_prompt called with prompt: {prompt}")
    # Force flush
    for handler in logging.root.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    # Extract the schema from the docstring
    doc = IImageBackgroundRemoval.__doc__
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
        f"Given a user request, extract the following parameters for the image background removal API:\n"
        f"{schema_str}\n"
        f"IMPORTANT: You MUST include these REQUIRED parameters:\n"
        f"- inputImage: The image URL, base64 data, or file path to remove background from\n"
        f"- model: Use 'runware:109@1' as the default background removal model (RemBG 1.4)\n"
        f"Return a JSON dictionary with the required parameters and any other relevant optional parameters."
    )
    user_prompt = f'User request: "{prompt}"\nOutput (as JSON):'

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content
    logging.info(f"OpenAI response: {content}")
    logging.info(f"OpenAI response length: {len(content)} characters")
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            params = json.loads(match.group(0))
            logging.info(f"Parsed parameters: {params}")
            logging.info(f"Parameter types: {[(k, type(v)) for k, v in params.items()]}")
            return params
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
            logging.error(f"Failed content: {match.group(0)}")
            return {"error": f"JSON decode error: {e}", "raw": content}
        except Exception as e:
            logging.error(f"Unexpected error parsing JSON: {type(e).__name__}: {e}")
            logging.error(f"Failed content: {match.group(0)}")
            return {"error": f"Failed to parse JSON: {e}", "raw": content}
    else:
        logging.error(f"No JSON found in response: {content}")
        return {"error": "No JSON found in LLM response", "raw": content}

# Starting the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
