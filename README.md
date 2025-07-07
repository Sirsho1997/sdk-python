# Runware Model Context Protocol SDK

The Python Runware MCP SDK allows you to use Runware tools via a local agent interface, including prompt enhancement, image inference, background removal, and generating caption for images, with natural language support.


## Get API Access

To use the Python Runware SDK, you need to obtain an API key. Follow these steps to get API access:

1. [Create a free account](https://my.runware.ai/) with [Runware](https://runware.ai/).
2. Once you have created an account, you will receive an API key and trial credits.

**Important**: Please keep your API key private and do not share it with anyone. Treat it as a sensitive credential.

## Documentation

For detailed documentation and API reference, please visit the [Runware Documentation](https://docs.runware.ai/) or refer to the [docs](docs) folder in the repository. The documentation provides comprehensive information about the available classes, methods, and parameters, along with code examples to help you get started with the Runware SDK Python.

## Installation


- Ensure you have Python and all dependencies installed:
  ```bash
  pip install -r requirements_mcp.txt
  ```
- Set your API keys in a `.env` file in your project root (you need both a Runware API key and an OpenAI API key):
  ```env
  RUNWARE_API_KEY=your_runware_api_key
  OPENAI_API_KEY=your_openai_api_key
  LOGGING=T  # (optional) Set to 'T' to enable logging to debug.log
  ```

### Start the MCP Server and Client

Run:
```bash
python runware_mcp_client.py
```

This will launch an interactive chat interface. You can type natural language requests (e.g., "Remove the background from this image..." or "Generate a sunset image using model X") and the agent will use the appropriate Runware tool.

- Type `quit` to exit the chat.
- If logging is enabled, all interactions will be recorded in `debug.log`.

### Notes
- The client and server communicate via standard input/output (stdio).
- You can extend the server with new tools or modify existing ones in `runware_mcp_server.py`.


