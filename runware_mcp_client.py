import asyncio
import os
import logging
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import ToolMessage, AIMessage

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure logging for client (using same file as server)
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug.log")

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create new handlers
file_handler = logging.FileHandler(log_file, mode='a')  # 'a' mode to append to existing file
console_handler = logging.StreamHandler()

# Set formatter
formatter = logging.Formatter('%(asctime)s [CLIENT] %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to root logger
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)

logging.info("Client started. Log file: %s", log_file)

# Initialize OpenAI model with key
model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# MCP server parameters (run runware_mcp.py)
server_params = StdioServerParameters(
    command="python",
    args=["runware_mcp_server.py"],
)

async def main():
    logging.info("Starting MCP client session")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            logging.info("Initializing session")
            await session.initialize()
            logging.info("Loading MCP tools")
            tools = await load_mcp_tools(session)
            logging.info(f"Loaded {len(tools)} tools")
            print("\n Available Tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description.split('.')[0]}")

            logging.info("Creating React agent")
            agent = create_react_agent(model, tools)

            print("\nType 'quit' to exit.")
            print("\n Chat started")
            logging.info("Chat session started")
            while True:
                print("\n __________")
                user_input = input("\nYou: ")
                if user_input.lower() == "quit":
                    logging.info("User requested to quit")
                    print("Exiting chat.")
                    break

                logging.info(f"User input: {user_input}")
                logging.info("Invoking agent with user input")
                try:
                    result = await agent.ainvoke({"messages": user_input})
                    logging.info("Agent response received successfully")
                except Exception as e:
                    logging.error(f"Agent invocation failed: {type(e).__name__}: {e}")
                    import traceback
                    logging.error(f"Agent error traceback: {traceback.format_exc()}")
                    print(f"Error: {e}")
                    continue

                final_response = None
                used_tools = set()

                logging.info(f"Processing agent result: {type(result)}")
                if isinstance(result, dict) and "messages" in result:
                    logging.info(f"Number of messages in result: {len(result['messages'])}")
                    for i, message in enumerate(result["messages"]):
                        logging.info(f"Message {i}: {type(message).__name__}")
                        if isinstance(message, ToolMessage):
                            used_tools.add(message.name)
                            logging.info(f"Tool message: {message.name}")
                            logging.info(f"Tool content: {message.content}")
                        elif isinstance(message, AIMessage):
                            tool_calls = message.additional_kwargs.get("tool_calls", [])
                            logging.info(f"Tool calls in message: {tool_calls}")
                            for call in tool_calls:
                                tool_name = call.get("function", {}).get("name")
                                used_tools.add(tool_name)
                                logging.info(f"Tool call: {tool_name}")
                                logging.info(f"Tool call args: {call.get('function', {}).get('arguments', {})}")

                    for message in reversed(result["messages"]):
                        if isinstance(message, AIMessage) and message.content:
                            final_response = message.content
                            logging.info(f"Final response: {final_response}")
                            break
                else:
                    logging.warning(f"Unexpected result structure: {result}")
                    logging.warning(f"Full result: {result}")

                if final_response:
                    print("\n🤖 Runware bot:")
                    print(final_response)
                else:
                    logging.warning("No final response generated")
                    print("Failed to generate response.")

                if used_tools:
                    logging.info(f"Tools used: {list(used_tools)}")
                    print("\nTools used in this response:")
                    for tool_name in used_tools:
                        print(f"- {tool_name}")
                else:
                    logging.info("No tools were used in this response")
                    print("\nNo preset tools were used in this response.")

if __name__ == "__main__":
    asyncio.run(main()) 