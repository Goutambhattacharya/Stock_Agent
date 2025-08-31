import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Mock Classes for Testing ---
class MockModel:
    """A dummy model used for testing without real API calls."""
    def run(self, query: str) -> dict:
        return {"message": f"Mock response for query: {query}"}

class MockTools:
    """Dummy tools placeholder."""
    pass


def load_environment() -> dict:
    """
    Load raw environment variables (API keys, tokens, zones).
    Returns a dict so other modules can use them.
    """
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "BRIGHT_DATA_API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser"),
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "openai"),  # default = openai
    }


def load_model_and_tools(use_mock: bool = False):
    """
    Load the model and tools.
    If use_mock=True, returns mock versions (for testing).
    Otherwise, loads real model and tools using environment variables.
    """
    env = load_environment()

    if use_mock:
        print("⚠️ Using MockModel (no real API calls).")
        return MockModel(), MockTools()

    provider = env["LLM_PROVIDER"]

    # === Using OpenAI ===
    if provider == "openai":
        api_key = env["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        tools = []  # Later: add Bright Data tools here
        return model, tools

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
