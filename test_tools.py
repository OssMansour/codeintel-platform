from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def search_code(query: str, project_id: str = "") -> str:
    """Search the code repository for relevant code chunks.
    Args:
        query: Natural language search query
        project_id: Optional project ID to scope the search
    """
    return "test result"

m = ChatOllama(base_url="http://ollama:11434", model="qwen2.5-coder:7b-instruct-q4_K_M", num_ctx=4096)
m_tools = m.bind_tools([search_code])
resp = m_tools.invoke([HumanMessage(content="Search the code for main Python files")])
print("TYPE:", type(resp).__name__)
print("TOOL_CALLS:", resp.tool_calls)
print("CONTENT:", repr(resp.content[:300]))
print("ADDITIONAL_KWARGS:", resp.additional_kwargs)
