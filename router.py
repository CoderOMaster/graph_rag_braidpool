from agno.agent import Agent
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.models.groq import Groq


def run_rag_db_agent(*args, **kwargs):
    # TODO: Implement vector semantic search
    return "Vector semantic search result"


def run_cypher_query_agent(*args, **kwargs):
    # TODO: Implement cypher query
    return "Cypher query result"


def create_team():
    """Create and configure the routing team with detailed instructions."""
    # Define the instructions as a multi-line string
    instructions = """You are an expert router that decides which agent is best suited to handle a user's query. Follow these guidelines:

1. Use the Vector DB Agent (semantic search) for:
   - Conceptual questions about programming (e.g., 'how to', 'what is', 'explain')
   - Error messages and exceptions
   - Requesting code examples or tutorials
   - Natural language queries about code functionality

   Example: "How do I parse JSON in Python?" -> Vector DB Agent

2. Use the Cypher Query Agent (graph database) for:
   - Questions about code structure and relationships
   - Dependency and import relationships
   - Function/method call hierarchies
   - Class inheritance and interface implementations

   Example: "What functions are called by main()?" -> Cypher Query Agent

3. If a query could be handled by both, prefer the Cypher Query Agent for structural accuracy.

4. If a query doesn't fit clearly, ask the user for clarification.

Always provide a brief explanation of your routing decision."""

    # Create the agents
    rag_agent = Agent(
        name="Vector DB Agent",
        role="""Handles natural language queries about code understanding, examples, and error explanations using vector semantic search. Examples: 
1. 'How do I implement a binary search in Python?'
2. 'Explain the concept of recursion with an example'
3. 'What does this error mean: 'NoneType' object has no attribute 'split'?'""",
        model=Groq(id="qwen/qwen3-32b"),
        tools=[run_rag_db_agent],
        show_tool_calls=True,
        debug_mode=True)

    cypher_agent = Agent(
        name="Cypher Query Agent",
        role="""Handles queries about code structure, relationships, and metadata using Cypher queries on a Neo4j graph database. Examples:
1. 'What are the dependencies of module X?'
2. 'Which functions call function Y?'
3. 'Show all classes that implement interface Z'""",
        model=Groq(id="qwen/qwen3-32b"),
        tools=[run_cypher_query_agent],
        show_tool_calls=True,
        debug_mode=True)

    # Create the team
    team = Team(
        model=Groq(id='qwen/qwen3-32b'),
        members=[rag_agent, cypher_agent],
        tools=[ReasoningTools(add_instructions=True)],
        enable_agentic_context=True,
        instructions=instructions,
        show_members_responses=True,
        mode='route'
    )
    return team


if __name__ == "__main__":
    team = create_team()
    query = "how to fix function not found error in python"
    team.print_response(
        query,
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )
