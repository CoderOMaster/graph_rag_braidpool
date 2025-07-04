from agno.agent import Agent
from agno.models.groq import Groq


def create_rewriter_agent():
    return Agent(
        name="Query Rewriter",
        role="Takes ambiguous, grammatically incorrect, or shorthand user queries and rewrites them for clarity and precision",
        model=Groq(id="qwen/qwen3-32b"),
        show_tool_calls=True,
        debug_mode=True
    )


if __name__ == "__main__":
    rewriter = create_rewriter_agent()
    test_query = "how 2 fix err in pythn code getting none type errorrr?"
    # We run the agent with the test query and expect a rewritten query
    rewriter.print_response(f"Rewrite the following query to be more clear, grammatically correct, and unambiguous: {test_query}. AND STRICTLY DO NO PRINT ANYTHING ELSE OTHER THAN RE-WRITTEN QUERY")
    # response = response.content
    # # print(f"Original query: {test_query}")
    # print(f"Rewritten query: {response}")
