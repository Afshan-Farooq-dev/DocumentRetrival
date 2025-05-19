from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver





import os


llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)                                                                                                                                       




def add(a: int, b: int):
    """
    Add two numbers.

    Args:
        a (int):
        b (int):
    """ 
    return a + b

def subtract(a: int, b: int):
    """
    Subtract two numbers.

    Args:
        a (int):
        b (int):
    """
    return a - b

def multiply(a: int, b: int):
    """
    Multiply two numbers.

    Args:
        a (int):
        b (int):
    """
    return a * b


tools = [add, subtract, multiply]

llm_tools_bound = llm.bind_tools(tools)


def run_with_tools(state: MessagesState):
    sys_msg = SystemMessage(content="You are a helpful assistant that can perform basic arithmetic operations by using the different tools that bind with you.")
    # response = llm_tools_bound.invoke([sys_msg]  + state['messages']))

    return {"messages": [llm_tools_bound.invoke([sys_msg] + state['messages'])]}
    


graph = StateGraph(MessagesState)
graph.add_node("llm", run_with_tools)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "llm")


graph.add_conditional_edges("llm", tools_condition) #ouput llm , decide which tool to use,

graph.add_edge("tools", END)
graph.add_edge("llm", END)



app = graph.compile()


response = app.invoke({
    "messages": [HumanMessage(content="add 3+5")]

})

# print("LangGraph output:", response)


final_tool_message = response['messages'][-1]
print("Final Answer:", final_tool_message.content)


