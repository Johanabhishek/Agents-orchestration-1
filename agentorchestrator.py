"""
Multi-Agent Orchestrator — two agents communicating via shared state
Agent 1: SearchAgent — finds restaurants
Agent 2: OrderAgent  — places the order

Run: python orchestrator.py
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import operator

GROQ_API_KEY = "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"
"""FOURSQUARE_API_KEY = "F4EIWXC5UA31WELI43YCLFOCKYO3FRZUX2BAGZKRRB3ZFFCJ" """

# ── Shared State ──────────────────────────────────────────────────────────────
# This is the object that flows between agents
# Each agent reads from it and writes back to it
# operator.add means messages get APPENDED, not overwritten

class OrderState(TypedDict):
    messages: Annotated[list, operator.add]  # full conversation
    search_results: str                       # SearchAgent writes here
    order_result: str                         # OrderAgent writes here
    user_request: str                         # original user input

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_restaurants(query: str, location: str) -> str:
    """Search for restaurants by food type and location."""
    mock_data = [
        {"name": "Spice Garden",  "cuisine": "indian",  "location": "koramangala", "rating": 4.5, "eta": "30 mins"},
        {"name": "Biryani Bros",  "cuisine": "indian",  "location": "indiranagar", "rating": 4.7, "eta": "40 mins"},
        {"name": "Wok This Way",  "cuisine": "chinese", "location": "koramangala", "rating": 4.2, "eta": "25 mins"},
        {"name": "Burger Barn",   "cuisine": "american","location": "koramangala", "rating": 4.1, "eta": "20 mins"},
        {"name": "Dosa Delight",  "cuisine": "indian",  "location": "whitefield",  "rating": 4.6, "eta": "30 mins"},
    ]

    results = [
        r for r in mock_data
        if query.lower() in r["cuisine"] or query.lower() in r["name"].lower()
    ]
    if location:
        location_results = [r for r in results if location.lower() in r["location"]]
        if location_results:
            results = location_results

    if not results:
        results = mock_data  # return all if nothing matches

    output = "\n".join([
        f"- {r['name']} | {r['cuisine']} | {r['location']} | Rating: {r['rating']} | ETA: {r['eta']}"
        for r in results
    ])
    return f"Found {len(results)} restaurants:\n{output}"


@tool
def place_order(restaurant_name: str, item: str) -> str:
    """Place a mock order at a restaurant."""
    import random
    order_id = f"ORD{random.randint(10000, 99999)}"
    eta = f"{random.randint(25, 45)} mins"
    total = f"₹{random.randint(150, 500)}"
    return f"✅ Order placed! ID: {order_id} | Restaurant: {restaurant_name} | Item: {item} | Total: {total} | ETA: {eta}"

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

# ── Agent 1: SearchAgent ──────────────────────────────────────────────────────

search_agent = create_react_agent(
    llm,
    tools=[search_restaurants],
    prompt="""You are a restaurant search specialist.
Your ONLY job is to search for restaurants based on the user's request.
Call search_restaurants with the appropriate cuisine and location.
Return the results clearly with restaurant names, IDs, ratings and menus.
Do NOT place any orders."""
)

# ── Agent 2: OrderAgent ───────────────────────────────────────────────────────

order_agent = create_react_agent(
    llm,
    tools=[place_order],
    prompt="""You are an order placement specialist.
You will receive search results and the user's request.
Your ONLY job is to place the order using place_order tool.
Pick the best matching restaurant from the search results.
Pick the best matching item from the menu.
Always use the restaurant_id field, never the name."""
)

# ── Graph Nodes ───────────────────────────────────────────────────────────────
# Each node is a function that takes state, runs an agent, returns updated state

def run_search_agent(state: OrderState) -> OrderState:
    print("\n[SearchAgent] Searching for restaurants...")
    result = search_agent.invoke({
        "messages": [HumanMessage(content=state["user_request"])]
    })
    for msg in result["messages"]:
        print(f"  [{msg.type}]: {msg.content[:200]}")
    response = result["messages"][-1].content
    print(f"[SearchAgent] Done.")
    return {
        "messages": [AIMessage(content=f"SearchAgent: {response}")],
        "search_results": response,
        "order_result": "",
        "user_request": state["user_request"]
    }

def run_order_agent(state: OrderState) -> OrderState:
    print("\n[OrderAgent] Placing order...")
    combined = f"""
User wants: {state['user_request']}

Real restaurants found:
{state['search_results']}

Pick the best restaurant and place the order now.
"""
    result = order_agent.invoke({
        "messages": [HumanMessage(content=combined)]
    })
    response = result["messages"][-1].content
    print(f"[OrderAgent] Done.")
    return {
        "messages": [AIMessage(content=f"OrderAgent: {response}")],
        "search_results": state["search_results"],
        "order_result": response,
        "user_request": state["user_request"]
    }

def summarise(state: OrderState) -> OrderState:
    print("\n[Orchestrator] Summarising...")
    return {
        "messages": [AIMessage(content=state["order_result"])],
        "search_results": state["search_results"],
        "order_result": state["order_result"],
        "user_request": state["user_request"]
    }

# ── Build Graph ───────────────────────────────────────────────────────────────

graph = StateGraph(OrderState)
graph.add_node("search_agent", run_search_agent)
graph.add_node("order_agent", run_order_agent)
graph.add_node("summarise", summarise)

graph.set_entry_point("search_agent")
graph.add_edge("search_agent", "order_agent")
graph.add_edge("order_agent", "summarise")
graph.add_edge("summarise", END)

app = graph.compile()

# ── Run it ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n Multi-Agent Orchestrator")
    print("=" * 10)

    user_input = input("You: ").strip()

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "search_results": "",
        "order_result": "",
        "user_request": user_input
    }

    print("\n🔄 Orchestrator starting...\n")
    result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("FINAL RESULT:")
    print(result["order_result"])