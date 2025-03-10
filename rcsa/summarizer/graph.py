from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

token_max = 4096
llm = ChatOpenAI(model="gpt-4o-mini")


class ConfigSchema(TypedDict):
    summary_prompt: str
    reduce_prompt: str
    final_prompt: str


class State(TypedDict):
    contents: list[str]
    summaries: list[str]
    final_summary: str


def summarize_each(state: State, config: RunnableConfig):
    # Create summary chain
    summary_prompt = config["configurable"].get(
        "summary_prompt", "為下面的文章產生一個簡短的摘要\n{content}"
    )
    summary_chain = (
        ChatPromptTemplate.from_template(summary_prompt) | llm | StrOutputParser()
    )

    # Summarize each content
    summaries = []
    for content in state["contents"]:
        summaries.append(summary_chain.invoke(input={"content": content}))
    return {"summaries": summaries}


def collapse_summaries(state: State, config: RunnableConfig):
    # Create reduce chain
    reduce_prompt = config["configurable"].get(
        "reduce_prompt", "摘要以下的多篇內容\n{concatenated_summaries}"
    )
    reduce_chain = (
        ChatPromptTemplate.from_template(reduce_prompt) | llm | StrOutputParser()
    )

    # merge summaries into texts whose length is less than 4096
    concatenated_summaries: list[str] = []
    current_list, current_length = [], 0
    for summary in state["summaries"]:
        length = llm.get_num_tokens(summary)
        if current_length + length > token_max:
            concatenated_summaries.append("\n".join(current_list))
            current_list, current_length = [summary], length
        else:
            current_list.append(summary)
            current_length += length
    if current_list:
        concatenated_summaries.append("\n".join(current_list))

    # Sumarrize concatenated summaries
    summaries = reduce_chain.batch(inputs=concatenated_summaries)

    return {"summaries": summaries}


def final_summarization(state: State, config: RunnableConfig):
    # Create final summary chain
    final_prompt = config["configurable"].get(
        "final_prompt", "總結以下的摘要\n{summaries}"
    )
    final_chain = (
        ChatPromptTemplate.from_template(final_prompt) | llm | StrOutputParser()
    )
    final_summary = final_chain.invoke(input="\n".join(state["summaries"]))
    return {"final_summary": final_summary}


def should_collapse(
    state: State,
) -> Literal["collapse_summaries", "final_summarization"]:
    num_tokens = sum(llm.get_num_tokens(summary) for summary in state["summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "final_summarization"


graph = StateGraph(State, ConfigSchema)
graph.add_node("summarize_each", summarize_each)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("final_summarization", final_summarization)
graph.add_edge(START, "summarize_each")
graph.add_conditional_edges("summarize_each", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("final_summarization", END)

app = graph.compile()
