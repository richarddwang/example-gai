from typing import Literal, TypedDict, final
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import split_list_of_docs, collapse_docs
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

token_max = 4096
llm = ChatOpenAI(name="gpt-4o-mini")

summary_chain = (
    ChatPromptTemplate.from_template("為下面的文章產生一個簡短的摘要\n{content}")
    | llm
    | StrOutputParser()
)
reduce_chain = (
    ChatPromptTemplate.from_template("摘要以下的多篇內容\n{concatenated_summaries}")
    | llm
    | StrOutputParser()
)


class State(TypedDict):
    contents: list[str]
    summaries: list[str]
    final_summary: str


def summarize_docs(state: State):
    summaries = []
    for content in state["contents"]:
        summaries.append(summary_chain.invoke(input={'content': content}))
    return {"summaries": summaries}


def collapse_summaries(state: State):
    # merge summaries into texts whose length is less than 4096
    concatenated_summaries: list[str] = []
    current_list, current_length = [], 0
    for summary in state["summaries"]:
        length = llm.get_num_tokens(summary)
        if current_length + length > token_max:
            concatenated_summaries.append('\n'.join(current_list))
            current_list, current_length = [summary], length
        else:
            current_list.append(summary)
            current_length += length
    if current_list:
        concatenated_summaries.append('\n'.join(current_list))
    
    # Sumarrize concatenated summaries
    summaries = reduce_chain.batch(inputs=concatenated_summaries)

    return {"summaries": summaries}

def generate_final_summary(state: State):
    final_summary = reduce_chain.invoke(input='\n'.join(state["summaries"]))
    return {"final_summary": final_summary}
    
graph = StateGraph(State)
graph.add_node('summary_docs', summarize_docs)
graph.add_node('collapse_summaries', collapse_summaries)
graph.add_node('generate_final_summary', generate_final_summary)

def should_collapse(
    state: State
)->Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = sum(llm.get_num_tokens(summary) for summary in state["summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"

graph.add_edge(START, 'summary_docs')
graph.add_conditional_edges('summary_docs', should_collapse)
graph.add_conditional_edges('collapse_summaries', should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()