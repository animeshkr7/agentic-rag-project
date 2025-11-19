
import os
import logging
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define the state for the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    intent: str
    company_name: str

# Define the structured output for the router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    intent: str = Field(
        ...,
        description="Given the user query, classify it as either 'rag' for questions that require information from a document, or 'general' for all other questions.",
    )
    company_name: str = Field(
        ...,
        description="The name of the company mentioned in the user's query. If no company is mentioned, default to 'none'.",
    )

# Create the agent
class Agent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.vectorstore = FAISS.load_local(
            "vectorstore/db_faiss",
            HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'}),
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vectorstore.as_retriever()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self.router)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("chat", self.chat_node)

        # Add edges
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self.should_route,
            {"rag": "retriever", "general": "chat"},
        )
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        workflow.add_edge("chat", END)

        return workflow.compile()

    def router(self, state: AgentState):
        """Routes the user query to the appropriate node."""
        logging.info("in router")
        messages = state["messages"]
        user_input = messages[-1].content
        
        structured_llm = self.llm.with_structured_output(RouteQuery)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert at routing a user query to the appropriate data source."),
                ("human", f"Given the user query: '{user_input}', classify the intent and extract the company name."),
            ]
        )
        router_chain = prompt | structured_llm
        result = router_chain.invoke({"user_query": user_input})
        logging.info(f"router result: {result}")
        return {"intent": result.intent, "company_name": result.company_name}

    def should_route(self, state: AgentState):
        """Determines which node to route to based on the intent."""
        logging.info(f"in should_route, intent: {state['intent']}")
        return state["intent"]

    def retriever_node(self, state: AgentState):
        """Retrieves documents from the vector store."""
        logging.info("in retriever_node")
        from langchain_core.messages import SystemMessage
        company_name = state["company_name"]
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"filter": {"company": company_name}}
        )
        documents = retriever.invoke(state["messages"][-1].content)
        logging.info(f"retrieved documents: {documents}")
        return {"messages": [SystemMessage(content=f"Retrieved documents: {documents}")]}

    def generator_node(self, state: AgentState):
        """Generates a response based on the retrieved documents."""
        logging.info("in generator_node")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. You must only use the retrieved documents to answer the user's question. If the documents do not contain the answer, you must say that you do not know."),
                ("human", "User question: {user_question}\n\nRetrieved documents: {documents}"),
            ]
        )
        generator_chain = prompt | self.llm
        response = generator_chain.invoke({"user_question": state["messages"][-2].content, "documents": state["messages"][-1].content})
        return {"messages": [response]}

    def chat_node(self, state: AgentState):
        """Handles general chat conversations."""
        logging.info("in chat_node")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{user_question}"),
            ]
        )
        chat_chain = prompt | self.llm
        response = chat_chain.invoke({"user_question": state["messages"][-1].content})
        return {"messages": [response]}

    def run(self, query: str):
        """Runs the agent with the given query."""
        from langchain_core.messages import HumanMessage
        return self.graph.invoke({"messages": [HumanMessage(content=query)]})
