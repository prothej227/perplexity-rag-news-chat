from typing import List, Dict, Union, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_perplexity import ChatPerplexity
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_chroma import Chroma


class PerplexityRagChatApp:
    _PERSIST_DIR = Path(__file__).parent.joinpath("news_chroma")

    def __init__(
        self,
        chat: ChatPerplexity,
        embeddings: Embeddings,
        persistence_directory: Optional[None] = None,
        retriever_config: Dict[str, Union[str, dict]] = {
            "search_type": "similarity",
            "search_kwargs": {"k": 3},
        },
    ) -> None:
        self._embeddings = embeddings
        self._chat = chat
        self._vectore_store: Chroma
        self._retriever: VectorStoreRetriever
        self._retriever_config = retriever_config
        self._persistence_directory = persistence_directory
        self._prompt: PromptTemplate = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are answering questions using ONLY the news articles below.

            Rules:
            - Base your answer strictly on the provided articles.
            - If the answer is not found, say "Not mentioned in the articles".
            - After the answer (only if answer is found), list the source file name(s).

            Articles:
            {context}

            Question:
            {question}

            Answer format:
            Answer: <your answer>
            Sources: <comma-separated file names>
            """,
        )

    @property
    def vector_store(self) -> Chroma:
        return Chroma(
            persist_directory=(
                str(PerplexityRagChatApp._PERSIST_DIR)
                if self._persistence_directory is None
                else self._persistence_directory
            ),
            embedding_function=self._embeddings,
        )

    @property
    def retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(**self._retriever_config)

    @property
    def prompt(self) -> PromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, value) -> None:
        if not (isinstance(value, PromptTemplate) or value):
            raise ValueError(
                "Prompt cannot be empty or not an instance of PromptTemplate"
            )
        self._prompt = value

    def rag_chain(self) -> Runnable[str, BaseMessage]:
        return (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self._chat
        )

    def ask_question(self, question: str):
        chain = self.rag_chain()
        response = chain.invoke(question)
        return response.content

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"\n[Source: {doc.metadata.get('source','unknown')}]\n{doc.page_content}"
            for doc in docs
        )
