from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_core import RagChatApp
import os
import argparse
from enum import Enum

load_dotenv()


class ModelProvider(Enum):
    PERPLEXITY = "perplexity"
    GOOGLE = "google"


def start(selected_model: str):
    # Initialize LLM
    if selected_model == ModelProvider.PERPLEXITY.value:
        llm = ChatPerplexity(
            temperature=0,
            model="sonar",
            timeout=None,
        )
    elif selected_model == ModelProvider.GOOGLE.value:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError("Model provider name is not supported.")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # RAG App
    chat_app = RagChatApp(
        chat=llm,
        embeddings=embeddings,
    )

    print("\nRAG News Chat")
    print(f"Model used: {selected_model.upper()}")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to end.\n")

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break

            response = chat_app.ask_question(question)
            print("\nBot 🤖:")
            print(response)
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nChat ended by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG ChatApp")
    parser.add_argument(
        "-m",
        "--model",
        choices=[e.value for e in ModelProvider],
        default="google",
        help="Model provider to use (perplexity or google)",
    )
    args = parser.parse_args()
    model = str(args.model).lower()

    # Dependency checks
    missing_deps = []

    if model == ModelProvider.PERPLEXITY.value and not os.getenv("PPLX_API_KEY"):
        missing_deps.append("PPLX_API_KEY")

    if model == ModelProvider.GOOGLE.value and not os.getenv("GOOGLE_API_KEY"):
        missing_deps.append("GOOGLE_API_KEY")

    if not (
        os.path.isdir("news_chroma") and os.path.isfile("news_chroma/chroma.sqlite3")
    ):
        missing_deps.append("Vector database resources e.g. 'news_chroma' folder.")

    if missing_deps:
        raise EnvironmentError(f"Missing required resources: {', '.join(missing_deps)}")

    start(selected_model=model)
