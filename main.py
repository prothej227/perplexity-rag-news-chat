from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_core import PerplexityRagChatApp
import os

load_dotenv()


def main():

    llm = ChatPerplexity(
        temperature=0,
        model="sonar",
        timeout=None,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chat_app = PerplexityRagChatApp(
        chat=llm,
        embeddings=embeddings,
    )

    print("\nPerplexity RAG News Chat")
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
    missing_deps = []
    if not os.getenv("PPLX_API_KEY"):
        missing_deps.append("PPLX_API_KEY")

    if not (
        os.path.isdir("news_chroma") and os.path.isfile("news_chroma/chroma.sqlite3")
    ):
        missing_deps.append("Vector database resources e.g. 'news_chroma' folder.")

    if missing_deps:
        raise EnvironmentError(f"Missing required resources: {', '.join(missing_deps)}")
    main()
