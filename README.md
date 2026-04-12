# Simple RAG Chat App (Perplexity + Google Gemini)

A minimal **Retrieval-Augmented Generation (RAG)** chat application
built with **LangChain**, **Perplexity AI**, **Google Gemini**, and
**ChromaDB**.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/prothej227/perplexity-rag-news-chat.git
cd perplexity-rag-news-chat
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```env
# Required depending on model used
PPLX_API_KEY=your_perplexity_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

## Parameters

- `-m`, `--model`\
  Model provider to use. Options:
  - `google`
  - `perplexity`

---

## Usage

Run the chat application:

```bash
python main.py --model google
```

Or using shorthand:

```bash
python main.py -m google
```

If no parameter is provided, it defaults to `google`.

---

## Example Interaction

```text
You: What memorandum circular was signed this 2025?
Bot 🤖:
Answer: ...
Sources: ...
------------------------------------------------------------
```

Type `exit` or `quit` to end the chat.

---

## Notes

- Each question triggers a fresh retrieval (stateless RAG).
- Answers are based **only** on retrieved documents.
- Supports multiple LLM providers (Perplexity + Google Gemini).
- If the answer is not found, the model will respond:

```text
Not mentioned in the articles
```
