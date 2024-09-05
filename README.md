# AskTube

AI-Powered Summarizer, Q&amp;A Assistant on Youtube Videos 🤖

---

## Technology

- Language: Python, Js
- Framework/Lib: Sanic, Peewee, Pytubefix, Sentence Transformers, Sqlite, Chroma, VueJs
- Embedding Provider: OpenAI, Gemini, VoyageAI, Local (Sentence Transformers)
- Q&A Provider: OpenAI(GPT), Gemini, Claude, Local (Ollama)

---

## Why does this project exist?

- I’ve seen several GitHub repositories offering **AI-powered** summaries for YouTube videos, but none include **Q&A**
  functionality.
- I want to implement a more comprehensive solution while also gaining experience with AI to build an RAG application for business class.
- Get rid out of langchain: Since it's a very small project, I try to avoid heavy library like langchain.

---

## The Idea / Architecture

> The real implmentation might differ with this art.

### Phase 1: Extract data from given URL

![P1.png](docs/P1.png)

### Phase 2: Storing embedding chapter subtitles

![P2.png](docs/P2.png)

### Phase 3: Asking (included enrich question)

![P3.png](docs/P3.png)

---

## Notice

> 1. Do not use this for production.
> 2. Do not request any advanced features for management.

----

# Demo

> [Placeholder]
