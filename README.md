# AskTube

AI-Powered Summarizer, Q&amp;A Assistant on Youtube Videos 🤖

---
## 🔨 Technology

- **Language**: Python, JS
- **Framework/Lib**: Sanic, Peewee, Pytubefix, Sentence Transformers, Sqlite, Chroma, VueJs
- **Embedding Provider**: OpenAI, Gemini, VoyageAI, Local (Sentence Transformers)
- **AI Provider**: OpenAI, Gemini, Claude, Local (Ollama)
- **Speech To Text**: OpenAI Whisper (Local)

---
## 🤷🏽 Why does this project exist?

- I’ve seen several GitHub repositories offering **AI-powered** summaries for YouTube videos, but none include **Q&A**
  functionality.
- I want to implement a more comprehensive solution while also gaining experience with AI to build my own RAG application.

---
## 💡 The Idea / Architecture

> The real implmentation might differ with this art.

### 1️⃣ Extract data from given URL

![P1.png](docs/P1.png)

### 2️⃣ Storing embedding chapter subtitles

![P2.png](docs/P2.png)

### 3️⃣ Asking (included enrich question)

![P3.png](docs/P3.png)

---
## 🔊 Notice

> 1. Do not use this for production.
> 2. Do not request any advanced features for management.

----
## 🏃🏽‍➡️ Demo

> [Placeholder]

---
## 🗒️ Todo
- [ ] Add support to calling OpenAI Whisper API

---
## ⁉️ FAQ?

1. Does I need paid API to run?
  - No, you can fully run in your local machine if your pc have ability
  - You can still using free API to run with some limitations.
2. [Placeholder]
