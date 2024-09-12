<p align="center">
  <img src="images/logo.png" alt="AskTube's Logo"/>
</p>
<p align="center">
  <strong>AskTube - An AI-powered YouTube video summarizer and QA assistant powered by Retrieval Augmented Generation (RAG) 🤖</strong>
</p>

---

<p align="center">
  <i>Run it entirely on your local machine with Ollama, or cloud-based models like Claude, OpenAI, Gemini, Mistral, and more</i>
</p>
    
---

## 🔨 Technology

- **Language**: Python, JS
- **Framework/Lib**: Sanic, Peewee, Pytubefix, Sentence Transformers, Sqlite, Chroma, VueJs, etc.
- **Embedding Provider**:
  - [x] OpenAI
  - [x] Gemini
  - [x] VoyageAI
  - [x] Mistral
  - [x] Sentence Transformers (Local)
- **AI Provider**:
  - [x] OpenAI
  - [x] Claude
  - [x] Gemini
  - [x] Mistral
  - [x] Ollama (Local)
- **Speech To Text**:
  - [x] Faster-Whisper (Local)
  - [ ] OpenAI
  - [ ] Gemini

---

## 🤷🏽 Why does this project exist?

- I’ve seen several GitHub repositories offering **AI-powered** summaries for YouTube videos, but none include **Q&A**
  functionality.
- I want to implement a more comprehensive solution while also gaining experience with AI to build my own RAG application.

---

## 💡 Architecture

> The real implementation might differ from this art due to its complexity.

### 1️⃣ Extract data from given URL

![P1.png](images/P1.png)

### 2️⃣ Storing embedding chapter subtitles

![P2.png](images/P2.png)

### 3️⃣ Asking (included enrich question)

![P3.png](images/P3.png)

---

## 🔊 Notice

> 1. Do not use this for production. This aimed for end-users on their local machines.
> 2. Do not request any advanced features for management.

---

## 🏃🏽‍➡️ Demo

> [Placeholder]

---

## ✍🏿 For development

- [Backend](./engine/README.md)

---

## ⁉️ FAQ and Troubleshooting

- **Do I need paid API to run?**
  - No, you can fully run in your local machine if your pc have ability
  - You can still use free API to run with some limitations.
