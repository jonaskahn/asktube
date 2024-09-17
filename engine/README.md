# Getting Started

## Setup development env

- [Python 3.10](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [ffmpeg](https://www.ffmpeg.org/download.html)

## Configuration & Install

### Configuration

The default source configuration for `torch` in [pyproject.toml](pyproject.toml):

```toml
torch = "^2.4.1"
```

If you want to explicitly use a CPU, GPU, or ROCM, remove line `torch = "^2.4.1"` and add one of these lines below:

**CPU**

```toml
#PYTORCH
torch = { version = "^2.4.1", source = "pytorch-src" }

[[tool.poetry.source]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"
```

**NVIDIA**

- CUDA118

```toml
#PYTORCH
torch = { version = "^2.4.1", source = "pytorch-src" }

[[tool.poetry.source]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/cu118"
priority = "explicit"
```

- CUDA121: `Use default`
- CUDA124

```toml
#PYTORCH
torch = { version = "^2.4.1", source = "pytorch-src" }

[[tool.poetry.source]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/cu124"
priority = "explicit"
```

**AMD**
> **Windows** does not support AMD GPU with **PyTorch**, only config for **Linux**

```toml
#PYTORCH
torch = { version = "^2.4.1", source = "pytorch-src" }

[[tool.poetry.source]]
name = "pytorch-src"
url = "https://download.pytorch.org/whl/rocm6.1"
priority = "explicit"
```

### Install libraries

```shell
poetry install
```

## Run

```shell
poetry run python engine/server.py
```

## Env configuration

| Name                            | Default (Optional)            | Note                                                                                         |
|---------------------------------|-------------------------------|----------------------------------------------------------------------------------------------|
| AT_DEBUG_MODE                   | on                            | Turn on app debugger                                                                         |
| AT_APP_DIR                      | tmp                           | Store database, vector, models                                                               |
| AT_AUDIO_CHUNK_DETECT_DURATION  | 30 ( seconds)                 | Time to chunk audio segments, use to detect language                                         |
| AT_AUDIO_CHUNK_CHAPTER_DURATION | 600 ( seconds)                | Time to chunk audio segments, use to automatically split a long audio file                   |
| AT_LANGUAGE_PREFER_USAGE        | en                            | Default subtitle language that will be chosen                                                |
| AT_QUERY_SIMILAR_THRESHOLD      | 0.4                           | Default threshold to query similar documents for each question                               |
| AT_TOKEN_CONTEXT_THRESHOLD      | 2048                          | Default threshold to use whole transcript if context is not found                            |
| AT_AUDIO_ENHANCE_ENABLED        | off                           | Using enhance audio process (experiment)                                                     |
| AT_GEMINI_API_KEY               | None                          | If you prefer using embedding and QA with Google                                             |
| AT_OPENAI_API_KEY               | None                          | If you want to use embedding and QA with OpenAI                                              |
| AT_CLAUDE_API_KEY               | None                          | Iff you want to use QA with Claude                                                           |
| AT_VOYAGEAI_API_KEY             | None                          | If you want to use embedding with VoyageAI                                                   |
| AT_MISTRAL_API_KEY              | None                          | If you want to use embedding and QA with Mistral                                             |
| AT_GEMINI_EMBEDDING_MODEL       | models/text-embedding-004     | Prefer GEMINI model for embedding texts                                                      |
| AT_OPENAI_EMBEDDING_MODEL       | text-embedding-ada-002        | Prefer OpenAI model for embedding texts                                                      |
| AT_VOYAGEAI_EMBEDDING_MODEL     | voyage-large-2                | Prefer VoyageAI model for embedding texts                                                    |
| AT_MISTRAL_EMBEDDING_MODEL      | mistral-embed                 | Prefer MistralAI model for embedding texts                                                   |
| AT_LOCAL_EMBEDDING_MODEL        | intfloat/multilingual-e5-base | Prefer Local model for embedding texts                                                       |
| AT_LOCAL_EMBEDDING_DEVICE       | auto                          | Provider device to embedding texts in local (prefer "mps", then "cuda", otherwise use "cpu") |
| AT_SPEECH_TO_TEXT_PROVIDER      | local                         | Speech to text provider (local, openai, gemini)                                              |
| AT_LOCAL_WHISPER_MODEL          | base                          | Provider model to speech to text in local                                                    |
| AT_LOCAL_WHISPER_DEVICE         | auto                          | Provider device to speech to text in local (prefer "cuda", otherwise use "cpu")              |
| AT_LOCAL_OLLAMA_HOST            | http://localhost:11434        | Ollama host to connect                                                                       |
| AT_LOCAL_OLLAMA_MODEL           | qwen2                         | Ollama model to QA                                                                           |

## Prefer ENV for running LOCAL

> If your pc has Nvidia GPU, use "Recommendation" settings.

| Name                      | Value                         | Recommendation                 | Note |
|---------------------------|-------------------------------|--------------------------------|------|
| AT_LOCAL_OLLAMA_HOST      | http://localhost:11434        | -                              | -    |
| AT_LOCAL_OLLAMA_MODEL     | qwen2                         | llama3.1                       | -    |
| AT_LOCAL_EMBEDDING_MODEL  | intfloat/multilingual-e5-base | intfloat/multilingual-e5-large | -    |
| AT_LOCAL_EMBEDDING_DEVICE | cpu                           | gpu                            | -    |
| AT_LOCAL_WHISPER_MODEL    | base                          | large-v3                       | -    |
| AT_LOCAL_WHISPER_DEVICE   | cpu                           | gpu                            | -    |

### Notes

- If you still want to use free services with better result, please use:
    - **VoyageAI**  for embedding - Free to use 50M tokens without limitation.
    - **Gemini 1.5 Flash** for QA - Free with rate limit.

## Prefer ENV for Free (With Limitation)

> If your pc has nvidia GPU, use "Recommendation" settings.

| Name                        | Value                          | Recommendation                 | Note |
|-----------------------------|--------------------------------|--------------------------------|------|
| AT_LOCAL_WHISPER_MODEL      | base                           | large-v3                       | -    |
| AT_LOCAL_WHISPER_DEVICE     | cpu                            | gpu                            | -    |
| AT_VOYAGEAI_API_KEY         | [enter-your-voyageai-api-key ] | [enter-your-voyageai-api-key ] | -    |
| AT_VOYAGEAI_EMBEDDING_MODEL | voyage-large-2                 | voyage-large-2                 | -    |
| AT_GEMINI_API_KEY           | [enter-your-gemini-api-key]    | [enter-your-gemini-api-key]    | -    |

