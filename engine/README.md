# How to run

## Install requirements libraries

```shell
poetry install
```

## Run

```shell
poetry run python engine/server.py
```

## Env configuration

| Name                                 | Default (Optional)            | Note                                                                       |
|--------------------------------------|-------------------------------|----------------------------------------------------------------------------|
| ASKTUBE_DEBUG_MODE                   | on                            | Turn on app debugger                                                       |
| ASKTUBE_APP_DIR                      | tmp                           | Store database, vector, models                                             |
| ASKTUBE_AUDIO_CHUNK_DETECT_DURATION  | 30 ( seconds)                 | Time to chunk audio segments, use to detect language                       |
| ASKTUBE_AUDIO_CHUNK_CHAPTER_DURATION | 600 ( seconds)                | Time to chunk audio segments, use to automatically split a long audio file |
| ASKTUBE_AUDIO_CHUNK_CHAPTER_DURATION | 600 ( seconds)                | Time to chunk audio segments, use to automatically split a long audio file |
| ASKTUBE_LANGUAGE_PREFER_USAGE        | en                            | Default subtitle language that will be chosen                              |
| ASKTUBE_QUERY_SIMILAR_THRESHOLD      | 0.5                           | Default threshold to query similar documents for each question             |
| ASKTUBE_GEMINI_API_KEY               | None                          | If you prefer using embedding and QA with Google                           |
| ASKTUBE_OPENAI_API_KEY               | None                          | If you want to use embedding and QA with OpenAI                            |
| ASKTUBE_CLAUDE_API_KEY               | None                          | Iff you want to use QA with Claude                                         |
| ASKTUBE_VOYAGEAI_API_KEY             | None                          | If you want to use embedding with VoyageAI                                 |
| ASKTUBE_MISTRAL_API_KEY              | None                          | If you want to use embedding and QA with Mistral                           |
| ASKTUBE_GEMINI_EMBEDDING_MODEL       | models/text-embedding-004     | Prefer GEMINI model for embedding texts                                    |
| ASKTUBE_OPENAI_EMBEDDING_MODEL       | text-embedding-ada-002        | Prefer OpenAI model for embedding texts                                    |
| ASKTUBE_VOYAGEAI_EMBEDDING_MODEL     | voyage-large-2                | Prefer VoyageAI model for embedding texts                                  |
| ASKTUBE_MISTRAL_EMBEDDING_MODEL      | mistral-embed                 | Prefer MistralAI model for embedding texts                                 |
| ASKTUBE_LOCAL_EMBEDDING_MODEL        | intfloat/multilingual-e5-base | Prefer Local model for embedding texts                                     |
| ASKTUBE_LOCAL_EMBEDDING_DEVICE       | cpu                           | Provider device to embedding texts in local (*cpu, gpu*)                   |
| ASKTUBE_LOCAL_WHISPER_MODEL          | base                          | Provider model to speech to text in local                                  |
| ASKTUBE_LOCAL_WHISPER_DEVICE         | cpu                           | Provider device to speech to text in local (*cpu,gpu*)                     |
| ASKTUBE_LOCAL_OLLAMA_HOST            | http://localhost:11434        | Ollama host to connect                                                     |
| ASKTUBE_LOCAL_OLLAMA_MODEL           | qwen2                         | Ollama model to QA                                                         |

## Prefer ENV for running LOCAL

> If your pc has gpu, use "Recommendation" settings

| Name                           | Value                         | Recommendation                 | Note |
|--------------------------------|-------------------------------|--------------------------------|------|
| ASKTUBE_LOCAL_OLLAMA_HOST      | http://localhost:11434        | -                              | -    |
| ASKTUBE_LOCAL_OLLAMA_MODEL     | qwen2                         | llama3.1                       | -    |
| ASKTUBE_LOCAL_EMBEDDING_MODEL  | intfloat/multilingual-e5-base | intfloat/multilingual-e5-large | -    |
| ASKTUBE_LOCAL_EMBEDDING_DEVICE | cpu                           | gpu                            | -    |
| ASKTUBE_LOCAL_WHISPER_MODEL    | base                          | large-v3                       | -    |
| ASKTUBE_LOCAL_WHISPER_DEVICE   | cpu                           | gpu                            | -    |

### Notes

- If you still want you free services with better result, please use VoyageAI for embedding. They are allow to use 50M token without limitation.