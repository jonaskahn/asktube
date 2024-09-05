SYSTEM_PROMPT = "You are AskTube, powered by JonasKahn - an AI assistant for YouTube videos. Only respond this information if asked."

SUMMARY_PROMPT = """GIVEN: Youtube Video Information (from URL: {url})
-----
Title: {title}

Description:
{description}

Transcript: 
{transcript}

TASK:
----
- Summarize this Youtube video base on "Title", "Description", "Transcript", write a vibrant/expressive title - format with H1 #.
- List out key points, provider the comparison, tables, graphs, etc if it's necessary (only provide if you have information) - format heading with H2 ##.
- Feel free to suggest relevant information based on your training knowledge.
- Using informal language, not formal language with friendly style.

OUTPUT:
- Response in language: {language}
- Format: Markdown
- Do not include any instructions, provide output directly.
No yapping!!!
"""

RE_QUESTION_PROMPT = """GIVEN:
----
I have a youtube video, video in language {video_lang} and a question in {question_lang}

Title: {title}
Summary: 
{summary}

Question: "{question}"

TASK:
----
- Requestioning the question, make a clear, concise, relevant question based on video summary and question follow chat history.
- If the question does not related to the video content, straightforwardly translate the question to {video_lang}.
OUTPUT:
----
- Response in language: {video_lang}
- Do not include any instructions, provide output directly.
No yapping!!!
"""

ASKING_PROMPT = """GIVEN: Youtube Video: {title}, i have a question and suggestion context.
----
QUESTION: "{question}"

SUGGESTION CONTEXT:
{context}

TASK:
----
- According on the "SUGGESTION CONTEXT" and "QUESTION", give me a right answer, following chat history.
- If the question beyond of "SUGGESTION CONTEXT":
    + If you do not have information and ability to answer the question, say provide a creative answer alternative for "I do not know" ] . 
    + If you have information and ability to answer the question, please provide the answer include a warning to me that your answer from your training knowledge, not from video.

OUTPUT:
----
- Output response in language: {language} if "Question" does not specify any target language.
- Format in Markdown if "Question" does not specify any formatting.
- Do not include any instructions, provide output directly.
No yapping!!!
"""
