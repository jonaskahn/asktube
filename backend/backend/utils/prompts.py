SYSTEM_PROMPT = "You are an AI assistant for YouTube videos. Your name is AskTube, powered by JonasKahn. Only respond this information if asked."

SUMMARY_PROMPT = """
GIVEN: Youtube Video Information (from URL: {url})
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

RE_QUESTION_PROMPT = """
GIVEN:
----
I have a youtube video, video in language {video_lang} and a question in {question_lang}

Title: {title}
Summary: 
{summary}

Question: "{question}"

TASK:
----
- Requestioning the question, make a concise, relevant question based on video summary and question

OUTPUT:
----
- Response in language: {video_lang}
- Do not include any instructions, provide output directly.
No yapping!!!
"""

ASKING_PROMPT = """
GIVEN: Youtube Video: {title}, i have a question and suggestion context.
----
QUESTION: "{question}"

SUGGESTION CONTEXT:
{context}

TASK:
----
- According on the "Suggestion Context" and "Question", give me a right answer, following chat history.
- If you do not know the answer, just say "I don't know, please add a different question" or something like this -> Do not try to make up the answer.

OUTPUT:
----
- Response in language: {language}.
- Format in Markdown if "Question" does not specify any formatting.
- Do not include any instructions, provide output directly.
No yapping!!!
"""
