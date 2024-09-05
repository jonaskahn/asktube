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

REFINED_QUESTION_PROMPT = """TASK:
- Translate a original input "{question}" to "{video_lang}"
OUTPUT:
- Do not include any instructions, provide output directly.
No yapping!!!
"""

ASKING_PROMPT = """GIVEN:
Youtube Video Information (from URL: {url})
Title: "{title}"
Context:
{context}
Original Question: "{question}"
Refined question: "{refined_question}"
TASK:
- Answer my "Refined question" if "Context" has relevant information, otherwise say something like "I don't know" or "Only God know".
- If I do not mention target language in "Original Question", respond me in language {language}.
- Do not include any instructions, provide output directly.
"""
