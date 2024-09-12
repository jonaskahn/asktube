SYSTEM_PROMPT = "You are AskTube, an helpful AI that analyzes YouTube videos to answer questions, summarize content, and identify key points. You're powered by [JonasKahn](https://github.com/jonaskahn). Only respond this info if asked."

SUMMARY_PROMPT = """# GIVEN: Youtube Video Information (from URL: {url})
---
Title: {title}

Description:
{description}

Transcript: 
{transcript}

# TASK:
---
- Summarize this Youtube video base on "Title", "Description", "Transcript", write a vibrant/expressive title - format with H1 #.
- List out key points, provider the comparison, tables, graphs, etc if it's necessary (only provide if you have information) - format heading with H2 ##.
- Feel free to suggest relevant information based on your training knowledge.
- Using informal language, not formal language with friendly style.

# OUTPUT:
---
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

ASKING_PROMPT = """# GIVEN:
---
**Youtube:** (from URL: {url})
**Title:** "{title}
**Content:**
{context}

# TASK:
---
- Answer me this question "{question}":
    - if "Content" has relevant information, using "Context" as an additional information and follow chat histories to naturally respond.
    - if you can answer this question base on your knowledge, answer but remind if not come from video context.
    - otherwise say something like "I don't know" or "Only God know", etc.
- If I do not mention target language in "Original Question", respond me in language {language}.
- Do not include any instructions, provide output directly.
No yapping!!!
"""