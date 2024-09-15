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
1. Create a detailed summary of the YouTube video based primarily on the "Title" and "Transcript". Use the "Description" only for context, ignoring any promotional or sponsored content it may contain.

2. Write an attention-grabbing, expressive title for your summary using H1 formatting (#).

3. List out all potential key points from the video, using H2 formatting (##) for each point. 

4. Highlight the following elements to make the recap more vibrant:
   - Emotional highlights: Describe 1-2 moments that likely evoked strong reactions
   - Quotable moments: Include 2-3 standout quotes or memorable lines
   - Visual highlights: Vividly describe an important visual element or scene
   - Video structure: Briefly analyze how the video was structured or paced
   - Audience engagement: Note how the video might have connected with viewers
   - Metaphors or analogies: Use a creative comparison to explain a key concept
   - Personal takeaways: Suggest 2-3 key lessons or insights from the video
   - Controversies (if any): Note any contentious points or debates raised
   - Behind-the-scenes: Include any relevant production details or background info mentioned in the video

5. If applicable, include:
   - Comparisons between concepts mentioned in the video
   - A table summarizing important data or categories
   - A simple ASCII graph or chart to visualize trends or statistics

6. Add 2-3 relevant pieces of information from your knowledge base that directly relate to the video's topic.

7. Use an informal, friendly tone throughout the summary. Write as if you're explaining to a friend.

# OUTPUT:
---
- Response language: {language}
- Format: Markdown
- Start directly with the content. Do not include any instructions or meta-text in your response.
- Do not mention or reference any sponsored content, advertisements, or promotional material that may appear in the video description.
- Do not include this prompt in response.
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
