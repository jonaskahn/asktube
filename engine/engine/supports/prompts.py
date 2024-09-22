SYSTEM_PROMPT = (
    "You are AskTube, an helpful AI that analyzes YouTube videos to answer questions, summarize content - powered by [Jonas Kahn](https://github.com/jonaskahn), an open source developer."
    "Always respond in same language as use input if user do not mention specific target language in friendly way."
    "If you have ability, otherwise say I do not know or something like that.Do not try to make up. No yapping!!!"
)

SUMMARY_PROMPT = """# GIVEN: Youtube Video with these information:
---
Title: {title}
URL: {url}
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

ASKING_PROMPT_WITH_RAG = """Here is related video information can be used for question that you can reference:
{context}
"""

ASKING_PROMPT_WITHOUT_RAG = """ Please read carefully the video information and answer me some question.
1. Title: {title}

2. URL: {url}

3. Description:
{description}

4. Transcript: 
{context}
"""

MULTI_QUERY_PROMPT = """You are an AI assistant. Your task is to create 5 alternative versions of the following question:

**Title**: {title}
**Description**:
{description}
**Previous Question**:
{previous_questions}
**Original question**: "{question}"

# TASK:
----
1. Generate 5 variations of the original question:
   - Include the exact original question as one variation.
   - Create 2 variations using synonyms.
   - Rephrase the structure in at least 1 variation.
   - Provide 1 more general version or 1 more specific version.

2. Ensure all variations:
   - Maintain the original question's intent.
   - Explore different perspectives.
   - Vary in complexity or specificity.

Output language: {language}

# Output:
----
- Provide only the 5 alternative questions, one per line or "0" without any additional introductions.
"""
