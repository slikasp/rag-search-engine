import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
MODEL = "gemma-3-27b-it"

def enhance_spelling(query: str) -> str:
    content = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    content = client.models.generate_content(
        model=MODEL,
        contents=content
    )
    if content.text is not None:
        return content.text
    else:
        return ""
    
def enhance_rewrite(query: str) -> str:
    content = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    content = client.models.generate_content(
        model=MODEL,
        contents=content
    )
    if content.text is not None:
        return content.text
    else:
        return ""

def enhance_expand(query: str) -> str:
    content = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie film"
- "action movie with bear" -> "action thriller bear chase fight"
- "comedy with bear" -> "comedy funny bear humor"

User query: "{query}"
"""
    content = client.models.generate_content(
        model=MODEL,
        contents=content
    )
    if content.text is not None:
        return content.text
    else:
        return ""
    
def rerank_individual(query: str, title: str, description: str) -> int:
    content = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    content = client.models.generate_content(
        model=MODEL,
        contents=content
    )
    if content.text is not None:
        return int(content.text)
    else:
        return 0
    
def rerank_batch(query: str, doc_list_str: list) -> list:
    content = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    content = client.models.generate_content(
        model=MODEL,
        contents=content
    )
    if content.text is not None:
        return json.loads(content.text)
    else:
        return []