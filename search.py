import asyncio
import aiohttp
from bs4 import BeautifulSoup
from collections import defaultdict
from urllib.parse import urlparse, urljoin
import spacy

# Seed URLs to start crawling from
seed_urls = [
    'https://en.wikipedia.org/wiki/Main_Page',
    'https://www.bbc.com/',
    'https://www.nytimes.com/',
    'https://www.aljazeera.com/',
    'https://edition.cnn.com/',
    'https://www.theguardian.com/international'
    'https://quora.com',
    'https://amazon.in',
    'https://facebook.com',
    'https://reddit.com',
    'https://openai.com/',
]

# Additional seed URLs can be added here

search_query = 'israel palestine-gaza'
max_depth = 3
score_threshold = 0.08  # Lower threshold for testing

url_scores = defaultdict(float)

# Load English language model in SpaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Please download the 'en_core_web_sm' model by running:")
    print("python -m spacy download en_core_web_sm")
    exit()

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                try:
                    html = await response.text(encoding='utf-8')
                    return html
                except UnicodeDecodeError as e:
                    print(f"UnicodeDecodeError: Failed to decode {url}: {e}")
                    return None
            else:
                print(f"Failed to fetch {url}. Status code: {response.status}")
                return None
    except aiohttp.ClientError as e:
        # print(f"Error fetching {url}: {e}")
        return None

def extract_text(html):
    if html is None:
        return ""
    soup = BeautifulSoup(html, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
    return text

def score_url(text, query):
    if text is None or query is None:
        return 0.0

    # Process text with SpaCy
    doc = nlp(text)

    # Calculate score based on relevance to search query
    relevance_score = 0.0
    for token in doc:
        if token.text.lower() in query.lower():
            relevance_score += 1.0  # Simple keyword matching score
        if token.text.lower() in ['today', 'latest']:
            relevance_score += 0.5  # Boost score for relevant terms

    # Example of additional scoring based on named entities
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            relevance_score += 0.5  # Boost score for dates mentioned

    # Normalize score by text length
    if len(doc) != 0:
        normalized_score = relevance_score / len(doc)
    else:
        normalized_score = 0

    return normalized_score

async def crawl(url, depth, session):
    if depth > max_depth:
        return
    html = await fetch_url(session, url)
    if html:
        text = extract_text(html)
        score = score_url(text, search_query)
        if score > score_threshold:
            url_scores[url] = score
            print(f"Score: {score:.2f} - {url}")  # Print when score meets threshold
            soup = BeautifulSoup(html, 'html.parser')
            tasks = []
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if next_url not in url_scores:
                    task = asyncio.create_task(crawl(next_url, depth + 1, session))
                    tasks.append(task)
            await asyncio.gather(*tasks)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in seed_urls:
            task = asyncio.create_task(crawl(url, 0, session))
            tasks.append(task)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
