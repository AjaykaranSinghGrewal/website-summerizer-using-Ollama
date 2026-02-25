"""
Website summarizer using Ollama instead of OpenAI.
"""

from openai import OpenAI
from bs4 import BeautifulSoup
import requests

# Standard headers to fetch a website
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

system_prompt = """
You are a helpful assistant that analyzes the contents of a website,
and provides a short, helpful summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website):
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}
    ]

def fetch_website_contents(url):
    """
    Return the title and contents of the website at the given url;
    truncate to 2,000 characters as a sensible limit
    """
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]


def summarize(url):
    """Fetch and summarize a website using Ollama."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=messages_for(website)
    )
    return response.choices[0].message.content


def main():
    """Main entry point for testing."""
    url = input("Enter a URL to summarize: ")
    print("\nFetching and summarizing...\n")
    summary = summarize(url)
    print(summary)


if __name__ == "__main__":
    main()
