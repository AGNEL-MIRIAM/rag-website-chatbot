import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scrape_page(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Target meaningful tags only
        tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "article", "section"])
        text = " ".join(tag.get_text(separator=" ", strip=True) for tag in tags)
        return " ".join(text.split())
    except Exception as e:
        print(f"Error scraping {url}:", e)
        return ""

def get_internal_links(url: str, soup: BeautifulSoup) -> list:
    base = urlparse(url)
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        parsed = urlparse(full)
        # Only same domain, no fragments or query strings
        if parsed.netloc == base.netloc and not parsed.fragment:
            links.add(parsed._replace(query="", fragment="").geturl())
    return list(links)

def scrape_website(start_url: str, max_pages: int = 5) -> str:
    visited = set()
    to_visit = [start_url]
    all_text = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url}")

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")

            tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "article"])
            text = " ".join(tag.get_text(separator=" ", strip=True) for tag in tags)
            all_text.append(" ".join(text.split()))

            # Add child links
            new_links = get_internal_links(url, soup)
            to_visit.extend([l for l in new_links if l not in visited])
        except Exception as e:
            print(f"Failed: {url} — {e}")

    return "\n\n".join(all_text)
