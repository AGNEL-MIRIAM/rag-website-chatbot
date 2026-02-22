import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrape_website(url: str) -> str:
    """
    Scrapes visible text content from a website.

    Args:
        url (str): Website URL

    Returns:
        str: Cleaned textual content
    """
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        text = soup.get_text(separator=" ")
        clean_text = " ".join(text.split())

        return clean_text

    except Exception as e:
        print(f"Error scraping website: {e}")
        return ""
