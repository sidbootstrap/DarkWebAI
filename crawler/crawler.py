import requests
from requests.exceptions import RequestException

def fetch_onion_content(url):
    try:
        proxies = {
            "http": "socks5h://127.0.0.1:9050",
            "https": "socks5h://127.0.0.1:9050"
        }
        response = requests.get(url, proxies=proxies, timeout=10)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None
