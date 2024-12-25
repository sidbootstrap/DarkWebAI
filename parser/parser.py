from bs4 import BeautifulSoup

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string if soup.title else "No Title"
    content = soup.get_text()
    return {"title": title, "content": content}
