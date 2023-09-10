import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import sys

def get_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    parsed_url = urlparse(url)

    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
            continue
        if not urlparse(href).netloc:  # Relative URL
            href = urljoin(parsed_url.scheme + '://' + parsed_url.netloc, href)
        parsed_href = urlparse(href)
        path_parts = parsed_href.path.split('/')
        if (
            parsed_href.netloc != parsed_url.netloc
            and not parsed_href.path.endswith('.jpg')
            and not parsed_href.path.endswith('.png')
           # and not any(part.lower().startswith('archive') for part in path_parts[:3])
            and not any(part.lower().startswith('wiki') for part in path_parts[:3])
            and not any('wiki' in part.lower() for part in path_parts[:3])
            and 'wiki' not in href.lower()
            and not any(part.lower().startswith('web.archive') for part in path_parts[:3])
            and not any('web.archive' in part.lower() for part in path_parts[:3])
            and 'web.archive' not in href.lower()
            and not any(part.lower().startswith('archive') for part in path_parts[:3])
            and not any('archive' in part.lower() for part in path_parts[:3])
            and 'archive' not in href.lower()

        ):
            links.append(href)

    return links

