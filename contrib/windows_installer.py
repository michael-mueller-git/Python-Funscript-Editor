import urllib.request
import requests
import re
import platform
import zipfile
import os

from packaging import version
from bs4 import BeautifulSoup # beautifulsoup4
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


release_url = "https://github.com/michael-mueller-git/Python-Funscript-Editor/releases"
html_text = requests.get(release_url).text
download_urls = { version.parse(re.search(r'v[^/]*', x).group().lower().replace("v", "")) : "https://github.com" + x \
        for x in [link.get('href') for link in BeautifulSoup(html_text, 'html.parser').find_all('a') \
            if link.get('href').endswith(".zip") and "/releases/" in link.get('href')]
}
latest = max(download_urls)
zip_file = "funscript-editor-v" +  str(latest) + ".zip"

if not os.path.exists(zip_file):
    download_url(download_urls[latest], zip_file)

with zipfile.ZipFile(zip_file) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        zf.extract(member, "funscript-editor")
