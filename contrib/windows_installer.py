import urllib.request
import requests
import re
import platform
import zipfile
import os
import sys
import shutil

from packaging import version
from bs4 import BeautifulSoup # beautifulsoup4
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# NOTE: this file is currently not working for this installer!
lua_extension_url = "https://raw.githubusercontent.com/michael-mueller-git/Python-Funscript-Editor/main/contrib/OpenFunscripter/extensions/Funscript%20Generator%20Windows/main.lua"

if platform.system() != "Windows":
    print("ERROR: This installer only work on Windows")
    sys.exit()

ofs_extension_dir = os.path.expandvars(r'%APPDATA%\OFS_data\extensions')

if not os.path.exists(ofs_extension_dir):
    print("ERROR: OFS is not installed. Please download and install OFS. Befor running this installer open OFS once!")
    print("Cancel installation")
    sys.exit()

release_url = "https://github.com/michael-mueller-git/Python-Funscript-Editor/releases"
html_text = requests.get(release_url).text
try:
    download_urls = { version.parse(re.search(r'v[^/]*', x).group().lower().replace("v", "")) : "https://github.com" + x \
            for x in [link.get('href') for link in BeautifulSoup(html_text, 'html.parser').find_all('a') \
                if link.get('href').endswith(".zip") and "/releases/" in link.get('href')]
    }
    latest = max(download_urls)
except:
    print("ERROR: download url not found")
    sys.exit()

extension_dir = os.path.join(ofs_extension_dir, "Funscript Generator Windows")
zip_file = os.path.join(extension_dir, "funscript-editor-v" +  str(latest) + ".zip")
dest_dir = os.path.join(os.path.dirname(zip_file), "funscript-editor")

os.makedirs(os.path.dirname(zip_file), exist_ok = True)
if not os.path.exists(zip_file):
    download_url(download_urls[latest], zip_file)

if os.path.exists(dest_dir):
    try: shutil.rmtree(dest_dir)
    except: print('Error while deleting old Version')

os.makedirs(dest_dir, exist_ok = True)
with zipfile.ZipFile(zip_file) as zf:
    for member in tqdm(zf.infolist(), desc='Extracting '):
        zf.extract(member, dest_dir)

with open(os.path.join(extension_dir, "main.lua"), "wb") as f:
    f.write(requests.get(lua_extension_url).content)
