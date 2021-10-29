import urllib.request
import requests
import re
import platform
import zipfile
import os
import sys
import time
import traceback
import shutil

from packaging import version
from bs4 import BeautifulSoup # beautifulsoup4
from tqdm import tqdm

VERSION = "v0.0.2"
LUA_EXTENSION_URL = "https://raw.githubusercontent.com/michael-mueller-git/Python-Funscript-Editor/main/contrib/Installer/assets/main.lua"
FUNSCRIPT_GENERATOR_RELEASE_URL = "https://github.com/michael-mueller-git/Python-Funscript-Editor/releases"
OFS_EXTENSION_DIR = os.path.expandvars(r'%APPDATA%\OFS\OFS_data\extensions')
LATEST_RELEASE_API_URL = 'https://api.github.com/repos/michael-mueller-git/Python-Funscript-Editor/releases/latest'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("Download latest release of Python-Funscript-Editor")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def error(msg):
    print("ERROR: " + msg)
    sys.exit()


def is_ofs_installed():
    if not os.path.exists(OFS_EXTENSION_DIR):
        error("OFS is not installed. Please download and install OFS. Befor running this installer open OFS once!")


def get_download_urls():
    # sometimes requests failed to fetch the url so we try up to 3 times
    for i in range(3):
        try:
            html_text = requests.get(FUNSCRIPT_GENERATOR_RELEASE_URL).text
            download_urls = { version.parse(re.search(r'v[^/]*', x).group().lower().replace("v", "")) : "https://github.com" + x \
                    for x in [link.get('href') for link in BeautifulSoup(html_text, 'html.parser').find_all('a') \
                        if link.get('href').endswith(".zip") and "/releases/" in link.get('href')]
            }
            latest = max(download_urls)
            return download_urls, latest, ""
        except:
            time.sleep(2)
            if i == 2:
                error("Download URL not found (" + FUNSCRIPT_GENERATOR_RELEASE_URL + ")")


def get_download_urls_with_api():
    # sometimes requests failed to fetch the url so we try up to 3 times
    for i in range(3):
        try:
            response = requests.get(LATEST_RELEASE_API_URL).json()
            assets_download_urls = ([x['browser_download_url'] for x in response['assets'] if 'browser_download_url' in x])
            program_download_url = [x for x in assets_download_urls if x.lower().endswith('.zip')]
            if len(program_download_url) == 0:
                error("MTFG Release not found")

            latest = response['tag_name'].lower().replace("v", "")
            return {latest: program_download_url[0]}, latest, response['body']
        except:
            time.sleep(2)
            if i == 2:
                error("Download URL not found (" + LATEST_RELEASE_API_URL + ")")


def is_latest_version_installed(version_file, version):
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            if str(f.read()).strip().lower() == "v"+str(version):
                print("You have already the latest version installed")
                sys.exit()


def update(download_urls, latest, release_notes):
    extension_dir = os.path.join(OFS_EXTENSION_DIR, "Funscript Generator Windows")
    zip_file = os.path.join(extension_dir, "funscript-editor-v" +  str(latest) + ".zip")
    dest_dir = os.path.join(os.path.dirname(zip_file), "funscript-editor")
    version_file = os.path.join(os.path.dirname(zip_file), "funscript-editor", "funscript_editor", "VERSION.txt")

    is_latest_version_installed(version_file, latest)

    print('New Version is available')
    print('')
    print('Release notes:')
    print(release_notes)

    trial = 0
    while True:
        os.makedirs(os.path.dirname(zip_file), exist_ok = True)
        if not os.path.exists(zip_file):
            download_url(download_urls[latest], zip_file)

        try:
            os.makedirs(dest_dir + "_update", exist_ok = True)
            with zipfile.ZipFile(zip_file) as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    zf.extract(member, dest_dir + "_update")
            break
        except:
            trial += 1
            if trial < 2:
                print("Local Version is corrupt redownloading")
                os.remove(zip_file)
                continue
            else:
                error("Installation failed")

    if os.path.exists(dest_dir):
        try: shutil.rmtree(dest_dir)
        except: error('Error while deleting old Version (Is OFS currenty running?)')

    shutil.move(dest_dir + "_update", dest_dir)

    with open(os.path.join(extension_dir, "main.lua"), "wb") as f:
        f.write(requests.get(LUA_EXTENSION_URL).content)


if __name__ == "__main__":
    print("MTFG OFS Extension Installer ", VERSION)
    try:
        if platform.system() != "Windows":
            error("This installer only work on Windows")

        is_ofs_installed()
        download_urls, latest, release_notes = get_download_urls_with_api()
        update(download_urls, latest, release_notes)

        print("Installation completed")

    except SystemExit as e:
        input()

    except:
        traceback.print_exc()
        input()
