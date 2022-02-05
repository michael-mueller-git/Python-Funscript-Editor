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
import subprocess
import ctypes

from packaging import version
from bs4 import BeautifulSoup # beautifulsoup4
from tqdm import tqdm

VERSION = "v0.0.4"
LUA_EXTENSION_URL = "https://raw.githubusercontent.com/michael-mueller-git/Python-Funscript-Editor/main/contrib/Installer/assets/main.lua"
FUNSCRIPT_GENERATOR_RELEASE_URL = "https://github.com/michael-mueller-git/Python-Funscript-Editor/releases"
OFS_EXTENSION_DIR = os.path.expandvars(r'%APPDATA%\OFS\OFS_data\extensions')
LATEST_RELEASE_API_URL = 'https://api.github.com/repos/michael-mueller-git/Python-Funscript-Editor/releases/latest'

USE_HTTP_ONLY=False
if os.path.exists("HTTP_ONLY.txt"):
    print("Use HTTP only mode")
    USE_HTTP_ONLY=True
    LUA_EXTENSION_URL = LUA_EXTENSION_URL.replace("https:", "http:")
    FUNSCRIPT_GENERATOR_RELEASE_URL = FUNSCRIPT_GENERATOR_RELEASE_URL.replace("https:", "http:")
    LATEST_RELEASE_API_URL = LATEST_RELEASE_API_URL.replace("https:", "http:")


def Mbox(title, text, style):
    """
    ##  Styles:
    ##  0 : OK
    ##  1 : OK | Cancel
    ##  2 : Abort | Retry | Ignore
    ##  3 : Yes | No | Cancel
    ##  4 : Yes | No
    ##  5 : Retry | Cancel
    ##  6 : Cancel | Try Again | Continue
    """
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    if USE_HTTP_ONLY:
        url = url.replace("https:", "http:")
    print("Download latest release of Python-Funscript-Editor")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def error(msg):
    print("ERROR: " + msg)
    sys.exit()


def is_ofs_installed():
    if not os.path.exists(OFS_EXTENSION_DIR):
        error("OFS is not installed. Please download and install OFS. Befor running this installer open OFS once!")


def get_download_urls_with_api():
    # sometimes requests failed to fetch the url so we try up to 3 times
    for i in range(3):
        try:
            response = requests.get(LATEST_RELEASE_API_URL).json()
            assets_download_urls = ([x['browser_download_url'] for x in response['assets'] if 'browser_download_url' in x])
            program_download_url = [x for x in assets_download_urls if x.lower().endswith('.zip') and "funscript-editor" in x ]
            if len(program_download_url) == 0:
                error("MTFG latest release not found (Try again later)")

            latest = response['tag_name'].lower().replace("v", "")
            return {latest: program_download_url[0]}, latest, response['body']
        except:
            time.sleep(2)
            if i == 2:
                error("Download URL not found (" + LATEST_RELEASE_API_URL + ")")


def process_exists(process_name):
    try:
        call = 'TASKLIST', '/FI', 'imagename eq %s' % process_name
        output = subprocess.check_output(call).decode()
        last_line = output.strip().split('\r\n')[-1]
        return last_line.lower().startswith(process_name.lower())
    except:
        return False


def install_main_lua(extension_dir, dest_dir):
    if os.path.exists(os.path.join(dest_dir, "main.lua")):
        shutil.copy2(os.path.join(dest_dir,"main.lua"), os.path.join(extension_dir, "main.lua"))
    else:
        print("Download main.lua from GitHub...")
        # sometimes requests failed to fetch the url so we try up to 3 times
        for i in range(3):
            try:
                with open(os.path.join(extension_dir, "main.lua"), "wb") as f:
                    f.write(requests.get(LUA_EXTENSION_URL).content)
                break
            except:
                if os.path.exists(dest_dir):
                    try: shutil.rmtree(dest_dir)
                    except: pass
                error('main.lua insallation failed')


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
    # print('')
    # print('Release notes:')
    # print(release_notes)

    trial = 0
    while True:
        os.makedirs(os.path.dirname(zip_file), exist_ok = True)
        if not os.path.exists(zip_file):
            download_url(download_urls[latest], zip_file)

        try:
            if os.path.exists(dest_dir + "_update"):
                try: shutil.rmtree(dest_dir + "_update")
                except: error('Error while deleting old update Version (Restart you computer and try again)')

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

    if process_exists("OpenFunscripter.exe"):
        error("OpenFunscripter is currently running, please close OpenFunscripter and execute this installer again, to update the MTFG Extension")

    if os.path.exists(dest_dir):
        try: shutil.rmtree(dest_dir)
        except: error('Error while deleting old Version (Is OFS currenty running?)')

    shutil.move(dest_dir + "_update", dest_dir)
    install_main_lua(extension_dir, dest_dir)



if __name__ == "__main__":
    print("MTFG OFS Extension Installer", VERSION)
    try:
        if platform.system() != "Windows":
            error("This installer only work on Windows")

        is_ofs_installed()
        print('Fetch latest release data from github.com')
        download_urls, latest, release_notes = get_download_urls_with_api()
        update(download_urls, latest, release_notes)

        print("Installation completed")
        time.sleep(4)

    except SystemExit as e:
        input()

    except:
        traceback.print_exc()
        input()
