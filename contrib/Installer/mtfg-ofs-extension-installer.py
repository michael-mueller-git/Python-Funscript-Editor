import urllib.request
import requests
import json
import platform
import zipfile
import os
import sys
import time
import traceback
import shutil
import subprocess

from packaging import version
from bs4 import BeautifulSoup # beautifulsoup4
from tqdm import tqdm


VERSION = "v0.2.1"
FUNSCRIPT_GENERATOR_RELEASE_URL = "https://github.com/michael-mueller-git/Python-Funscript-Editor/releases"
OFS_EXTENSION_DIR = os.path.expandvars(r'%APPDATA%\OFS\OFS2_data\extensions')
OFS_V1_EXTENSION_DIR = os.path.expandvars(r'%APPDATA%\OFS\OFS_data\extensions')
LATEST_RELEASE_API_URL = 'https://api.github.com/repos/michael-mueller-git/Python-Funscript-Editor/releases/latest'
EXTENSION_NAME = "Funscript Generator Windows"

if os.path.exists(os.path.expandvars(r'%APPDATA%\OFS\OFS3_data')):
    OFS_EXTENSION_DIR = os.path.expandvars(r'%APPDATA%\OFS\OFS3_data\extensions')

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("Download latest release of Python-Funscript-Editor")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def uninstall():
    version_file = os.path.join(OFS_EXTENSION_DIR, EXTENSION_NAME, "funscript-editor", "funscript_editor", "VERSION.txt")
    if os.path.exists(version_file):
        os.remove(version_file)

def error(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def is_ofs_installed():
    print('check if', OFS_EXTENSION_DIR, 'exists')
    if not os.path.exists(OFS_EXTENSION_DIR):
        if os.path.exists(OFS_V1_EXTENSION_DIR):
            error("Please update your [OFS](https://github.com/OpenFunscripter/OFS/releases) Installation. Then run this installer again")
        else:
            error("OFS is not installed. Please download and install [OFS](https://github.com/OpenFunscripter/OFS/releases). Befor running this installer open OFS once!!")


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


def get_required_installer_version(mtfg_dir):
    if not os.path.exists(os.path.join(mtfg_dir, "install.json")):
        uninstall()
        error("Installer notes missing, please use an newer version")

    with open(os.path.join(mtfg_dir, "install.json"), 'r') as f:
        installer_notes = json.load(f)

    return installer_notes['minRequiredVersion']


def process_exists(process_name):
    try:
        call = 'TASKLIST', '/FI', 'imagename eq %s' % process_name
        output = subprocess.check_output(call).decode()
        last_line = output.strip().split('\r\n')[-1]
        return last_line.lower().startswith(process_name.lower())
    except:
        return False


def install_lua_scripts(root_src_dir, extension_dir):
    if not os.path.exists(root_src_dir):
        uninstall()
        error(str(root_src_dir) + " do not exists (corrupt install pack?)")
    for src_dir, _, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, extension_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)


def is_latest_version_installed(version_file, version):
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            if str(f.read()).strip().lower() == "v"+str(version):
                print("You have already the latest version installed")
                sys.exit()


def update(download_urls, latest, release_notes):
    extension_dir = os.path.join(OFS_EXTENSION_DIR, EXTENSION_NAME)
    zip_file = os.path.join(extension_dir, "funscript-editor-v" +  str(latest) + ".zip")
    mtfg_dir = os.path.join(os.path.dirname(zip_file), "funscript-editor")
    version_file = os.path.join(os.path.dirname(zip_file), "funscript-editor", "funscript_editor", "VERSION.txt")

    is_latest_version_installed(version_file, latest)

    print('New Version is available')

    trial = 0
    while True:
        os.makedirs(os.path.dirname(zip_file), exist_ok = True)
        if not os.path.exists(zip_file):
            download_url(download_urls[latest], zip_file)

        try:
            if os.path.exists(mtfg_dir + "_update"):
                try: shutil.rmtree(mtfg_dir + "_update")
                except: error('Error while deleting old update Version (Restart you computer and try again)')

            os.makedirs(mtfg_dir + "_update", exist_ok = True)
            with zipfile.ZipFile(zip_file) as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    zf.extract(member, mtfg_dir + "_update")
            break
        except:
            trial += 1
            if trial < 2:
                print("Local Version is corrupt redownloading")
                os.remove(zip_file)
                continue
            else:
                uninstall()
                error("Installation failed")

    if process_exists("OpenFunscripter.exe"):
        uninstall()
        error("OpenFunscripter is currently running, please close OpenFunscripter and execute this installer again, to update the MTFG Extension")

    if os.path.exists(mtfg_dir):
        try: shutil.rmtree(mtfg_dir)
        except:
            uninstall()
            error('Error while deleting old Version (Is OFS currenty running?)')

    min_required_installer_version = get_required_installer_version(mtfg_dir + "_update")
    print('check min required installer version', min_required_installer_version)
    if version.parse(min_required_installer_version) > version.parse(VERSION.lower().replace('v', '')):
        uninstall()
        error("min required installer version is " + str(min_required_installer_version))

    shutil.move(mtfg_dir + "_update", mtfg_dir)
    install_lua_scripts(os.path.join(mtfg_dir, "OFS"), extension_dir)



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
