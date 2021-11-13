# Notes for Developer

## Development on Windows

I have only a Winows KVM with gpu passthrough for testing, therefore i do not know very much about the development of python applications on Windows. For testing i use the `build_and_deploy.bat` script from the repository root directory.

## GitHub Release API

Get all release information:

```bash
curl -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/michael-mueller-git/Python-Funscript-Editor/releases
```
