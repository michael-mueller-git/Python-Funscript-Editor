# App Documentation

## MkDocs

MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file. There's a single configuration file named `mkdocs.yml` and a folder named `docs` that will contain your documentation source files. ([mkdocs.org](https://www.mkdocs.org/)).

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. Make sure you're in the same directory as the `mkdocs.yml` configuration file, and then start the server by running the `mkdocs serve` command. Open up `http://127.0.0.1:8000/` in your browser, and you'll see the default home. The dev-server also supports auto-reloading, and will rebuild your documentation whenever anything in the configuration file, documentation directory, or theme directory changes.

## Build the Documentation

```bash
mkdocs build
```

This will create a new directory, named `site` with the Documentation.
