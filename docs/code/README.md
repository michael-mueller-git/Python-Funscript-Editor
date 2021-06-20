# Code Documentation

## Sphinx

[Sphinx](https://www.sphinx-doc.org/en/master/) is a tool that makes it easy to create intelligent and beautiful documentation. Python code can include documentation right inside its source code. The default way of doing so relies on docstrings, which are defined in a triple quote format. To make your documentation look beautiful, you can take advantage of Sphinx, which is designed to make pretty Python documents. In order to tell Sphinx what and how to generate the documentation, Sphinx use the configuration file `conf.py`.

## Build the Documentation

On a linux host run:

```bash
./generate_doc.sh
```

This will create a new directory, named `_build` with the Documentation.
