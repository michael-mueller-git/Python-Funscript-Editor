
help:
	@echo "Makefile rules:"
	@echo "- all"
	@grep ".*:$$" ./Makefile | rev | cut -c 2- | rev | sort | xargs -I{} echo "- {}"

clean:
	@rm -rf ./build
	@rm -rf *.egg-info

package:
	@python3 setup.py bdist_wheel

install:
	@find dist -iname "*.whl" | sort | tail -n 1 | xargs -I {} pip3 install --force-reinstall {}

uninstall:
	@pip3 uninstall funscript-editor

docs:
	@chmod +x docs/code/generate_doc.sh
	@docs/code/generate_doc.sh
	@bash -c 'cd docs/app && bash build.sh'

all: docs package install clean

.PHONY: docs

