# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOC_INDEX     = $(BUILDDIR)/html/index.html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# add a "make clean" that removes everything in build, source/auto_api and source/auto_examples
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/auto_api/*
	rm -rf $(SOURCEDIR)/auto_examples/*
.PHONY: clean

# define a "make view" that opens the docs html index in the default browser
view:
	@# Check that the documentation has been built
	@if [ ! -f "$(DOC_INDEX)" ]; then \
	    echo "$(DOC_INDEX) not found - run 'make html' first."; \
	    exit 1; \
	fi; \
	# Launch the index.html with the system default browser
	@python -c "import pathlib, webbrowser; \
	    p = pathlib.Path('$(DOC_INDEX)').resolve(); \
	    webbrowser.open(p.as_uri())"