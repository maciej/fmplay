PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin

.PHONY: install

install:
	mkdir -p "$(BINDIR)"
	ln -sfn "$(CURDIR)/bin/fmplay" "$(BINDIR)/fmplay"
