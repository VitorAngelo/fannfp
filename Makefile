SUBDIRS = lib examples

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

app: lib

.PHONY: clean
clean:
	@for a in $(SUBDIRS); do \
	    echo "processing folder $$a"; \
	    $(MAKE) clean -C $$a; \
	done;
	@echo "Done!"

