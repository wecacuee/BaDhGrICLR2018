OO=out
main.pdf: $(OO)/main.pdf aaai18.sty aaai.bst
	ln -fs $< $@

aaai18.sty aaai.bst: AuthorKit18.zip
	unzip -o $< AuthorKit18.zip '*/LaTeX/*.*' -x '__MACOSX/*'
	find AuthorKit18/ \( -name '*.sty' -or -name '*.bst' -or -name '*.cls' \) \
		-exec ln -sf \{} \;
	touch $@

AuthorKit18.zip:
	wget http://www.aaai.org/Publications/Templates/AuthorKit18.zip -O $@

images/plot_reward_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py probability $< $@

include pdflatex.mk
