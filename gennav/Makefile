OO=out
gennav_conference.pdf: $(OO)/gennav_conference.pdf iclr2018_conference.sty gennav_conference_filtered.bib
	ln -fs $< $@

images/plot_reward_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py probability $< $@

gennav_conference_filtered.bib: gennav_conference.bib /z/home/dhiman/wrk/group-bib/shared.bib
	aux2bib $(OO)/gennav_conference.aux > $@

include pdflatex.mk
