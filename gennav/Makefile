OO=out

$(OO)/main.pdf: images/plot_reward_3D-1000.pdf images/plot_probability_3D-1000.pdf \
			iclr2018_conference.sty main_filtered.bib 

BaDhGrICLR2018.pdf: $(OO)/main.pdf
	ln -fs $< $@

images/plot_reward_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz py/plot.py
	python py/plot.py probability $< $@

main_filtered.bib: main.bib /z/home/dhiman/wrk/group-bib/shared.bib
	aux2bib $(OO)/main.aux > $@

include pdflatex.mk
