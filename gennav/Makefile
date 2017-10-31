SHELL:=/bin/bash
OO=out

all: BaDhGrICLR2018.pdf BaDhGrNIPS2017.pdf

BaDhGrICLR2018.pdf: $(OO)/main.pdf
	ln -fs $< $@
	if [ $${DISPLAY} == ":0" ]; then \
		xdg-open $@ & \
	fi

BaDhGrNIPS2017.pdf: $(OO)/nips-drl-symposium.pdf
	ln -fs $< $@
	if [ $${DISPLAY} == ":0" ]; then \
		xdg-open $@ & \
	fi

$(OO)/main.pdf: iclr2018_conference.sty main_filtered.bib 

$(OO)/nips-drl-symposium.pdf: nips_2017.sty main_filtered.bib

nips_2017.sty:
	wget https://media.nips.cc/Conferences/NIPS2017/Styles/nips_2017.sty -O $@

images/plot_reward_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py probability $< $@

images/plot_summary_bar_plots.pdf: ./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv #py/plot.py
	python py/plot.py summary_bar_plots $(<D) $@

images/plot_ntrain_summary.pdf: ./exp-results/ntrained.csv #py/plot.py
	python py/plot.py ntrain_summary $< $@

./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv:
	pushd $(@D) && ./copy-files.sh ; popd
	pushd py && python process.py ; popd

./exp-results/ntrained.csv: #py/process.py
	pushd py && python process.py ntrain_data; popd

# Nice way for optional dependencies
maybe_shared_bib=$(wildcard /z/home/dhiman/wrk/group-bib/shared.bib)

main_filtered.bib: main.bib $(maybe_shared_bib)
	aux2bib $(OO)/main.aux > $@

$(OO)/grid-2d-dmlab-figure-standalone.pdf:

#include snapshots.mk

include pdflatex.mk
