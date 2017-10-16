OO=out

BaDhGrICLR2018.pdf: $(OO)/main.pdf
	ln -fs $< $@

$(OO)/main.pdf: iclr2018_conference.sty main_filtered.bib 

images/plot_reward_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py probability $< $@

images/plot_summary_bar_plots.pdf: ./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv
	python py/plot.py summary_bar_plots $(<D) $@

images/plot_ntrain_summary.pdf: ./exp-results/Random_Goal_Random_Spawn_Random_Maze.csv
	python py/plot.py ntrain_summary $< $@

./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv:
	pushd $(@D) && ./copy-files.sh ; popd
	pushd py && python process.py ; popd

main_filtered.bib: main.bib /z/home/dhiman/wrk/group-bib/shared.bib
	aux2bib $(OO)/main.aux > $@

$(OO)/grid-2d-dmlab-figure-standalone.pdf:

include pdflatex.mk
