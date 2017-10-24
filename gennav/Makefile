OO=out

BaDhGrICLR2018.pdf: $(OO)/main.pdf
	ln -fs $< $@
	xdg-open $@ &

$(OO)/main.pdf: iclr2018_conference.sty main_filtered.bib 

images/plot_reward_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py reward $< $@

images/plot_probability_%.pdf: npz_files/%.npz #py/plot.py
	python py/plot.py probability $< $@

images/plot_summary_bar_plots.pdf: ./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv py/plot.py
	python py/plot.py summary_bar_plots $(<D) $@

images/plot_ntrain_summary.pdf: ./exp-results/ntrained.csv py/plot.py
	python py/plot.py ntrain_summary $< $@

./exp-results/Static_Goal_Random_Spawn_Static_Maze.csv:
	pushd $(@D) && ./copy-files.sh ; popd
	pushd py && python process.py ; popd

main_filtered.bib: main.bib $(OO)/main.aux #/z/home/dhiman/wrk/group-bib/shared.bib
	aux2bib $(OO)/main.aux > $@

$(OO)/grid-2d-dmlab-figure-standalone.pdf:

./exp-results/training-09x09-0127-on-0127.png: #./exp-results/2017_10_19_14_28_16-test_3D_4actions_blind-none_0001_local/videos/99988800.mp4 py/process.py
	echo python py/process.py video_to_snapshot $< $@

./exp-results/training-1000-on-0127.png: #./exp-results/2017_10_19_17_34_49-test_3D_4actions_blind-none_0001_local/videos/99968400.mp4
	echo python py/process.py video_to_snapshot $< $@

include pdflatex.mk
