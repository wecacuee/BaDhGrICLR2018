for f in /z/home/shurjo/implicit-mapping/a3c-random-mazes/*/gen_stats*.json;
do
    lf=${f#/z/home/shurjo/implicit-mapping/a3c-random-mazes/};
    mkdir -p $(dirname $lf) ; 
    cp $(dirname $f)/*.json $(dirname $lf);
done

for f in /z/home/shurjo/implicit-mapping/a3c-random-mazes/*/videos/*.mp4;
do
    lf=${f#/z/home/shurjo/implicit-mapping/a3c-random-mazes/};
    mkdir -p $(dirname $lf) ; 
    cp $f $lf;
done
