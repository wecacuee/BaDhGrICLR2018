for f in /z/home/shurjo/implicit-mapping/a3c-random-mazes/*/gen_stats*.json;
do
    lf=${f#/z/home/shurjo/implicit-mapping/a3c-random-mazes/};
    mkdir -p $(dirname $lf) ; 
    rsync -ca $(dirname $f)/*.json $(dirname $lf);
    echo -n "."
done

for f in /z/home/shurjo/implicit-mapping/a3c-random-mazes/*/videos/*.mp4;
do
    lf=${f#/z/home/shurjo/implicit-mapping/a3c-random-mazes/};
    mkdir -p $(dirname $lf) ; 
    rsync -ca $f $lf;
    echo -n "."
done
