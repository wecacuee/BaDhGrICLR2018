from __future__ import print_function
import __builtin__
import sys
import os.path as op
import time
from IPython.lib import deepreload
import subprocess
import traceback
#__builtin__.reload = deepreload.reload
while True:
    import plot
    plot_file = plot.__file__.replace(".pyc", ".py")
    plot_mtime = op.getmtime(plot_file)
    try:
        reload(plot)
        from plot import *
        plot.keeplotting("/tmp/plot.pdf")
        subprocess.Popen("evince /tmp/plot.pdf".split())
    except Exception as e:
        traceback.print_exc(e)

    time.sleep(0.5)
    while op.getmtime(plot_file) <= plot_mtime:
        time.sleep(0.5)
    print("file modified ")
    
    
