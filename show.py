import sys
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt

# Flags
ALIGN_TBL_HEIGHT = False
problem_images=["2019-02-13-18-53-5412345",
"2019-02-13-19-15-2512345",
"2019-02-13-19-10-3012345",
"2019-02-13-19-43-3512345",
"2019-02-13-19-34-3612345",
"2019-02-13-19-00-3712345",
"2019-02-13-18-54-5912345",
"2019-02-13-18-54-0712345",
"2019-02-13-19-03-0712345"]
# Parameters
input_dirname = 'snapshots-thomas' if len(sys.argv) < 2 else sys.argv[1]
output_base_dir = 'snapshots-thomas-noheight' if len(sys.argv) < 2 else sys.argv[2] 
sensor = 'primesense'
# Get files

fnames = os.listdir(input_dirname)
for fname in fnames:
    if 'Depth' in fname:
        for prb in problem_images[:]:
            if prb in fname:
                im = skimage.data.imread("%s/%s"%(input_dirname, fname), as_gray=True)
                im = (im / 1000.0)

                plt.imshow(im, cmap=plt.cm.gray_r)
                plt.show()