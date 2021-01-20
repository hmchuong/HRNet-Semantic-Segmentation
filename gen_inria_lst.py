import os
import glob
import numpy as np

root_dir = "/home/ubuntu/inria"
subdir = "test"

out_file = open("data/list/inria/test.lst", "w")
for image_name in np.load(os.path.join(root_dir, "ids.{}.npy".format(subdir))):
    image_name = image_name.decode()

    out_file.write("images/{}\tgt/{}\n".format(image_name, image_name))
out_file.close()