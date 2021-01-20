import glob

root_dir = "/home/ubuntu/gleason"
subdir = "train"

out_file = open("data/list/gleason/train.lst", "w")
for label_path in glob.glob(root_dir + "/images/{}/*.jpg".format(subdir)):
    label_name = label_path.replace(root_dir + "/", "")
    image_name = label_name.replace("images", "gt").replace(".jpg", "_classimg_nonconvex.png")
    out_file.write("{}\t{}\n".format(label_name, image_name))

out_file.close()