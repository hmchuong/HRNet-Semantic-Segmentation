import glob

root_dir = "/home/ubuntu/deep_globe"
subdir = "test"

out_file = open("data/list/deepglobe/test.lst", "w")
for label_path in glob.glob(root_dir + "/{}/Label/*.png".format(subdir)):
    label_name = label_path.replace(root_dir + "/", "")
    image_name = label_name.replace("Label", "Sat").replace("_mask.png", "_sat.jpg")
    out_file.write("{}\t{}\n".format(image_name, label_name))

out_file.close()