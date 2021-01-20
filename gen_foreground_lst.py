import glob

root_dir = "data/foreground"


def write_list_file(path_dir, root_dir, img_extension, gt_extension, f):
    for image_path in glob.glob(root_dir + path_dir):
        img_name = image_path.replace(root_dir, "")
        label_name = img_name.replace(img_extension, gt_extension)
        out_file.write("{}\t{}\n".format(img_name, label_name))

# Train
out_file = open("data/list/foreground/train.lst", "w")

# For DUTS-TE
write_list_file("DUTS-TE/*.jpg", "data/foreground/", ".jpg", ".png", out_file)

# For DUTS-TR
write_list_file("DUTS-TR/*.jpg", "data/foreground/", ".jpg", ".png", out_file)

# For ecssd
write_list_file("ecssd/*.jpg", "data/foreground/", ".jpg", ".png", out_file)

# For fss
write_list_file("fss/*/*.jpg", "data/foreground/", ".jpg", ".png", out_file)

# For MSRA_10K
write_list_file("MSRA_10K/*.jpg", "data/foreground/", ".jpg", ".png", out_file)

out_file.close()

# Val
out_file = open("data/list/foreground/val.lst", "w")

# For BIG
write_list_file("BIG/val/*.jpg", "data/foreground/", "_im.jpg", "_gt.png", out_file)

out_file.close()

# Test
out_file = open("data/list/foreground/testbig.lst", "w")

# For BIG
write_list_file("BIG/test/*.jpg", "data/foreground/", "_im.jpg", "_gt.png", out_file)

out_file.close()

# Test
out_file = open("data/list/foreground/testpascal.lst", "w")

# For BIG
write_list_file("pascal/*_im.png", "data/foreground/", "_im.png", "_gt.png", out_file)

out_file.close()