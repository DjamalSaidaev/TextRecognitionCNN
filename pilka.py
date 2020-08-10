from imutils import paths
from PIL import Image, ImageDraw
import os
import glob
import cv2 as cv
import numpy as np


for file in glob.glob("dataset/**/*.png", recursive=True):
    img = Image.open(file)
    new_img = Image.new("RGBA", img.size, "WHITE")
    new_img.paste(img, (0, 0), img)
    new_img.convert('RGB').save(file.replace("png", "jpg"), "JPEG")
    img = cv.imread(file.replace("png", "jpg"))
    img = np.array(img)
    img = 255 - img
    cv.imwrite(file.replace("png", "jpg"), img)
    files = glob.glob('dataset/**/*.png', recursive=True)

    for f in files:
        os.remove(f)
