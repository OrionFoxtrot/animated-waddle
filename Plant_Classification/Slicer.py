import PIL
from itertools import product
import os


def tile(filename, dir_in, dir_out, d):
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    name, ext = os.path.splitext(filename)
    img = PIL.Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

if __name__ == '__main__':
    print("Slicer")
    #tile("test.tif",r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\FullImage",r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\SlicedImage",1000)
    tile("test_0_0.tif",r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\SlicedImage",r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\EvenSmallerSlices",100)


    print("Complete")