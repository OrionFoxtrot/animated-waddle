import os
import numpy as np
from scipy.spatial.distance import cdist
import glob

from xml.dom.minidom import parse
import sys
from tqdm import tqdm


def main():
    working_dir = r'C:\Users\lohat_jay97s3\OneDrive\Computer Science'
    #get files
    files = []
    os.chdir(working_dir+'\IMAGES')
    for file in glob.glob("*.jpg"):
        base,_= os.path.splitext(file)
        files.append(base)
    print(f"Found {len(files)} files to parse")

    for file in tqdm(files):
        xmlfile = working_dir+'\XML\\'+file+'.xml'
        centroids = Get_Centroids(xmlfile)
        np.savetxt(working_dir+'\TXT\\'+file+'.txt',centroids,fmt= '%s')
    return 0


def Get_Centroids(xmlfile):
    dom = parse(xmlfile)

    filenames = dom.getElementsByTagName('filename')

    # should be just 1 filename
    assert len(filenames) == 1

    filename_element = filenames[0]

    filename = filename_element.firstChild.nodeValue

    # print('filename:', filename)

    bndboxes = dom.getElementsByTagName('bndbox')

    # print(f'found {len(bndboxes)} bndbox elements')

    centroids = []

    tags = ['xmin', 'xmax', 'ymin', 'ymax']

    for bndbox in bndboxes:

        values = dict()

        for tag in tags:
            elems = bndbox.getElementsByTagName(tag)

            # should just be 1 of each thing
            assert len(elems) == 1

            value = elems[0].firstChild.nodeValue

            values[tag] = int(value)

        # print(values)

        x = (values['xmax'] + values['xmin']) * 0.5
        y = (values['ymax'] + values['ymin']) * 0.5

        centroids.append((x, y))

    centroids = np.array(centroids,dtype=int)
    return centroids
    #
    # (base, _) = os.path.splitext(filename)
    #
    # centroids = np.array(centroids, dtype=np.uint8)
    # np.savetxt(base + ".txt", centroids, fmt='%s')
    # print(centroids)


if __name__ == '__main__':
    main()
