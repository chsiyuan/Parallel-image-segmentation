from PIL import Image
import numpy as np
import StringIO
import argparse
import time


def img_to_txt(filename):
    im = Image.open(filename)
    a = np.array(im)
    height, width, _ = a.shape
    output_str = ''
    output_str += str(height) + " " + str(width) + '\n'
    for i in range(3):
        line = ''
        for h in range(height):
            for w in range(width):
                line += chr(a[h, w, i])
        output_str += line
    with open(filename[:-4]+'.txt', 'wb') as f:
        f.write(output_str)


def txt_to_img(filename):
    with open(filename, 'rb') as f:
        read_str = f.read()

    buf = StringIO.StringIO(read_str)
    firstline = buf.readline()
    height, width = firstline.split()
    height, width = int(height), int(width)
    array = [buf.read(height * width), buf.read(height * width), buf.read(height * width)]
    output_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        for h in range(height):
            for w in range(width):
                output_array[h, w, i] = ord(array[i][h * width + w])
    img = Image.fromarray(output_array, 'RGB')
    img.save('./images/output_'+time.strftime('%m-%d_%H-%M-%S')+'.jpg')
    #img.show()

# usage: read_jpeg -r <image path> or -w <txt path>
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--read', '-r', action='store', help='read image to output.txt')
    parser.add_argument('--write', '-w', action='store', help='convert txt to image and save it to out.jpg')
    args = parser.parse_args()
    if args.read:
        img_to_txt(args.read)
    if args.write:
        txt_to_img(args.write)
