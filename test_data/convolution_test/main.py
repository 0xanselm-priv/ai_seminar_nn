from PIL import Image
import sys
import numpy as np
import scipy.signal


class Convolution(object):

    def __init__(self, params):
        self.main()

    def get_kernel(self, kernel):
        if kernel == "horizontal":
            return np.fliplr(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        elif kernel == "vertical":
            return np.fliplr(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        elif kernel == "edges":
            return np.fliplr(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        elif kernel == "identity":
            return np.fliplr(np.array([[0,0,0], [0,1,0], [0,0,0]]))
        else:
            print("Wrong Kernel.")
            sys.exit()

    def array2img(self, array):
        conv = Image.fromarray(array)
        conv.show()
        print("Output size", conv.size)

    def load_image(self):
        img = Image.open("frankfurt.jpg")
        img = img.convert("L")  # convert rgb to gra)
        img.show()
        img.save("ffm_gray.jpg")
        print("Input size", img.size)
        edges_kernel = self.get_kernel("edges")
        hori_kernel = self.get_kernel("horizontal")
        verti_kernel = self.get_kernel("vertical")

        edges_kernel_conv = scipy.signal.convolve2d(
            img, edges_kernel, mode="same", boundary="symm")
        hori_kernel_conv = scipy.signal.convolve2d(
            img, hori_kernel, mode="same", boundary="symm")
        verti_kernel_conv = scipy.signal.convolve2d(
            img, verti_kernel, mode="same", boundary="symm")
        
        self.array2img(edges_kernel_conv)
        self.array2img(hori_kernel_conv)
        self.array2img(verti_kernel_conv)
        
    def main(self):
        self.load_image()


if __name__ == "__main__":
    a = ""
    Convolution(a)
