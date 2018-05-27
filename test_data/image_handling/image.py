from PIL import Image


def bw_converter():
    # A converter built to translate an image to just black and white colors
    fp = "C:/Users/R/Desktop/test_data.png"    
    png = Image.open("C:/Users/R/Desktop/bw_germany_map.png")
    rgb_png = png.convert("RGB")
    x = png.getbbox()[2]
    y = png.getbbox()[3]
    png_new = Image.new("RGB", (x, y))
    
    png_new.save(fp, "png")
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # two loops to analyze each pixel of a given image
    # and strictly convert it to black and white
    for i in range(png.getbbox()[2]):
        for j in range(png.getbbox()[3]):
            r, g, b = rgb_png.getpixel((i, j))
            if (r > 120 and g > 120 and b > 120):                
                png_new.putpixel((i, j), white) 
                png_new.save(fp, "png")
                print("White")
            if (r == 0 and g == 0 and b == 0):
                png_new.putpixel((i, j), black) 
                png_new.save(fp, "png")
                print("Black")
    
    print("Old PNG", rgb_png.getpixel((41, 37))) 
    print("New PNG", png_new.getpixel((41, 37)))  


def bw_test(fp):
    # Function to test if pixels are in correct colors
    png = Image.open(fp)
    print("Path:", fp)
    print("Dimensions", "m:", png.getbbox()[3], "n:", png.getbbox()[2])    
    print("Pixelcount:", png.getbbox()[3] * png.getbbox()[2])
    print("Colors:", png.getcolors())

    
def extract_px_values(fp):
    # extract the pixel values of said black and white image and 
    # save them in a list to write it in txt file in data_write func
    white = (255, 255, 255)
    black = (0, 0, 0)
    test_data = []
    png = Image.open(fp)
    for i in range(png.getbbox()[2]):
        for j in range(png.getbbox()[3]):
            color = png.getpixel((i, j))
            if (color == white):                
                a = (i, j, "0", "white")
                print(a)
                test_data.append(a)
            else:                
                a = (i, j, "1", "black")
                print(a)
                test_data.append(a)
    return test_data


def data_write(fp, px_values):
    # simple func to store the pixel values, color names and depending category
    # in a easy to use txt
    # that txt should be the core of our learning data
    with open(fp, "a") as f:
        f.write("k: Category "+"c: Color"+"\n")
        f.write("m "+"n "+"k "+"c"+"\n")
        for i in px_values:
            f.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + " " + "\n")
    print("Ready", f.closed)

    
def main():
    # it s a main function
    fp = "C:/Users/R/Desktop/test_data.png"
    fp_data = "C:/Users/R/Desktop/test_data.txt"
    bw_test(fp)
    px_values = extract_px_values(fp)
    data_write(fp_data, px_values)    
    

if __name__ == '__main__':
    main()
