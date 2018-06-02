from PIL import Image
import os
import time  

def bw_converter(fp, fp_raw):
    # A converter built to translate an image to just black and white colors  
    png = Image.open(fp_raw)
    rgb_png = png.convert("RGB")
    x = png.getbbox()[2]
    y = png.getbbox()[3]
    png_new = Image.new("RGB", (x, y))
    
    png_new.save(fp, "png")
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # two loops to analyze each pixel of a given image
    # and strictly convert it to black and white
    progress = 0
    for i in range(png.getbbox()[2]):
        for j in range(png.getbbox()[3]):
            r, g, b = rgb_png.getpixel((i, j))
            progress += 1
            if (r > 120 and g > 120 and b > 120):                
                png_new.putpixel((i, j), white) 
                png_new.save(fp, "png")
                print(round((progress/(x*y))*100,3),"%")
            if (r == 0 and g == 0 and b == 0):
                png_new.putpixel((i, j), black) 
                png_new.save(fp, "png")
                print(round((progress/(x*y))*100,3),"%")
    
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
    x = png.getbbox()[2]
    y = png.getbbox()[3]
    progress = 0
    for i in range(png.getbbox()[2]):
        for j in range(png.getbbox()[3]):
            progress += 1
            color = png.getpixel((i, j))
            if (color == white):                
                a = (i, j, "0", "white")
                print(round((progress/(x*y))*100,3),"%")
                test_data.append(a)
            else:                
                a = (i, j, "1", "black")
                print(round((progress/(x*y))*100,3),"%")
                test_data.append(a)
    return test_data

def file_delete(fp_data):  
    ## Try to delete the file ##
    try:
        os.remove(fp_data)
    except:
        print ("Test Data File not found. Creating new")


def data_write(fp_data, px_values):
    # simple func to store the pixel values, color names and depending category
    # in a easy to use txt
    # that txt should be the core of our learning data
    file_delete(fp_data)
    with open(fp_data, "w") as f:
        f.write("k: Category "+"c: Color"+"\n")
        f.write("m "+"n "+"k "+"c"+"\n")
        for i in px_values:
            f.write(str(i[0]) + "_" + str(i[1]) + "_" + str(i[2]) + "_" + str(i[3]) + "_" + ";")
    print("Finished", f.closed)

    
def main():
    # it s a main function
    a = time.time()
    fp_raw = "bw_germany_map.png"
    fp = "test_data.png"
    fp_data = "test_data.txt"    
    bw_converter(fp, fp_raw)
    bw_test(fp)
    px_values = extract_px_values(fp)
    data_write(fp_data, px_values)
    b = time.time() 
    print("All processes finished in", round(b-a, 3), "seconds")
    

if __name__ == '__main__':
    main()
