import requests
from PIL import Image
from math import radians, cos, sin, asin, sqrt
import io

KILOMETERS_TO_PIXELS = 33554432/40075
WIDTH_PIXELS = 640
HEIGHT_PIXELS = 416
WIDTH_IN_GRADS = 0.00687414424079777712
HEIGHT_IN_GRADS = 0.00446819375651850287


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    #print(km)
    square = (km*km)/2
    #print(square)
    return km


def download_by_coordinates(x, y):
    #x = 128.810
    #y = 54.889847
    url = "https://static-maps.yandex.ru/1.x/?lang=en_US&ll=" + str(x) + "," + str(y) + "&size=640,450&z=17&l=sat"
    res = requests.get(url=url)

    img = Image.open(io.BytesIO(res.content))
    return cut(img)
    #print(res.content)
    #return res.content


def parse_map(x1, y1, x2, y2):
    res = []
    width = int(haversine(x1, y1, x2, y1)/(WIDTH_PIXELS/KILOMETERS_TO_PIXELS))
    height = int(haversine(x1, y1, x1, y2)/(HEIGHT_PIXELS/KILOMETERS_TO_PIXELS))
    print(width, height)
    x = x1
    y = y1
    for i in range(width):
        res.append([])
        for j in range(height):
            img = download_by_coordinates(x+WIDTH_IN_GRADS/2, y+HEIGHT_IN_GRADS/2)
            res[i].append(Image.open(io.BytesIO(img)))
            #out = open("imgs/img"+str(i)+"_"+str(j)+".jpg", "wb")
            #out.write(img)
            #out.close()
            x += WIDTH_IN_GRADS
            #print(x, y, x+WIDTH_IN_GRADS/2, y-HEIGHT_IN_GRADS/2)
        x = x1
        y -= HEIGHT_IN_GRADS/1.65
        print(i)

    return res


def cut(img):
    border = (0, 0, 640, 416)  # left, up, right, bottom
    cropped2 = img.crop(border)
    img_byte_arr = io.BytesIO()
    cropped2.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def cut2(img):
    res = []
    total_index = 0
    # for i in range(1):
    #    for j in range(1):
    #img = Image.open("imgs/img0_33.jpg")
    for x in range(0, 640, 32):
        for y in range(0, 416, 32):
            border = (x, y, x + 32, y + 32)  # left, up, right, bottom
            cropped2 = img.crop(border)
            #cropped2.show()
            #cropped2.save("photo/cutPixels"+str(total_index)+".jpg")
            res.append(cropped2)
            total_index += 1
    return res


def create_big_image(res):
    print(res)
    total_height = 0
    total_width = 0
    for i in res:
        total_height += i[0].size[1]
    for i in res[0]:
        total_width += i.size[0]

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0

    for row in res:
        for im in row:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            #print(x_offset, y_offset)

        x_offset = 0
        y_offset += row[0].size[1]
    img_byte_arr = io.BytesIO()
    new_im.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def get_260_images(x, y):
    img_raw = download_by_coordinates(x, y)
    #img = Image.frombytes("RGB", (540, 640), img_raw)
    img = Image.open(io.BytesIO(img_raw))
    border = (0, 0, 640, 416)  # left, up, right, bottom
    cropped = img.crop(border)
    #print(cropped.size[0], cropped.size[1])
    cropped_list = cut2(cropped)
    for i in cropped_list:
        img_byte_arr = io.BytesIO()
        i.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        print(img_byte_arr)


#get_260_images(104.110513, 52.359349)







#create_big_image(parse_map(104.110513, 52.359349, 104.455896, 52.182851))
#create_big_image(parse_map(104.615912, 52.243023, 104.639966, 52.231430))
#create_big_image(parse_map(104.608487, 52.247439, 104.647154, 52.234793))


#create_big_image(parse_map(104.558411, 52.345928, 104.774704, 52.226971))


#create_big_image(parse_map(105.266356, 52.733527, 105.294937, 52.724408))


#x1 = 0.0
#x2 = 0.0068741442
#y = 0.0
#eps = 1e-15
#while haversine(x1, y, x2, y) < WIDTH_PIXELS/KILOMETERS_TO_PIXELS:
#    x2 += eps
#    print(haversine(x1, y, x2, y), WIDTH_PIXELS/KILOMETERS_TO_PIXELS)
#print("{0:.20f}".format(x2))

