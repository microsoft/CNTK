import urllib
import re
import os
import numpy as np
from PIL import Image

try:
    from urllib.request import urlretrieve, urlopen
except ImportError: 
    from urllib import urlretrieve, urlopen

def download_data(images_dir, link):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    images_html = urlopen(link).read().decode('utf-8')
    
    #looking for .jpg images whose names are numbers
    image_regex = "[0-9]+.jpg"
    
    #remove duplicates
    image_list = set(re.findall(image_regex, images_html))
    print("Starting download...")
    
    for image in image_list:
        filename = os.path.join(images_dir, image)
        if not os.path.isfile(filename):
            urlretrieve(link + image, filename)
        else:
            print("File already exists", filename)
    
    print("Data available at: ", images_dir)

#extract 64 x 64 patches from BSDS dataset
def prep_64(images_dir, patch_h, patch_w, train64_lr, train64_hr, tests):
    if not os.path.exists(train64_lr):
        os.makedirs(train64_lr)
    
    if not os.path.exists(train64_hr):
        os.makedirs(train64_hr)
    
    if not os.path.exists(tests):
        os.makedirs(tests)

    k = 0
    num = 0
    
    print("Creating 64 x 64 training patches and tests from:", images_dir)
    
    for entry in os.listdir(images_dir):
        filename = os.path.join(images_dir, entry)
        img = Image.open(filename)
        rect = np.array(img)
    
        num = num + 1
        
        if num % 50 == 0:
            img.save(os.path.join(tests, str(num) + ".jpg"))
            continue
    
        x = 0
        y = 0
    
        while(y + patch_h <= img.width):
            x = 0
            while(x + patch_w <= img.height):
                patch = rect[x : x + patch_h, y : y + patch_w]
                img_hr = Image.fromarray(patch, 'RGB')
                img_lr = img_hr.resize((patch_w // 2, patch_h // 2), Image.ANTIALIAS)
                img_lr = img_lr.resize((patch_w, patch_h), Image.BICUBIC)
                
                out_hr = os.path.join(train64_hr, str(k) + ".jpg")
                out_lr = os.path.join(train64_lr, str(k) + ".jpg")
                
                k = k + 1
                img_hr.save(out_hr)
                img_lr.save(out_lr)
                
                x = x + 42
            y = y + 42
    print("Done!")

#extract 224 x 224 and 112 x 112 patches from BSDS dataset
def prep_224(images_dir, patch_h, patch_w, train112, train224):
    if not os.path.exists(train112):
        os.makedirs(train112)
    
    if not os.path.exists(train224):
        os.makedirs(train224)

    k = 0
    print("Creating 224 x 224 and 112 x 112 training patches from:", images_dir)
    
    for entry in os.listdir(images_dir):
        filename = os.path.join(images_dir, entry)
        img = Image.open(filename)
        rect = np.array(img)
    
        x = 0
        y = 0
    
        while(y + patch_h <= img.width):
            x = 0
            while(x + patch_w <= img.height):
                patch = rect[x : x + patch_h, y : y + patch_w]
                img_hr = Image.fromarray(patch, 'RGB')
                img_lr = img_hr.resize((patch_w // 2, patch_h // 2), Image.ANTIALIAS)
                
                for i in range(4):
                    out_hr = os.path.join(train224, str(k) + ".jpg")
                    out_lr = os.path.join(train112, str(k) + ".jpg")
                
                    k = k + 1
                
                    img_hr.save(out_hr)
                    img_lr.save(out_lr)
                    img_hr = img_hr.transpose(Image.ROTATE_90)
                    img_lr = img_lr.transpose(Image.ROTATE_90)
                
                x = x + 64
            y = y + 64
    print("Done!")