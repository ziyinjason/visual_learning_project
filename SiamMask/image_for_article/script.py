import numpy as np
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import cv2
import pyvips

from util import *

Image.MAX_IMAGE_PIXELS = None
RESIZE_RATIO = 0.25

def onclick(event):
    global ax_pos_list
    global click_pos_list
    if ax1.in_axes(event):
        ax_pos = ax1.transAxes.inverted().transform((event.x, event.y))
        click_pos = [1. - ax_pos[1], ax_pos[0]]
        print (click_pos)
        ax_pos_list.append(ax_pos)
        click_pos_list.append(click_pos)

if __name__ == '__main__':
    
    root = tk.Tk()
    root.withdraw()

    print ("Please select the STITCHED image.")
    stitched_path = filedialog.askopenfilename()
    print ("Please select the ANNOTATION image.")
    crack_path = filedialog.askopenfilename()
    print ("Please select the ORIGINAL image.")
    orig_path = filedialog.askopenfilename()
    print ("Loading stitched image from: %s" % stitched_path)
    stitched_img = Image.open(stitched_path)
    print ("Loading annotation image from: %s" % crack_path)
    crack_img = Image.open(crack_path)
    print ("Loading original image from: %s" % orig_path)
    #orig_img = Image.open(orig_path) # No change
    #orig_img = Image.open(orig_path).transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT) # Folder 1
    orig_img = Image.open(orig_path).transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM) # Folder 2
    #orig_img = Image.open(orig_path).transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT) # Folder 3
    #orig_img = Image.open(orig_path).transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT) # Folder 4

    print ("Resizing images with ratio %f" % RESIZE_RATIO)
    stitched_width, stitched_height = stitched_img.size
    resized_stitched_img = stitched_img.resize((int(stitched_width*RESIZE_RATIO), int(stitched_height*RESIZE_RATIO)), Image.ANTIALIAS)
    orig_width, orig_height = orig_img.size
    resized_orig_img = orig_img.resize((int(orig_width*RESIZE_RATIO), int(orig_height*RESIZE_RATIO)), Image.ANTIALIAS)

    print ("Plotting images")
    global ax1
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(resized_stitched_img)
    ax1.set_title('Please select the bounding box \n for the rough location of the original image')
    ax1.set_axis_off()
    ax2.imshow(resized_orig_img)
    ax2.set_title('Original image')
    ax2.set_axis_off()

    global ax_pos_list
    global click_pos_list
    ax_pos_list = []
    click_pos_list = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    bbox = None
    while True:
        plt.draw()
        plt.pause(0.1)
        if len(click_pos_list) == 2:
            plt.close()
            stitched_img_np = np.asarray(resized_stitched_img, dtype=np.uint8)
            orig_img_np = np.asarray(resized_orig_img, dtype=np.uint8)
            height, width = stitched_img_np.shape[0:2]
            
            y1 = int(click_pos_list[0][0] * height)
            x1 = int(click_pos_list[0][1] * width)
            y2 = int(click_pos_list[1][0] * height)
            x2 = int(click_pos_list[1][1] * width)
            cropped_img_np = stitched_img_np[y1:y2, x1:x2]

            print ("Extracting SIFT features...")
            (kpsA, featuresA) = detectAndDescribe(cropped_img_np)
            (kpsB, featuresB) = detectAndDescribe(orig_img_np)

            print ("Matching keypoints...")
            # match features between the two images
            M = matchKeypoints(kpsA, kpsB,
                featuresA, featuresB)
    
            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            (matches, H, status) = M
            
            H_inv = inv(H)
            translation_mat = np.identity(3)
            translation_mat[0][2] = x1
            translation_mat[1][2] = y1
            scale_mat = np.identity(3)
            scale_mat[0][0] = 1. / RESIZE_RATIO
            scale_mat[1][1] = 1. / RESIZE_RATIO
            H_inv = translation_mat @ H_inv
            result = cv2.warpPerspective(orig_img_np, H_inv,
                (stitched_img_np.shape[1], stitched_img_np.shape[0]))
            result_mask = cv2.warpPerspective(np.ones_like(orig_img_np), H_inv,
                (stitched_img_np.shape[1], stitched_img_np.shape[0]))
            blended = stitched_img_np * (1-result_mask) + result

            H = H @ inv(translation_mat)
            H = scale_mat @ H @ inv(scale_mat)
            np.save(orig_path[:-4], H)

            vis = drawMatches(cropped_img_np, orig_img_np, kpsA, kpsB, matches, status)
            numpy2vips(vis).write_to_file("matching.png")
            numpy2vips(blended).write_to_file("overlay.png")
            numpy2vips(stitched_img_np).write_to_file("orig.png")

            stitched_img_np = np.asarray(stitched_img, dtype=np.uint8)
            crack_img_np = np.asarray(crack_img, dtype=np.uint8)
            orig_img_np = np.asarray(orig_img, dtype=np.uint8)
            warped_crack = cv2.warpPerspective(crack_img_np, H,
                (crack_img_np.shape[1] + orig_img_np.shape[1], crack_img_np.shape[0] + orig_img_np.shape[0]))[0:orig_img_np.shape[0], 0:orig_img_np.shape[1]]

            # Visualization only
            warped_stitched = cv2.warpPerspective(stitched_img_np, H,
                (stitched_img_np.shape[1] + orig_img_np.shape[1], stitched_img_np.shape[0] + orig_img_np.shape[0]))
            vips_image = numpy2vips(warped_stitched[0:orig_img_np.shape[0], 0:orig_img_np.shape[1]])
            vips_image.write_to_file("corresponding.png")
            overlayed_crack = (1 - np.expand_dims(warped_crack, axis=-1) / 255) * orig_img_np + np.expand_dims(warped_crack, axis=-1)
            vips_overlayed_crack = numpy2vips(overlayed_crack)
            vips_overlayed_crack.write_to_file("overlay_crack.png")

            #warped_crack = np.asarray(Image.fromarray(warped_crack).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)) # Folder 1
            warped_crack = np.asarray(Image.fromarray(warped_crack).transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)) # Folder 2
            #warped_crack = np.asarray(Image.fromarray(warped_crack).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)) # Folder 3
            #warped_crack = np.asarray(Image.fromarray(warped_crack).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)) # Folder 4
            vips_crack = numpy2vips(np.expand_dims(warped_crack, axis=-1))
            vips_crack.write_to_file(orig_path[:-4]+"_label.png")

            ax_pos_list.clear()
            click_pos_list.clear()
            break
        