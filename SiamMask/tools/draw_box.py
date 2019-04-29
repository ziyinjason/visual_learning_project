import cv2
import numpy as np
import copy 

def draw_bbox(im, bbox, color, width):
    """Draw bounding box on given image.
    Args:
    im: H*W*3
    bbox: 8, nparray [x_tl,y_tl,x_tr,y_tr,x_bl,y_bl,x_br,y_br]
          4, nparray [x_tl,y_tl,x_br,y_br]
    color: (0,0,0) - (255,255,255)
    width: width of the bounding box

    Returns:
    None
    """
    im1=copy.deepcopy(im)
    im1 = np.ascontiguousarray(im1, dtype=np.uint8)
    if len(bbox) == 8:
        for i in range(0, 4):
            cv2.line(im1, (bbox[2 * i], bbox[2 * i + 1]),
                     (bbox[(2 * i + 2) % 8], bbox[(2 * i + 3) % 8]), color,
                     width)
    elif len(bbox) == 4:
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        cv2.line(im1, (x0, y0), (x0, y1), color, width)
        cv2.line(im1, (x0, y0), (x1, y0), color, width)
        cv2.line(im1, (x1, y1), (x0, y1), color, width)
        cv2.line(im1, (x1, y1), (x1, y0), color, width)

    return im1