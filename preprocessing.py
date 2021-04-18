import cv2
import numpy as np
from vis_utils import get_image_and_patches_plot, plot_to_image
from color_constancy_alb import shade_of_gray_cc
from matplotlib import pyplot as plt


def get_image_patches(image, patch_size=(256, 256)):
    h, w, c = image.shape
    ph, pw = patch_size

    if h < ph or w < pw:
        raise IndexError('Image dim is smaller that requested patch')

    # Center crop corner coordinates
    hc = int((h-ph)/2)
    wc = int((w-pw)/2)

    patches = {
        'tl': image[:ph, :pw, :],
        'tr': image[:ph:, w-pw:w, :],
        'bl': image[h-ph:h, :pw, :],
        'br': image[h-ph:h, w-pw:w, :],
        'center': image[hc:hc+ph, wc:wc+pw, :]
    }
    return patches


def test_patching():
    
    image_path = 'data/testo.jpg'
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    patches = get_image_patches(img)
    figure = get_image_and_patches_plot(cv2.resize(img, (256, 256)), patches)

    combined_image = plot_to_image(figure)
    cv2.imwrite('patches.jpg', combined_image)
    cv2.imwrite('original.jpg', img)


def fill_holes(im_th):
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh


def get_bbox_limits(bw):
    contours, hierarchy = cv2.findContours(
        bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) == 1:
        cnt = contours[0]
    else:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def test_zoom(img):

    bw = binarize(img)
    filled = fill_holes(bw)
    x, y, w, h = get_bbox_limits(filled)

    roi = img[y:y+h, x:x+w, :]

    # Check whether the intensity differs
    ratios = img.sum(axis=0).sum(axis=0) / roi.sum(axis=0).sum(axis=0)

    if np.all(np.abs(1-ratios) < 0.05):
        result = roi
    else:
        # mean intensity inside bbox differss
        result = img

    plt.plot(),
    plt.xticks([]), plt.yticks([])

    plt.imshow(img, vmin=0, vmax=255)
    plt.show()

    plt.imshow(bw, 'gray', vmin=0, vmax=255)
    plt.show()

    plt.imshow(filled, 'gray', vmin=0, vmax=255)
    plt.show()

    plt.imshow(roi, vmin=0, vmax=255)
    plt.show()

    return result


def main():

    image_path = 'data/zoom_me.jpg'
    image_path = 'data/testo.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    # test_zoom(img)

    plt.plot(),
    plt.xticks([]), plt.yticks([])

    cc = shade_of_gray_cc(img)
    dbg_im = np.hstack((img, cc))

    plt.imshow(cv2.cvtColor(dbg_im, cv2.COLOR_RGB2BGR), vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    main()
