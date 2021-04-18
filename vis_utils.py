import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import io
import itertools

import numpy as np
import cv2
import PIL

from torchvision import transforms


def get_prediction_plot(cfg, images, labels, pred_labels_idx):

    dim = {
        4: (2, 2),
        8: (4, 2),
        6: (3, 2),
        9: (3, 3),
        16: (4, 4),
    }

    n_rows, n_cols = dim[len(images)]

    # Create a figure to contain the plot.
    figure, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(20, 14))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i + j * n_rows

            image = images[idx]
            # true_label_idx = np.argmax(labels[idx])
            true_label = str(cfg.class_names[labels[idx]])
            pred_label = str(cfg.class_names[pred_labels_idx[idx]])

            axs[i, j].label_outer()
            axs[i, j].imshow(image)

            title = 'g_truth - ' + true_label + ', \n ' + 'pred- ' + pred_label
            color = ("green" if pred_label == true_label else "red")

            axs[i, j].set_title(title, color=color)

    plt.tight_layout()
    return figure


def get_conf_mat_plot(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # color = "white" if cm[i, j] > threshold else "black"
        color = 'green'
        plt.text(j, i, cm[i, j], horizontalalignment="center", verticalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)

    return figure


def get_roc_plot(fpr, tpr, roc_auc, class_names):
    
    figure= plt.figure(figsize=(12, 8))

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'khaki',
                              'purple', 'plum', 'sienna', 'tan', 'teal', 'olive',
                              'turquoise', 'tomato', 'crimson', 'orange', 'lime'])

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve for {class_name} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    return figure


def get_pr_plot(precision, recall, average_precision, class_names):
    
    figure= plt.figure(figsize=(12, 8))

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'khaki',
                              'purple', 'plum', 'sienna', 'tan', 'teal', 'olive',
                              'turquoise', 'tomato', 'crimson', 'orange', 'lime'])

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'PR curve for {class_name} (avg_prec = {average_precision[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

    return figure


def get_image_and_patches_plot(image, patches):

    n_rows, n_cols = 2, 3

    images = [image] + list(patches.values())
    names = ['original'] + list(patches.keys())

    # Create a figure to contain the plot.
    figure, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(20, 14))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i + j * n_rows

            image_to_show = images[idx]
            axs[i, j].label_outer()
            axs[i, j].imshow(image_to_show)

            title = names[idx]
            axs[i, j].set_title(title)

    plt.tight_layout()
    return figure


def draw_cam_on_image(img, cam_img):
    h, w, c = img.shape
    cam = cam_img - np.min(cam_img)
    cam = cam / np.max(cam)  # Normalize between 0-1
    cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    im_with_cam = np.float32(heatmap) + np.float32(img)
    im_with_cam = im_with_cam / np.max(im_with_cam)
    im_with_cam = np.uint8(255 * im_with_cam)
    return im_with_cam


def plot_to_image(fig):
    """Convert a Matplotlib figure to cv2 image"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_im = PIL.Image.open(buf)
    cv2_im = np.array(pil_im)
    return cv2_im[:,:,:3]


def image_to_tensor(image):
    return transforms.ToTensor()(image)


def tensor_to_image(tensor):
    pil_im = transforms.ToPILImage()(tensor.squeeze(0))
    im_cv2 = np.array(pil_im)
    return im_cv2
