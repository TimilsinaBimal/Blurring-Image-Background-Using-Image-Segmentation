import matplotlib.pyplot as plt


def display(display_list, iou=0):
    plt.figure(figsize=(10, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i] + "  IOU: %.2f" % iou)
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def display_blur(display_list, iou=0):
    plt.figure(figsize=(10, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask', 'Blur Image']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i] + "  IOU: %.2f" % iou)
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
