import os
import random
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from PIL import Image
# pylint: disable=not-context-manager
from collections import OrderedDict
from typing import Union


def permute(x: Union[torch.Tensor, np.ndarray], dims: Union[list, tuple]):
    """A wrapper function to permute a torch.Tensor or numpy.ndarray.

    Args:
        x (Union[torch.Tensor, numpy.ndarray]): tensor or array to be permuted
        dims (Union[list, tuple]): permutation of dims
    Returns:
        The permuted tensor (Numpy or Torch)
    """

    if isinstance(x, np.ndarray):
        return x.transpose(dims)
    elif isinstance(x, torch.Tensor):
        return x.permute(*dims)
    else:
        raise NotImplementedError("Only torch.Tensor and numpy.ndarray supported")


def swap_channels(x: Union[torch.Tensor, np.ndarray],
                  dim_channels: str = 'last'):
    """An helper function to fix image(s) shape by swapping the position of the
    channels as expected

    Args:
        x (Union[torch.Tensor, numpy.ndarray]): a 2-D, 3-D or 4-D tensor.
        dim_channels (str): where the channels of the image should be.
            It can be `first` or `last`. If channels are not in the right
            position, the tenosr is permuted.
    Returns:
        A new tensor of the image(s) with channels in the specified position.
    """

    shape = list(x.shape)
    x_new = x

    assert dim_channels in ('first', 'last'), \
        "Only first and last dimension for channels"

    assert len(shape) in (2, 3, 4), \
        "Single image or batch of images are only allowed. " \
        "Your tensor has shape {}".format(str(x.shape))

    # -- grey-scale single img
    if len(shape) == 2:
        if dim_channels == 'first':
            x_new = x.reshape(*(x.shape + [1]))
        else:
            x_new = x.reshape(*([1] + x.shape))

    # -- rgb or grey-scale single img
    elif len(shape) == 3:
        # -- move from first to last dim
        if shape[0] in (1, 3) and dim_channels == 'last':
            x_new = permute(x, (1, 2, 0))

        # -- move from last to first dim
        elif shape[2] in (1, 3) and dim_channels == 'first':
            x_new = permute(x, (2, 0, 1))

    # -- batch of rgb or grey-scale imgs
    elif len(shape) == 4:
        # -- move from first to last dim
        if shape[1] in (1, 3) and dim_channels == 'last':
            x_new = permute(x, (0, 2, 3, 1))

        # -- move from last to first dim
        elif shape[3] in (1, 3) and dim_channels == 'first':
            x_new = permute(x, (0, 3, 1, 2))

    return x_new


def prepare_dict_images(imgs_dict: dict, dim_channels: str = 'last'):
    """This function checks inside a dict that all tensors have the same batch
    size and channels are in the same position.

    Args:
        imgs_dict (dict): each key in the dictionary contains a batch of images.
        dim_channels (str): it can be `first` or `last`, swap channels to this
            position.

    Returns: a new dictionary

    """

    # get batch size
    bs = imgs_dict[list(imgs_dict.keys())[0]].shape[0]
    assert all([v.shape[0] == bs for k, v in imgs_dict.items()]), \
        "batch size should be the same for all images in the dictionary"

    # -- swap channels axis if necessary and move to numpy cpu
    new_imgs_dict = OrderedDict()
    for k, imgs in imgs_dict.items():
        new_imgs_dict[k] = swap_channels(imgs.cpu().numpy(), dim_channels)

    return new_imgs_dict


def write_video(frames, fname, fps=15, codec='mp4v', to_int=False):
    """Utility function to serialize a 4D numpy tensor to video and save it to
    filesystem.
    http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/

    Args:
        frames: 4D numpy array
        fname (str): output filename
        fps (int): frame rate of the ouput video
        codec (str): 4 digits string codec, default='H264' converted to RGB
        to_int (bool): if True, it converts from [0, 1.0] to [0, 255]
    """

    from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR
    fourcc = VideoWriter_fourcc(*codec)

    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)
    assert frames.ndim == 4

    h, w = frames.shape[1:3]
    writer = VideoWriter(
        fname, fourcc, fps, (w, h),
        isColor=frames.shape[-1] in (2, 3))

    for f in frames:
        if frames.shape[-1] == 3:
            f = cvtColor(np.float32(f), COLOR_RGB2BGR)
        if to_int:
            f = (f * 255.0)
        f = f.astype('uint8')
        writer.write(f)

    writer.release()


def hconcat_per_image(imgs_dict: dict, dim_channels: str = 'last',
                      stack: bool = False):
    """This function concats multiple images from a dictionary horizontally.
    Batch dimension remains independent.

    Args:
        imgs_dict (dict): each key contains a batch of images to concatenate
        dim_channels (str): `first` or `last`, swap channels to this position.
        stack (bool): stack concatenated images on batch dimension.

    Returns: a torch.Tensor or a list of concatenated images
    """

    # get batch size
    bs = imgs_dict[list(imgs_dict.keys())[0]].shape[0]
    new_imgs_dict = prepare_dict_images(imgs_dict, dim_channels)

    _new_batch = []
    for idx in range(bs):
        _imgs_to_compare = [swap_channels(img[idx], 'last')
                            for _, img in new_imgs_dict.items()]
        # -- stack images horizontally
        _imgs_to_compare = np.hstack(_imgs_to_compare)

        # -- make it RGB
        _imgs_to_compare = torch.tensor(_imgs_to_compare)
        _imgs_to_compare = swap_channels(_imgs_to_compare, 'first')
        if _imgs_to_compare.shape[0] == 1:
            _imgs_to_compare = _imgs_to_compare.repeat(3, 1, 1)

        # -- restore channels as expected
        _imgs_to_compare = swap_channels(_imgs_to_compare, dim_channels)
        # -- populate batch
        _new_batch.append(_imgs_to_compare)

    if stack:
        _new_batch = torch.stack(_new_batch)

    return _new_batch


def create_grid(imgs_dict: dict,
                nrow: int,
                dim_channels: str = 'last',
                kwargs_grid: dict = {}):

    """This is an helper function that arranges multiple images in a dictionary
    into a grid creating a single final image. It is useful to arrange together
    images referring to the same input, e.g. multiple reconstructions.

    Args:
        imgs_dict (dict): each key in the dictionary contains a batch of images.
        nrow (int): number of comparisons per row
        dim_channels (str): it can be `first` or `last`, swap channels to this
            position.
        kwargs_grid (dict): kwargs for torchvision.utils.make_grid
    """

    bs = imgs_dict[list(imgs_dict.keys())[0]].shape[0]
    new_imgs_dict = prepare_dict_images(imgs_dict, dim_channels)
    n = len(new_imgs_dict.keys()) + 1

    # -- stack on batch size: a comparison every n images
    _new_batch = []
    for idx in range(bs):
        # every n images there is a comparison
        for _, img in new_imgs_dict.items():
            _new_batch += [torch.tensor(img[idx])]

    # -- make_grid requires channels on the first dimension
    _new_batch = swap_channels(torch.stack(_new_batch), 'first')
    grid = torchvision.utils.make_grid(
        _new_batch, nrow=nrow, **kwargs_grid)
    # -- restore to the expected channel shape
    grid = swap_channels(grid, dim_channels)

    return grid


def hconcat_and_make_grid(imgs_dict,
                          dim_channels: str = 'last',
                          add_label: bool = True,
                          kwargs_grid: dict = {}):
    """This function combines hconcat_per_image and make_grid.
    First it concats horizontally the images to be compared into a single image,
    then it arranges the resulting images into a grid.

    Args:
        imgs_dict (dict): each key in the dictionary contains a batch of images.
        dim_channels (str): it can be `first` or `last`, swap channels to this
            position.
        add_label (bool): if true, it draws keys to the first series of images.
        kwargs_grid (dict): kwargs for torchvision.utils.make_grid
    """
    from torchvision.transforms import ToPILImage, ToTensor
    from PIL import ImageDraw

    _new_batch = hconcat_per_image(imgs_dict, dim_channels='first', stack=True)
    # arrange in a grid the concatenated images
    grid = torchvision.utils.make_grid(_new_batch, **kwargs_grid).byte()

    if add_label:
        # -- create label to draw from keys
        label = " - ".join(list(imgs_dict.keys()))
        grid = ToPILImage()(grid)
        draw = ImageDraw.Draw(grid)
        draw.text((2, 2), label, fill=(255, 0, 0))
        grid = ToTensor()(grid) * 255.

    # -- restore to the expected channel shape
    grid = swap_channels(grid, dim_channels)
    return grid


# --------------------------------------------------------------------
# Legacy code
# --------------------------------------------------------------------


def global_extrema(arrays):
    return min([x.min() for x in arrays]), max([x.max() for x in arrays])


def scale_sections(sections, scaling_scope):
    '''
    input: unscaled sections.
    returns: sections scaled to [0, 255]
    '''
    new_sections = []

    if scaling_scope == 'layer':
        for section in sections:
            new_sections.append(scale_image_for_display(section))

    elif scaling_scope == 'network':
        global_min, global_max = global_extrema(sections)

        for section in sections:
            new_sections.append(scale_image_for_display(
                section, global_min, global_max))
    return new_sections


def scale_image_for_display(image, minimum=None, maximum=None):
    image = image.astype(float)

    minimum = image.min() if minimum is None else minimum
    image -= minimum

    maximum = image.max() if maximum is None else maximum

    if maximum == 0:
        return image
    else:
        image *= 255 / maximum
        return image.astype(np.uint8)


def pad_to_shape(array, shape, constant=245):
    padding = []

    for actual_dim, target_dim in zip(array.shape, shape):
        start_padding = 0
        end_padding = target_dim - actual_dim

        padding.append((start_padding, end_padding))

    return np.pad(array, padding, mode='constant', constant_values=constant)


def apply_colormap(image, colormap='magma'):
    from tensorboard.plugins.beholder import colormaps
    if colormap == 'grayscale':
        return image
    cm = getattr(colormaps, colormap)
    return image if cm is None else cm[image]


def resize(image, height, width):
    if len(image.shape) == 2:
        image = image.reshape([image.shape[0], image.shape[1], 1])

    # resized = F.interpolate(image, [height, width], mode='nearest')
    image = F.to_pil_image(image)
    resized = F.resize(image, [height, width], interpolation=0)
    resized = np.array(resized)
    return resized


def read_image(filename):
    with Image.open(filename) as image_file:
        return np.array(image_file)


def write_image(array, filename):
    with Image.fromarray(array) as image:
        image.save(filename, 'png')


def get_image_relative_to_script(filename):
    script_directory = os.path.dirname(__file__)
    filename = os.path.join(script_directory, 'resources', filename)

    return read_image(filename)


def make_rows_for_comparison(x, x_rec, nrows=4):
    assert (x.shape == x_rec.shape), \
        "x and x_rec should have the same shape"

    comparisons = []
    img_per_row = 8
    n = min(x.size(0), img_per_row)
    for row in range(nrows):
        end = min((row + 1) * n, x.size(0))

        # -- Only get n images
        original_imgs = x[row * n: end]
        rec_imgs = x_rec[row * n: end]
        comparisons.append(original_imgs)
        comparisons.append(rec_imgs)

    comparisons = torch.cat(comparisons, dim=0)
    comparisons = comparisons.expand(-1, 3, -1, -1)

    return comparisons


def tensor2im(x, imtype=np.uint8):
    """It converts a Tensor into an image array (numpy)

    Args:
        x: a torch.Tensor image.
        imtype: the desired type of the converted numpy array

    Returns: The image numpy arrray.
    """

    if isinstance(x, torch.Tensor):
        image_tensor = x.data
    else:
        return x
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def confusion_matrix_fig(cm, labels, normalize=False):

    from textwrap import wrap
    from itertools import product

    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = ['\n'.join(wrap(l, 40)) for l in labels]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")

    return fig



class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

