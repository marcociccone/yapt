import numpy as np
import visdom
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools

from . import im_util
from collections import deque
from math import floor, sqrt
from .shared_config import SECTION_HEIGHT, IMAGE_WIDTH, DEFAULT_CONFIG, SECTION_INFO_FILENAME

MIN_SQUARE_SIZE = 3


class Visualizer():
    def __init__(self, viz=None, server=None, port=None,
                 id_exp=None, name_exp=None):
        self.saved = False
        if viz is None:
            viz = visdom.Visdom(server=server, port=port,
                                raise_exceptions=True)
        self.viz = viz
        self.name_exp = name_exp
        self.id_exp = id_exp

    def reset(self):
        self.saved = False

    def save_env(self, env):
        self.viz.save(env)

    def throw_visdom_connection_error(self):
        print("""\n\nCould not connect to Visdom server
              (https://github.com/facebookresearch/visdom) for
              displaying training progress.\nYou can suppress connection
              to Visdom using the option --display_id -1. To install
              visdom, run \n$ pip install visdom\n, and start the server
              by \n$ python -m visdom.server.\n\n""")
        exit(1)


class AdversarialVisualizer(Visualizer):

    def __init__(self, viz=None, **kwargs):
        super(AdversarialVisualizer, self).__init__(viz, **kwargs)

    def show_img(self, image, win='gen_img', normalize=False, env=None):
        self.viz.images(
            image, win=win,
            env=env, nrow=1,
            opts=dict(title='Generated Image',
                      caption='Generated Image',
                      height=200, width=200)
            )

    def show_images(self, image, adversarial, win='show_adv',
                    normalize=False, env=None, nshow=3):

        img, adv = image, adversarial
        if normalize:
            img = image / 255
            adv = adversarial / 255
        pert = adv - img
        pert = pert / abs(pert).max() * 0.2 + 0.5

        imgs = []
        for idx, (_img, _adv, _pert) in enumerate(zip(img, adv, pert)):
            if idx == nshow:
                break
            imgs.append((255 * np.stack([_img, _adv, _pert])).astype(np.uint8))
        imgs = np.concatenate(imgs, axis=0)

        self.viz.images(
            imgs, win=win,
            env=env, nrow=3,
            opts=dict(title='Adversarial Example',
                      caption='Original/Perturbed/Noise',
                      height=200*nshow, width=600)
            )

    def plot_scatter_importance(self, x, y, win, title,
                                flatten=True, env=None):
        shape = x.shape
        if flatten:
            x = x.flatten()
            y = y.flatten()

        data_std = np.stack([x, y], axis=-1)
        self.viz.scatter(
            data_std, win=win, env=env,
            opts={'title': title + str(shape), 'markersize': 6})

    def plot_heatmap_importance(self, x, y, key, idx, xmax=None, ymax=None,
                                colormap='RdBu', env=None):
        # TODO: check why it is so slow
        # Subplots to compare attack/importance on the same window
        trace_attack = go.Heatmap(z=x.cpu().numpy(), colorscale=colormap)
        trace_importance = go.Heatmap(z=y.cpu().numpy(), colorscale=colormap)
        fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
        fig.append_trace(trace_attack, 1, 1)
        fig.append_trace(trace_importance, 1, 2)

        name = key+'_attack/importance_feature_%d' % idx
        fig['layout'].update(title=name)
        offline.iplot(fig)
        self.viz.plotlyplot(fig, win=name, env=env)

        # self.viz.heatmap(
        #     x, win=name,
        #     opts={'title': name, 'colormap': colormap})

        # name = key+'_performance_importance_feature_%d' % idx
        # self.viz.heatmap(
        #     y, win=name, env=env,
        #     opts={'title': name, 'colormap': colormap})

    def plot_pruning_effect(self, data, key='', env=None):

        data_list = list(zip(*data))
        assert len(data_list) in (3, 5), "Check data len"

        x, y_clean, y_attack = data_list[:3]
        x = np.array(x)
        data_dict = dict(clean=np.array(y_clean),
                         attack=np.array(y_attack))

        if len(data_list) == 5:
            y_clean_new, y_attack_new = data_list[3:]
            # y_clean_new = np.array(y_clean_new)
            y_attack_new = np.array(y_attack_new)
            data_dict.update({'recomputed_attack': y_attack_new})

        for kk, vv in data_dict.items():
            name = '%s_%s' % (kk, key)
            if self.name_exp is not None:
                name = '%s_%s' % (self.name_exp, name)
            self.viz.line(
                X=x, Y=vv,
                name=name,
                win='plot_pruning_' + key, env=env,
                update='append',
                opts=dict(title='Pruning over units ' + key))

    def plot_pruning_stats(self, data, key='', env=None):

        steps = data['steps']
        for k, v in data.items():
            if k == 'steps':
                continue
            v = np.stack(v, axis=0)
            data_dict = {'median': np.median(v, axis=1),
                         'avg': np.mean(v, axis=1)}

            for kk, vv in data_dict.items():
                name = '%s_%s_%s' % (k, kk, key)
                if self.name_exp is not None:
                    name = '%s_%s' % (self.name_exp, name)
                self.viz.line(
                    X=steps, Y=vv,
                    name=name,
                    win=('plot_%s_pruning_' + key) % k, env=env,
                    update='append',
                    opts=dict(title=('%s over pruned units ' + key) % k))

    def plot_lines(self, data, name='',
                   win_name=None, title='', env=None):
        steps = data['steps']
        values = data['values']
        self.viz.line(
            X=steps, Y=values,
            name=name,
            win=win_name, env=env,
            update='append',
            opts=dict(title=title))

    def plot_loss_scatter(self, data, key='', env=None):

        steps = data['steps']

        shape = x.shape
        if flatten:
            x = x.flatten()
            y = y.flatten()

        data_std = np.stack([x, y], axis=-1)
        self.viz.scatter(
            data_std, win=win, env=env,
            opts={'title': title + str(shape), 'markersize': 6})

        for k, v in data.items():
            if k == 'steps':
                continue
            v = np.stack(v, axis=0)
            data_dict = {'median': np.median(v, axis=1),
                         'avg': np.mean(v, axis=1)}

            for kk, vv in data_dict.items():
                name = '%s_%s_%s' % (k, kk, key)
                if self.name_exp is not None:
                    name = '%s_%s' % (self.name_exp, name)
                self.viz.line(
                    X=steps, Y=vv,
                    name=name,
                    win=('plot_%s_pruning_' + key) % k, env=env,
                    update='append',
                    opts=dict(title=('%s over pruned units ' + key) % k))


    def plot_rgb_importance(self, x, y, key, idx):
        R = x * 255
        G = y * 255
        B = np.zeros_like(x)
        plots = np.stack([R, G, B], axis=0)

        self.viz.images(
            [plots], win=key+'_rgb_importance_feature_%d' % idx,
            nrow=1,
            opts=dict(title=key+'_rgb_importance_feature_%d' % idx,
                      height=400, width=400)
            )

    def plot_importance(self, x_stats, y_stats, env=None, heatmap=False):

        for key in x_stats.keys():
            x_mean = x_stats.running_mean(key)
            y_std = y_stats.running_std(key)
            assert x_mean.shape == y_std.shape, "Shapes must be the same"
            flatten = True if 'spatial' in key else False

            # Scatter plot importance proxy metrics
            win_name = key + '_importance_MEAN_ABS'
            self.plot_scatter_importance(
                x_mean, y_std, win=win_name, env=env,
                title=win_name, flatten=flatten)

            # Heatmap importance proxy metrics
            if heatmap and 'spatial' in key:
                for idx, (xx, yy) in enumerate(
                    zip(x_mean.permute([2, 0, 1]),
                        y_std.permute([2, 0, 1]))):

                    self.plot_heatmap_importance(
                        xx/x_mean.max(), yy/y_std.max(), key, idx)

    def plot_importance_perclass(self, x_stats_perclass, y_stats_perclass,
                                 suffix=''):

        assert len(x_stats_perclass) == len(y_stats_perclass), """
            nclasses should be the same for x and y staticts"""
        for idx in range(len(x_stats_perclass)):
            for key in x_stats_perclass[idx].keys():
                x_mean = x_stats_perclass[idx].running_mean(key)
                y_std = y_stats_perclass[idx].running_std(key)
                assert x_mean.shape == y_std.shape, "Shapes must be the same"
                flatten = True if 'spatial' in key else False

                win_name = key + '_importance_MEAN_ABS (per class)'
                self.plot_scatter_importance(
                    x_mean, y_std, win=win_name, title=win_name,
                    flatten=flatten, env='class %d' % idx)


class ArrayVisualizer(Visualizer):
    """
        Array visualization adapted from tensorboard beholder plugin
        https://github.com/tensorflow/tensorboard/blob/
        cb5a8cafdf29a0c2e0bd30f909e63ecdadbd5320/tensorboard/plugins/beholder/visualizer.py
    """
    def __init__(self, viz=None,
                 show_all=False, mode='values', scaling='layer',
                 window_size=15, **kwargs):

        super(ArrayVisualizer, self).__init__(viz, **kwargs)
        self.show_all = show_all
        self.mode = mode
        self.scaling = scaling
        self.window_size = window_size
        self.sections_over_time = deque([], DEFAULT_CONFIG['window_size'])

    def _reshape_conv_array(self, array, section_height, image_width):
        """Reshape a rank 4 array to be rank 2, where each column of
        block_width is a filter, and each row of block height is an
        input channel. For example:

        [[[[ 11,  21,  31,  41],
           [ 51,  61,  71,  81],
           [ 91, 101, 111, 121]],
          [[ 12,  22,  32,  42],
           [ 52,  62,  72,  82],
           [ 92, 102, 112, 122]],
          [[ 13,  23,  33,  43],
           [ 53,  63,  73,  83],
           [ 93, 103, 113, 123]]],
         [[[ 14,  24,  34,  44],
           [ 54,  64,  74,  84],
           [ 94, 104, 114, 124]],
          [[ 15,  25,  35,  45],
           [ 55,  65,  75,  85],
           [ 95, 105, 115, 125]],
          [[ 16,  26,  36,  46],
           [ 56,  66,  76,  86],
           [ 96, 106, 116, 126]]],
         [[[ 17,  27,  37,  47],
           [ 57,  67,  77,  87],
           [ 97, 107, 117, 127]],
          [[ 18,  28,  38,  48],
           [ 58,  68,  78,  88],
           [ 98, 108, 118, 128]],
          [[ 19,  29,  39,  49],
           [ 59,  69,  79,  89],
           [ 99, 109, 119, 129]]]]

           should be reshaped to:

           [[ 11,  12,  13,  21,  22,  23,  31,  32,  33,  41,  42,  43],
            [ 14,  15,  16,  24,  25,  26,  34,  35,  36,  44,  45,  46],
            [ 17,  18,  19,  27,  28,  29,  37,  38,  39,  47,  48,  49],
            [ 51,  52,  53,  61,  62,  63,  71,  72,  73,  81,  82,  83],
            [ 54,  55,  56,  64,  65,  66,  74,  75,  76,  84,  85,  86],
            [ 57,  58,  59,  67,  68,  69,  77,  78,  79,  87,  88,  89],
            [ 91,  92,  93, 101, 102, 103, 111, 112, 113, 121, 122, 123],
            [ 94,  95,  96, 104, 105, 106, 114, 115, 116, 124, 125, 126],
            [ 97,  98,  99, 107, 108, 109, 117, 118, 119, 127, 128, 129]]
        """

        # E.g. [100, 24, 24, 10]: this shouldn't be reshaped like normal.
        if array.shape[1] == array.shape[2] and array.shape[0] != array.shape[1]:
            array = np.rollaxis(np.rollaxis(array, 2), 2)

        block_height, block_width, in_channels = array.shape[:3]
        rows = []

        max_element_count = section_height * int(image_width / MIN_SQUARE_SIZE)
        element_count = 0

        for i in range(in_channels):
            rows.append(array[:, :, i, :].reshape(block_height, -1, order='F'))

            # This line should be left in this position. Gives it one extra
            # row.
            if element_count >= max_element_count and not self.show_all:
                break

            element_count += block_height * in_channels * block_width

        return np.vstack(rows)

    def _reshape_irregular_array(self, array, section_height, image_width):
        '''Reshapes arrays of ranks not in {1, 2, 4}
        '''
        section_area = section_height * image_width
        flattened_array = np.ravel(array)

        if not self.show_all:
            flattened_array = flattened_array[:int(
                section_area / MIN_SQUARE_SIZE)]

        cell_count = np.prod(flattened_array.shape)
        cell_area = section_area / cell_count

        cell_side_length = max(1, floor(sqrt(cell_area)))
        row_count = max(1, int(section_height / cell_side_length))
        col_count = int(cell_count / row_count)

        # Reshape the truncated array so that it has the same aspect ratio as
        # the section.

        # Truncate whatever remaining values there are that don't fit.
        # Hopefully it doesn't matter that the last few (< section
        # count) aren't there.
        section = np.reshape(flattened_array[:row_count * col_count],
                             (row_count, col_count))

        return section

    def _determine_image_width(self, arrays, show_all):
        final_width = IMAGE_WIDTH

        if show_all:
            for name, array in arrays.items():
                rank = len(array.shape)

                if rank == 1:
                    width = len(array)
                elif rank == 2:
                    width = array.shape[1]
                elif rank == 4:
                    width = array.shape[1] * array.shape[3]
                else:
                    width = IMAGE_WIDTH

                if width > final_width:
                    final_width = width

        return final_width

    def _determine_section_height(self, array, show_all):
        rank = len(array.shape)
        height = SECTION_HEIGHT

        if show_all:
            if rank == 1:
                height = SECTION_HEIGHT
            if rank == 2:
                height = max(SECTION_HEIGHT, array.shape[0])
            elif rank == 4:
                height = max(SECTION_HEIGHT, array.shape[0] * array.shape[2])
            else:
                height = max(SECTION_HEIGHT,
                             np.prod(array.shape) // IMAGE_WIDTH)

        return height

    def _arrays_to_sections(self, tensors_dict):
        """
        input: unprocessed dict of numpy arrays.
        returns: a dict containing the reshaped numpy arrays
                That needs to wait until after variance is computed.
        """
        sections = {}
        sections_to_resize_later = {}
        show_all = self.show_all
        image_width = self._determine_image_width(tensors_dict, show_all)

        for idx, (name, array) in enumerate(tensors_dict.items()):
            rank = len(array.shape)
            section_height = self._determine_section_height(array, show_all)

            if rank == 1:
                section = np.atleast_2d(array)
            elif rank == 2:
                section = array
            elif rank == 4:
                section = self._reshape_conv_array(
                    array, section_height, image_width)
            else:
                section = self._reshape_irregular_array(
                    array, section_height, image_width)

            # Only calculate variance for what we have to. In some cases
            # (biases), the section is larger than the array, so we
            # don't want to calculate variance for the same value over
            # and over - better to resize later. About a 6-7x speedup
            # for a big network with a big variance window.

            section_size = section_height * image_width
            array_size = np.prod(array.shape)

            if section_size > array_size:
                sections[name] = section
                sections_to_resize_later[name] = section_height
            else:
                sections[name] = im_util.resize(
                    section, section_height, image_width)

        self.sections_over_time.append(sections)
        if self.mode == 'variance':
            sections = self._sections_to_variance_sections(
                self.sections_over_time)

        # NOTE: no need to resize in visdom
        # for idx, height in sections_to_resize_later.items():
        #     sections[idx] = im_util.resize(sections[idx],
        #                                             height,
        #                                             image_width)
        return sections

    def _sections_to_variance_sections(self, sections_over_time):
        '''Computes the variance of corresponding sections over time.

        Returns:
          a dict of np arrays.
        '''
        variance_sections = {}
        for name in sections_over_time[0].keys():
            time_sections = [sections[name] for sections in sections_over_time]
            variance = np.var(time_sections, axis=0)
            variance_sections[name] = variance

        return variance_sections

    # def _sections_to_image(self, sections):
    #     padding_size = 5

    #     sections = im_util.scale_sections(sections, self.scaling)

    #     final_stack = [sections[0]]
    #     padding = np.zeros((padding_size, sections[0].shape[1]))

    #     for section in sections[1:]:
    #         final_stack.append(padding)
    #         final_stack.append(section)

    #     return np.vstack(final_stack).astype(np.uint8)

    # def _maybe_clear_deque(self):
    #     '''Clears the deque if certain parts of the config have changed.'''

    #     for config_item in ['values', 'mode', 'show_all']:
    #         if self.config[config_item] != self.old_config[config_item]:
    #             self.sections_over_time.clear()
    #             break

    #     self.old_config = self.config

    #     window_size = self.window_size
    #     if window_size != self.sections_over_time.maxlen:
    #         self.sections_over_time = deque(
    #             self.sections_over_time, window_size)

    def _save_section_info(self, tensors_dict, sections):

        self.infos = {}
        for name, section in sections.items():
            info = {}
            info['name'] = name
            info['shape'] = str(tensors_dict[name].shape)
            info['min'] = '{:.3e}'.format(section.min())
            info['mean'] = '{:.3e}'.format(section.mean())
            info['max'] = '{:.3e}'.format(section.max())
            info['range'] = '{:.3e}'.format(section.max() - section.min())
            info['height'] = section.shape[0]

            self.infos[name] = info

    def build_frame(self, tensors_dict):

        assert isinstance(tensors_dict, dict)

        # self._maybe_clear_deque()
        sections = self._arrays_to_sections(tensors_dict)
        self._save_section_info(tensors_dict, sections)

        return sections

    def visualize_heatmaps(self, tensors_dict, colormap='RdBu', env=None):
        sections = self.build_frame(tensors_dict)
        for name, array in sections.items():
            h, w = array.shape
            title = name + " " + self.infos[name]['shape']
            self.viz.heatmap(
                array, win=name,
                env=env,
                opts={'title': title, 'colormap': colormap,
                      'height': 200+2*h, 'width': 200+2*w}
            )
