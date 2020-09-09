import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def plot_past_future(past_y, pred_y, true_y, plot_dir, n, yrange=None, title='', show=True):
    '''

    :param past_y: n x past_window
    :param pred_y: n x future_window
    :param true_y: n x future_window
    :param plot_dir: directory to save the printed plot at
    :param n: index of series (int) or list of series ([int]) indeces to plot
    :return:
    '''

    assert past_y.shape[0] == pred_y.shape[0], 'Series dimensions mismatch'
    assert pred_y.shape[0] == true_y.shape[0], 'Series dimensions mismatch'

    tlen = past_y.shape[1] + pred_y.shape[1]
    if isinstance(n, int):
        _plot_save(n, past_y, plot_dir, pred_y, show, title, tlen, true_y, yrange)
    elif isinstance(n, list) or isinstance(n, tuple):
        for nn in tqdm(n, desc='plot past-future'):
            _plot_save(nn, past_y, plot_dir, pred_y, show, title, tlen, true_y, yrange)
    else:
        raise Exception('plot_multi_horizon: incorrect series indeces')


def _plot_save(n, past_y, plot_dir, pred_y, show, title, tlen, true_y, yrange):
    plt.plot(list(range(past_y.shape[1])), past_y[n, :].flatten(), color='b')
    plt.plot(list(range(tlen - pred_y.shape[1], tlen)), true_y[n, :].flatten(), color='g')
    plt.plot(list(range(tlen - pred_y.shape[1], tlen)), pred_y[n, :].flatten(), color='r')
    axes = plt.gca()
    if yrange is None:
        r = np.max(np.abs(true_y[n, :] - pred_y[n, :]))
        m = np.concatenate((true_y[n, :], pred_y[n, :], past_y[n, :]))
        yrange = (np.min(m) - r, np.max(m) + r)
    axes.set_ylim(yrange)
    plt.legend(['past', 'true', 'pred'])
    plt.title(title)

    if plot_dir is not None:
        fig_file = os.path.join(plot_dir, 'past_future_fig_' + str(n) + '.png')
        plt.savefig(fig_file, format='png', dpi=640)
    if show:
        plt.show()
    plt.close()


def plot_partial_full(full_y, past_y, future_y, n, title=''):
    assert full_y.shape[0] == past_y.shape[0], 'Series dimensions mismatch'
    assert full_y.shape[0] == future_y.shape[0], 'Series dimensions mismatch'

    if isinstance(n, int):
        _plot_pf(full_y, past_y, future_y, n, title)
    elif isinstance(n, list) or isinstance(n, tuple):
        for nn in n:
            _plot_pf(full_y, past_y, future_y, nn, title)


def _plot_pf(full_y, past_y, future_y, n, title):
    tlen = full_y.shape[1]
    plt.plot(list(range(full_y.shape[1])), full_y[n, :].flatten(), color='b', linewidth=3, linestyle='-.')
    plt.plot(list(range(past_y.shape[1])), past_y[n, :].flatten(), color='g')
    plt.plot(list(range(tlen - future_y.shape[1], tlen)), future_y[n, :].flatten(), color='r')
    plt.legend(['full', 'past', 'future'])
    plt.title(title)
    plt.show()
    plt.close()


def plot_stack(y, yhat, block_outs, n, labels=None, block_plot_cnt=4, plot_dir=None, show=True):

    if len(block_outs) == 0:
        return

    if isinstance(n, int):
        _stack_plot(n, y[n], yhat[n],
                    {k: v[n] for k, v in block_outs.items()},
                    block_plot_cnt, labels, plot_dir, show)
    elif isinstance(n, list) or isinstance(n, tuple):
        for nn in tqdm(n, desc='plot stack'):
            _stack_plot(nn, y[nn], yhat[nn],
                        {k: v[nn] for k, v in block_outs.items()},
                        block_plot_cnt, labels, plot_dir, show)


def _stack_plot(n, y, y_hat, others, k, lables=None, plot_dir=None, show=True):
    plt.figure(figsize=[20, 5])
    yy_hat = 0
    for i, (name, value) in enumerate(others.items()):
        yy_hat += value
        plt.subplot(1, k + 1, i + 2)
        plt.plot(value)
        plt.grid()
        if lables is not None:
            plt.legend([lables[i]])
        if i == k:
            break

    plt.subplot(1, k + 1, 1)
    plt.plot(yy_hat, color='r')
    plt.plot(y, color='g')
    plt.grid()
    plt.legend(['Predicted', 'True'])

    if plot_dir is not None:
        fig_file = os.path.join(plot_dir, 'stack_out_fig_' + str(n) + '.png')
        plt.savefig(fig_file, format='png', dpi=640)
    if show:
        plt.show()
    plt.close()
