import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
if __name__ == '__main__':
    fig_dir = '/home/eicg/Documents/Lin/Understanding Long-term Activity/eccv22_img2vid/figs'


    # results = [ 60.5,60.3 ,62.0 ,61.5,60.9,60.0,62.2,62.2 ,60.9 ,58.7,60.5 ]
    results1 = [ 60.5,60.3 ,62.0 ,61.5,60.9, 60.0 ]
    # results2 = [ 99.2, 99.3 ,99.1 ,99.2, 99.2 ]
    results3 = [ 69.8, 72.1,72.6, 72.3, 72.1, 72.1 ]

    # n_iters = list(range(1, 12))
    n_iters = list(range(1, 7))

    # major_ticks = np.arange(58, 63, 1)
    # minor_ticks = np.arange(58, 63, 0.1)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot( n_iters, results1)
    # plt.plot( n_iters, results2)
    plt.plot( n_iters, results3)
    plt.xlabel('# iterations')
    plt.ylabel('Acc')
    # plt.ylim(58, 63)

    ax.set_xticks( n_iters)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.legend(['E->H', 'B->U' ])
    # plt.legend(['E->H', 'S->U', 'B->U' ])

    plt.grid(True)
    plt.show()

    fig.savefig(osp.join(fig_dir, 'n_iters_vs_performance_e2h_rebuttal_v3.svg'))

