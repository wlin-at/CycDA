import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
if __name__ == '__main__':
    fig_dir = '/home/eicg/Documents/Lin/Understanding Long-term Activity/eccv22_img2vid/figs'

    percentage_list = list(np.arange(0.1, 1.1, 0.1))
    ps_acc_list = [ 0.857,0.769, 0.681, 0.635, 0.596, 0.57, 0.529, 0.501,0.468, 0.427 ]
    result_acc_list = [ 0.2256, 0.3615, 0.4949, 0.4974, 0.4872, 0.5436, 0.5564,  0.5333,  0.5282, 0.5513   ]

    major_ticks = np.arange(0, 0.91, 0.1)
    minor_ticks = np.arange(0, 0.91, 0.02)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1,1,1)

    plt.plot( percentage_list, ps_acc_list )
    plt.plot(percentage_list, result_acc_list)

    plt.xlabel('p x 100%')
    plt.ylabel('Acc')

    ax.set_xticks( percentage_list )

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # ax.grid(which='both')

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.legend(['video pseudo labels', 'spatio-temporal learning'])

    plt.grid(True)
    plt.show()

    fig.savefig(osp.join(fig_dir, 'thresh_p_vs_performance.svg'))



