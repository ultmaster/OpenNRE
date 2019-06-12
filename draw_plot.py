import sklearn.metrics
import matplotlib

# Use 'Agg' so this program could run on a remote server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

result_dir = './test_result'


def main():
    models = sys.argv[1:]
    min_y = 1.0
    for model in models:
        x = np.load(os.path.join(result_dir, model + '_x' + '.npy'))
        y = np.load(os.path.join(result_dir, model + '_y' + '.npy'))
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)

        # plt.plot(x, y, lw=2, label=model + '-auc='+str(auc))
        plt.plot(x, y, lw=2, label=model)
        p_01 = p_03 = 0
        for recall, prec in zip(x, y):
            if recall >= 0.1:
                p_01 = max(p_01, prec)
            if recall >= 0.3:
                p_03 = max(p_03, prec)
            if recall <= 0.4:
                min_y = min(min_y, prec)

        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(f1))
        print('    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300],
                                                                        (y[100] + y[200] + y[300]) / 3))
        print('    P@10%: {} | P@30%: {}'.format(p_01, p_03))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([min_y - 0.05, 1.05])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve'))


if __name__ == "__main__":
    main()
