#plot the results on local machine

import numpy as np
import matplotlib.pyplot as plt

pred = np.load('datasets/rings/Test_rings/test_predcraterdist_n30016.npy')
truth = np.load('datasets/rings/Test_rings/test_GTcraterdist_n30016_cutrad1.npy')

norm = False
nbins = 100

plt.hist(pred, nbins, range=[min(truth),max(truth)], normed=norm, label='pred')
#plt.hist(pred, nbins, normed=norm, label='pred')
plt.hist(truth, nbins, normed=norm, alpha=0.5, label='ground truth')
plt.legend()
plt.yscale('log')
plt.savefig('output_dir/images/cutrad1_bin100.png')
plt.show()
