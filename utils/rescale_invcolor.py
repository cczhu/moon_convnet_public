#######################################
#rescale and invert color (if desired)#
########################################################################

import numpy as np

def rescale_and_invcolor(data, inv_color, rescale):
    #rescaling and inverting images - assumes that images are already normalized between 0-1.
    #https://www.mathworks.com/help/vision/ref/contrastadjustment.html
    #Since maxpooling is used, we want the interesting stuff (craters) to be 1, not 0.
    #But ignore null background pixels, keep them at 0.

    low, hi = 0.1, 1        #low, hi image rescaling values

    for img in data:
        if inv_color == 1:
            img[img > 0.] = 1. - img[img > 0.]
        if rescale == 1:
            minn, maxx = np.min(img[img>0]), np.max(img[img>0])
            img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
    return data

