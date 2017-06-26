######################
#custom loss function#
########################################################################

import numpy as np
from skimage.feature import match_template
import cv2

def template_match_target(target, match_thresh2=50, minrad=3, maxrad=75):
    #Match Threshold (squared)
    # for template matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, remove (x2,y2,r2) circle (it is a duplicate).
    # for predicted target -> csv matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, positive detection
    
    # minrad/maxrad are the radii to search over during template matching
    # hyperparameters, probably don't need to change
    ring_thickness = 2       #thickness of rings for the templates. 2 seems to work well.
    template_thresh = 0.5    #0-1 range, if template matching probability > template_thresh, count as detection
    target_thresh = 0.1      #0-1 range, pixel values > target_thresh -> 1, pixel values < target_thresh -> 0
    
    # target - can be predicted or ground truth
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0
    
    radii = np.linspace(minrad,maxrad,maxrad-minrad,dtype=int)
    coords = []        #coordinates extracted from template matching
    for r in radii:
        # template
        n = 2*(r+ring_thickness+1)
        template = np.zeros((n,n))
        cv2.circle(template, (r+ring_thickness+1,r+ring_thickness+1), r, 1, ring_thickness)
        
        # template match
        result = match_template(target, template, pad_input=True)   #skimage
        coords_r = np.asarray(zip(*np.where(result > template_thresh)))
        
        # store x,y,r
        for c in coords_r:
            coords.append([c[1],c[0],r])

    # remove duplicates from template matching at neighboring radii/locations
    coords = np.asarray(coords)
    i, N = 0, len(coords)
    while i < N:
        diff = (coords - coords[i])**2
        diffsum = np.asarray([sum(x) for x in diff])
        index = (diffsum == 0)|(diffsum > match_thresh2)
        coords = coords[index]
        N, i = len(coords), i+1

    return coords


def template_match_target_to_csv(target, csv_coords, minrad=3, maxrad=75):
    #Match Threshold (squared)
    # for template matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, remove (x2,y2,r2) circle (it is a duplicate).
    # for predicted target -> csv matching, if (x1-x2)^2 + (y1-y2)^2 + (r1-r2)^2 < match_thresh2, positive detection
    match_thresh2 = 50
    
    #get coordinates from template matching
    templ_coords = template_match_target(target, match_thresh2, minrad, maxrad)

    # compare template-matched results to "ground truth" csv input data
    N_match = 0
    csv_duplicate_flag = 0
    N_csv, N_templ = len(csv_coords), len(templ_coords)
    for tc in templ_coords:
        diff = (csv_coords - tc)**2
        diffsum = np.asarray([sum(x) for x in diff])
        index = (diffsum == 0)|(diffsum > match_thresh2)
        N = len(np.where(index==False)[0])
        if N > 1:
            #print "multiple matches found in csv file for template matched crater ", tc, " :"
            #print csv_coords[np.where(index==False)]
            csv_duplicate_flag = 1
        N_match += N
        csv_coords = csv_coords[index]
        if len(csv_coords) == 0:
            break

    return N_match, N_csv, N_templ, csv_duplicate_flag


