import numpy as np 
import os
import cv2
import pickle as pkl
from glob import glob
from time import time
import re
import math
import pandas as pd

from multiprocessing import Pool

"""
Sorts based on natural ordering of numbers, ie. "12" > "2" 
"""
def naturalSort(String_): 
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]


"""
Writes a string to a file
"""
def write_file(path, text):
    with open(path, 'w') as path_stream:
        path_stream.write(text)
        path_stream.close()


def calculate_f1(prediction_files, cutoff_brightness=200, cutoff_area=100):

    t1 = time()

    false_positive_dict = {}
    false_negative_dict = {}
    match_dict = {}
    fulloutfile = ''
    summaryoutfile = ''
    epoch_set = set()

    for prediction_file in prediction_files:

        #print('\n\n')

        ta = time()

        prediction = cv2.imread(prediction_file)

        base = prediction_file.rpartition('/')[2].rpartition('.png')[0]

        #print(base)

        epoch = int(base.rpartition('_epoch')[2].partition('.')[0])
        epoch_set.add(epoch)
        try:
            false_positive_dict[epoch]
        except:
            false_positive_dict[epoch] = []
            false_negative_dict[epoch] = []
            match_dict[epoch] = []

        pthresh = cv2.threshold(prediction, cutoff_brightness, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('data/membrane/test_stitched/' + base + '_binary.png', pthresh)


        pthresh = cv2.imread('data/membrane/test_stitched/' + base + '_binary.png',0)

        h,w = pthresh.shape[:2]

        tb = time()
        #print(tb-ta)

        # white_image = 255.0 * np.ones((h,w), np.uint8)

        # contours, im2  = cv2.findContours(cv2.bitwise_not(pthresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < h*w/5.0:
        #         cv2.drawContours(white_image,cnt,0,0,-1)
            
        # cv2.imwrite('data/membrane/test_stitched/' + base + '_contours.png', white_image)

        basebase = base.rpartition('.')[0].rpartition('.')[0]

        filtered = cv2.imread('data/membrane/test_stitched/' + base + '_binary.png')
        cv2.rectangle(filtered,(0,0),(w,h), (255,255,255),-1)

        tb1 = time()
        #print('getting filtered')
        #print(tb1-tb)

        filtered_bw = pthresh.copy()

        tb2 = time()
        #print(tb2-tb)

        testis_locs = pd.read_csv('testislocs/' + basebase + '.locs.csv')

        tc = time()
        #print(tc-tb2)

        def matchQ(cx,cy):
            for row in testis_locs.itertuples():
                dist = math.sqrt((row.x - cx)**2 + (row.y - cy)**2)
                if dist <= 35:
                    return True
            return False

        false_positives = 0
        cxs = []
        cys = []
        contours, im2  = cv2.findContours(filtered_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.9*h*w and area > 10:
                M = cv2.moments(cnt)
                #cv2.drawContours(filtered,[cnt],0,(0,0,0),2)
                if M['m00'] != 0 and area > cutoff_area:
                    cx = int(M['m10']/M['m00']*2.0)
                    cy = int(M['m01']/M['m00']*2.0)
                    cxs.append(cx)
                    cys.append(cy)
                    if matchQ(cx,cy):
                        cv2.drawContours(filtered,[cnt],0,(0,0,0),-1)
                    else:
                        cv2.drawContours(filtered,[cnt],0,(255,0,0),-1)
                        #print(cx,cy)
                        false_positives += 1
                        #print(false_positives)
                    cv2.circle(filtered, (int(cx/2),int(cy/2)), 10, (255,0,0), 2)

                    #cv2.circle(filtered, (cx,cy), 5, (0,255,255), -1)

        def loc_matchQ(x,y):
            for i in list(range(len(cxs))):
                dist = math.sqrt((x - cxs[i])**2 + (y - cys[i])**2)
                if dist <= 35:
                    return True
            return False


        td = time()
        #print(td-tc)

        td1 = time()
        #print(td1-td)

        matches = 0
        false_negatives = 0 
        for row in testis_locs.itertuples():

            #print(row.x,row.y)
            if loc_matchQ(row.x,row.y):
                cv2.circle(filtered, (int(row.x/2),int(row.y/2)), 30, (0,255,0), 5)
                matches += 1
            else:
                cv2.circle(filtered, (int(row.x/2),int(row.y/2)), 30, (0,0,255), 5)
                false_negatives += 1

            #cv2.circle(fimage, (x,y), 50, (0,255,255), 5)

        te = time()
        #print(te-td)

        #print(matches,false_positives,false_negatives)
        if matches == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = float(matches)/float(matches + false_positives)
            recall = float(matches)/float(matches + false_negatives)
            f1 = 2*precision*recall/float(precision + recall)

        match_dict[epoch].append(matches)
        false_negative_dict[epoch].append(false_negatives)
        false_positive_dict[epoch].append(false_positives)

        tf = time()
        #print(tf-te)

        outfile = 'Image ' + base + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
        fulloutfile += outfile
        #write_file('results/' + base + '.results.txt', outfile)
        #print(outfile)

        cv2.imwrite('data/membrane/test_stitched/' + base + '.labeled.png', filtered)

    #print(match_dict)

    master_match_dict = {}
    master_false_negative_dict = {}
    master_false_positive_dict = {}
    master_f1_dict = {}
    for epoch in epoch_set:
        matches = sum(match_dict[epoch])
        false_negatives = sum(false_negative_dict[epoch])
        false_positives = sum(false_positive_dict[epoch])

        if matches == 0:
                precision = 0
                recall = 0
                f1 = 0
        else:
            precision = float(matches)/float(matches + false_positives)
            recall = float(matches)/float(matches + false_negatives)
            f1 = 2*precision*recall/float(precision + recall)

        master_match_dict[epoch] = matches
        master_false_positive_dict[epoch] = false_negatives
        master_false_negative_dict[epoch] = false_positives
        master_f1_dict[epoch] = f1


    for epoch in sorted(list(master_match_dict.keys())):
        matches = master_match_dict[epoch]
        false_negatives = master_false_positive_dict[epoch]
        false_positives = master_false_negative_dict[epoch]
        f1 = master_f1_dict[epoch]

        summaryoutfile += 'Total for epoch ' + str(epoch) +  ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                    '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
    
    best_epoch = max(master_f1_dict, key=lambda key: master_f1_dict[key])

    matches = master_match_dict[best_epoch]
    false_negatives = master_false_positive_dict[best_epoch]
    false_positives = master_false_negative_dict[best_epoch]
    f1 = master_f1_dict[best_epoch]

    summaryoutfile += 'Best epoch: ' + str(best_epoch) + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                    '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'

    fulloutfile += summaryoutfile
    run_str = str(cutoff_brightness) + '.' + str(cutoff_area)
    write_file('results/' + basebase + '.' + run_str + '.full_results.txt', fulloutfile)
    write_file('results/' + basebase + '.' + run_str + '.results.txt', summaryoutfile)
    print(summaryoutfile)


    t2 = time()

    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    return best_epoch,f1,matches,false_negatives,false_positives,cutoff_brightness,cutoff_area

def tuple_wrapped_calculate_f1(input_tuple):
    prediction_files,cutoff_brightness,cutoff_area = input_tuple
    return calculate_f1(prediction_files, cutoff_brightness=cutoff_brightness, cutoff_area=cutoff_area)

run_name = 'augmented'
base_path = 'data/membrane/test_stitched/'
prediction_files = sorted(glob(base_path + '?????.?.' + run_name + '_epoch*.stitched.png'), key=naturalSort)

#[150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
#[10, 20, 40, 60, 80, 100, 120, 140, 180, 220, 300, 400][11:13]
#[180, 190, 200, 210, 220]

input_list = [(prediction_files,cutoff_brightness,cutoff_area) for \
        cutoff_brightness in [225] for \
        cutoff_area in [500]]


if False:
    for file in glob('testislocs/*.pkl'):
        base = file.rpartition('.pkl')[0]
        testis_locs = pkl.load(open(base + '.pkl', 'rb'))
        testis_locs.to_csv(base + '.csv')

if False:
    pool = Pool(4)
    output_list = pool.map(tuple_wrapped_calculate_f1, input_list) 
else:
    output_list = []
    for input_tuple in input_list:
        output_list.append(tuple_wrapped_calculate_f1(input_tuple))

records_list = []
for record_tuple in output_list:
    epoch,f1,matches,false_negatives,false_positives,cutoff_brightness,cutoff_area = record_tuple
    records_list.append({
        'epoch':epoch,
        'cutoff_brightness':cutoff_brightness,
        'cutoff_area':cutoff_area,
        'f1':f1,
        'matches':matches,
        'false_negatives':false_negatives,
        'false_positives':false_positives
        })

records_df = pd.DataFrame(records_list)

max_index = records_df['f1'].idxmax()

records_df.to_csv('results/' + run_name + '.results.csv')
records_df.to_pickle('results/' + run_name + '.results.pkl')

cutoff_brightness=records_df.iloc[max_index]['cutoff_brightness']
cutoff_area=records_df.iloc[max_index]['cutoff_area']
calculate_f1(prediction_files, cutoff_brightness=cutoff_brightness, cutoff_area=cutoff_area)

print(records_df.iloc[max_index])




