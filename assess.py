import numpy as np 
import os
import cv2
import pickle as pkl
from glob import glob
from time import time
import re
import math
import pandas as pd
import sys
from random import randint
import matplotlib.pyplot as plt

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
    print(cutoff_brightness,cutoff_area)

    t1 = time()

    false_positive_dict = {}
    false_negative_dict = {}
    match_dict = {}
    fulloutfile = ''
    summaryoutfile = ''
    epoch_set = set()

    for prediction_file in prediction_files:

        prediction = cv2.imread(prediction_file)
        h,w = prediction.shape[:2]

        base = prediction_file.rpartition('/')[2].rpartition('.png')[0]
        basebase = base.rpartition('.')[0].rpartition('.')[0]

        print(base)

        epoch = int(base.rpartition('_epoch')[2].partition('.')[0])
        epoch_set.add(epoch)
        try:
            false_positive_dict[epoch]
        except:
            false_positive_dict[epoch] = []
            false_negative_dict[epoch] = []
            match_dict[epoch] = []

        pthresh = cv2.threshold(prediction, cutoff_brightness, 255, cv2.THRESH_BINARY)[1]
        filtered = pthresh.copy()
        pthresh = cv2.cvtColor(pthresh, cv2.COLOR_BGR2GRAY)

        # try:
        #     cv2.imwrite('data/membrane/test_stitched/' + base + '_binary.png', pthresh)
        #     pthresh = plt.imread('data/membrane/test_stitched/' + base + '_binary.png',0)
        #     filtered = cv2.imread('data/membrane/test_stitched/' + base + '_binary.png')
        # except:
        #     pthresh = np.zeros((h,w),np.uint8)
        #     filtered = np.zeros((h,w,3),np.uint8)

        cv2.rectangle(filtered,(0,0),(w,h), (255,255,255),-1)

        filtered_bw = pthresh.copy()

        testis_locs = pd.read_csv('testislocs/' + basebase + '.locs.csv')

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

        outfile = 'Image ' + base + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
        fulloutfile += outfile

        #cv2.imwrite('data/membrane/test_stitched/' + base + '.labeled.png', filtered)

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

    bestoutfile = 'Best epoch: ' + str(best_epoch) + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                    '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
    summaryoutfile += bestoutfile
    fulloutfile += summaryoutfile
    run_str = str(cutoff_brightness) + '.' + str(cutoff_area)
    write_file('results/' + basebase + '.' + run_str + '.full_results.txt', fulloutfile)
    write_file('results/' + basebase + '.' + run_str + '.results.txt', summaryoutfile)
    print(summaryoutfile)


    t2 = time()

    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    return best_epoch,f1,matches,false_negatives,false_positives,cutoff_brightness,cutoff_area

def calculate_holdout_f1(run_name, cutoff_brightness, cutoff_area, epoch):
    print(cutoff_brightness,cutoff_area,epoch)

    t1 = time()

    prediction_files = sorted(glob('data/membrane/holdout_stitched/' + '?????.?.' + run_name + '.stitched.png'), key=naturalSort)

    false_positive_list = []
    false_negative_list = []
    match_list = []
    fulloutfile = ''
    summaryoutfile = ''

    for prediction_file in prediction_files:

        prediction = cv2.imread(prediction_file)

        base = prediction_file.rpartition('/')[2].rpartition('.png')[0]

        print(base)

        pthresh = cv2.threshold(prediction, cutoff_brightness, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('data/membrane/holdout_stitched/' + base + '_binary.png', pthresh)
        pthresh = cv2.imread('data/membrane/holdout_stitched/' + base + '_binary.png',0)
        h,w = pthresh.shape[:2]

        basebase = base.rpartition('.')[0].rpartition('.')[0]

        filtered = cv2.imread('data/membrane/holdout_stitched/' + base + '_binary.png')
        cv2.rectangle(filtered,(0,0),(w,h), (255,255,255),-1)


        filtered_bw = pthresh.copy()

        testis_locs = pd.read_csv('testislocs/' + basebase + '.locs.csv')

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

        #print(matches,false_positives,false_negatives)
        if matches == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = float(matches)/float(matches + false_positives)
            recall = float(matches)/float(matches + false_negatives)
            f1 = 2*precision*recall/float(precision + recall)

        match_list.append(matches)
        false_negative_list.append(false_negatives)
        false_positive_list.append(false_positives)

        outfile = 'Image ' + base + ':\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
        fulloutfile += outfile
        #write_file('results/' + base + '.results.txt', outfile)
        #print(outfile)

        cv2.imwrite('data/membrane/holdout_stitched/' + base + '.labeled.png', filtered)

    matches = sum(match_list)
    false_negatives = sum(false_negative_list)
    false_positives = sum(false_positive_list)

    if matches == 0:
            precision = 0
            recall = 0
            f1 = 0
    else:
        precision = float(matches)/float(matches + false_positives)
        recall = float(matches)/float(matches + false_negatives)
        f1 = 2*precision*recall/float(precision + recall)

    summaryoutfile += 'Total for holdout set:\nMatches = ' + str(matches) + '\nFalse positives = ' + str(false_positives) + \
                    '\nFalse negatives = ' + str(false_negatives) + '\nF1 score = ' + str(round(f1,3)) + '\n\n'
    
    write_file('results/' + run_name + '.holdout.full_results.txt', fulloutfile)
    write_file('results/' + run_name + '.holdout.results.txt', summaryoutfile)
    print(summaryoutfile)


    t2 = time()

    print('Run time: ' + str(round(t2-t1, 2)) + ' s')

    return f1,matches,false_negatives,false_positives


def tuple_wrapped_calculate_f1(input_tuple):
    prediction_files,cutoff_brightness,cutoff_area = input_tuple
    return calculate_f1(prediction_files, cutoff_brightness=cutoff_brightness, cutoff_area=cutoff_area)


def grid_search(run_name, base_path = 'data/membrane/test_stitched/'):
    print('Running grid search...')
    mt1 = time()

    prediction_files = sorted(glob(base_path + '?????.?.' + run_name + '_epoch*.stitched.png'), key=naturalSort)

    cutoff_brightness_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,120,140,160,180,200,210,220,225,230,235,240,242,245,246,247,248,249,250,251,252,253,254]
    cutoff_area_list = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1500]

    input_list = [(prediction_files,cutoff_brightness,cutoff_area) for \
            cutoff_brightness in cutoff_brightness_list for \
            cutoff_area in cutoff_area_list]

    if False:
        print('Using multiprocessing')
        pool = Pool(12)
        output_list = pool.map(tuple_wrapped_calculate_f1, input_list) 
    else:
        print('Not using multiprocessing')
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

    records_df.to_csv('results/' + run_name + '.refine.results.csv')
    records_df.to_pickle('results/' + run_name + '.refine.results.pkl')

    cutoff_brightness=records_df.iloc[max_index]['cutoff_brightness']
    cutoff_area=records_df.iloc[max_index]['cutoff_area']
    calculate_f1(prediction_files, cutoff_brightness=cutoff_brightness, cutoff_area=cutoff_area)

    print(records_df.iloc[max_index])

    mt2 = time()

    print('Full runtime: ' + str(round(mt2-mt1,2)))

    # raise Exception('for triggering printing\nFull runtime: ' + str(round(mt2-mt1,2)))

    return dict(records_df.iloc[max_index])

def main(run_name):
    grid_search(run_name)

"""
for terminal input
"""
if __name__ == '__main__':
    try:
        sys.argv[1]
    except:
        raise Exception('You need to pass in the run name.')
    run_name = str(sys.argv[1])
    print('Running')
    main(run_name)
    print('Ran')





