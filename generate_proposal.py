import numpy as np
import pandas as pd
import os
from random import shuffle



def generate_proposals(start_gt, end_gt, label, n_frame, alpha=5, beta=2.5, n_to_generate=100):
    duration = end_gt - start_gt
    proposals = []

    while n_to_generate:
        iou = np.random.beta(alpha, beta)
        not_success = True
        while not_success:
            is_start = np.random.randint(2)
            endpoint1 = np.random.randint(start_gt, end_gt)
            if is_start:
                start_ps = endpoint1
                intersection = end_gt - start_ps
                if intersection / duration < iou:
                    continue
                x = (intersection - duration * iou) / iou
                end_ps = round(end_gt + x)
                if end_ps > n_frame:
                    continue
            else:
                end_ps = endpoint1
                intersection = end_ps - start_gt
                x = (intersection - duration * iou) / iou
                if intersection / duration < iou:
                    continue
                start_ps = round(start_gt - x)
                if start_ps < 0:
                    continue
            not_success = False
            n_to_generate = n_to_generate - 1
            proposals.append([label, iou, intersection/(end_ps - start_ps), start_ps, end_ps])
    return proposals


def generate_proposal_file_per_video(index, video_path, gt_path, mapping, f, n_ps_per_gt):
    video = pd.read_csv(gt_path, header=None)
    video = video[video.columns[0]].values.tolist()
    n_frame = len(video)
    current_label = video[0]
    start_idx = 0
    n_gt = 0
    gt=[]
    proposals = []
    for i in range(n_frame):
        if video[i] == current_label:
            continue
        else:
            end_idx = i - 1
            label = mapping[current_label]

            if label != 0:
                n_gt = n_gt + 1
                gt.append([label, start_idx, end_idx])
            print(current_label, mapping[current_label], start_idx, end_idx)
            start_idx = i
            current_label = video[i]

    print(len(proposals))

    f.write("#%s\n" %index)
    f.write(video_path + "\n")
    f.write(str(n_frame)+"\n" + "1" + "\n")
    f.write(str(n_gt) + "\n")
    for i in range(n_gt):
        f.write(str(gt[i][0]) + " " + str(gt[i][1]) + " "+ str(gt[i][2]) + "\n")
        ps = generate_proposals(start_gt=gt[i][1], end_gt=gt[i][2], label=gt[i][0], n_frame=n_frame,
                                n_to_generate=n_ps_per_gt)
        proposals.extend(ps)
    shuffle(proposals)
    f.write(str(len(proposals)) + "\n")
    for i in range(len(proposals)):
        f.write(str(proposals[i][0]) + " " + str(proposals[i][1]) + " " + str(proposals[i][2]) + " " +
                str(proposals[i][3]) + " " + str(proposals[i][4]) + "\n")






def main():
    path = "CS6101/"
    mapping_filepath = path + "splits/mapping_bf.txt"
    mapping_df = pd.read_csv(mapping_filepath, header=None, sep=" ")

    mapping = dict(zip(mapping_df[mapping_df.columns[1]], mapping_df[mapping_df.columns[0]]))
    print(mapping)

    videos = os.listdir(path + "groundtruth")
    print()
    print(len(videos))

    output_filepath = "data/breakfast_proposal.txt"
    f = open(output_filepath, "w")
    for i in range(len(videos)):
        generate_proposal_file_per_video(i, video_path= path + "groundtruth/" + videos[i],
                                         gt_path=path + "groundtruth/" + videos[i],
                                         mapping=mapping,
                                         f=f,
                                         n_ps_per_gt=100)
    f.close()

if __name__ == '__main__':
    main()





