import os
import torch
import torch.nn.functional as F



def check_mk_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)
    return paths


def from_label_to_onehot(labels,num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes, labels.size(2), labels.size(3),labels.size(4)).to(labels.device)
    target = one_hot.scatter_(1, labels.to(torch.int64), 1)
    return target

if __name__ == '__main__':

    # test for heatmap
    # mask = tio.ScalarImage(
    #     path='careIIChallenge/preprocessed/mask_private/125.nii.gz')
    # mask_to_heatmap(mask.data)
    # mask_to_polygon((mask.data[0, :, :, 172]))
    labels = torch.ones((2,1,5,5,5))
    onehot = from_label_to_onehot(labels,5)
    print(onehot.shape)