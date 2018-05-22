from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from net import sphere20


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n / n_folds:(i + 1) * n / n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model_path=None):
    predicts = []
    model = sphere20().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '~/dataset/lfw/lfw-112X96/'
    with open('~/Project/sphereface/test/data/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    for i in range(6000):
        p = pairs_lines[i].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

        img1 = Image.open(root + name1).convert('RGB')
        img2 = Image.open(root + name2).convert('RGB')
        img1, img1_, img2, img2_ = transform(img1), transform(F.hflip(img1)), transform(img2), transform(F.hflip(img2))
        img1, img1_ = Variable(img1.unsqueeze(0).cuda(), volatile=True), Variable(img1_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        img2, img2_ = Variable(img2.unsqueeze(0).cuda(), volatile=True), Variable(img2_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        f1 = torch.cat((model(img1), model(img1_)), 1).data.cpu()[0]
        f2 = torch.cat((model(img2), model(img2_)), 1).data.cpu()[0]

        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    '''
    _, result = eval(model_path='checkpoint/SphereFace_24_checkpoint.pth')
    np.savetxt("result.txt", result, '%s')
    '''
    eval(model_path='checkpoint/CosFace_30_checkpoint.pth')
    '''
    for epoch in range(1, 31):
        eval('checkpoint/CosFace_' + str(epoch) + '_checkpoint.pth')
    '''
