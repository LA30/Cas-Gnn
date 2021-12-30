import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import Data
import argparse

from model import CasGnn


parser=argparse.ArgumentParser()
parser.add_argument('--checkpoints', type=str, default='./checkpoints/CasGnn.pth')
parser.add_argument('--data_root', type=str, default='/data/Datasets/SOD/NJUD/test_data')


def main(args):
    # data
    data_root = args.data_root
    test_loader = torch.utils.data.DataLoader(Data(data_root, transform=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = CasGnn().cuda()
    model.load_state_dict(torch.load(args.checkpoints))
    mae = func_eval(test_loader, model)


def func_eval(test_loader, model):
    mae = 0
    model.eval()
    for id, (data, depth, mask, img_name, img_size) in enumerate(test_loader):
        datas = [data, depth, mask]

        with torch.no_grad():
            inputs = data.cuda()
            depth = depth.cuda()
            n, c, h, w = inputs.size()
            depth = depth.unsqueeze(1).repeat(1, c, 1, 1)

        pred = model([inputs, depth])
        out = F.softmax(pred, dim=1)
        out = out.max(1)[1].squeeze_(1).float() * out[:, 1]

        mae += abs(out.detach().cpu() - mask.float()).mean()

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    return mae


if __name__ == '__main__':
    main(parser.parse_args())
