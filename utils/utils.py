import torch
def generate_mean_std(args):

    mean = torch.tensor(args.data.MEAN).cuda()
    std = torch.tensor(args.data.STD).cuda()

    view = [1, len(args.data.MEAN), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std