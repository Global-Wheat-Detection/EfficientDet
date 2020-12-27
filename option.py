import argparse


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object '
                                     'detection network - Zylo117')

    # Dataset / Dataloader configurations
    parser.add_argument('-p', '--project', type=str,
                        default='global_wheat',
                        help='project file that contains parameters')
    parser.add_argument('--train_split', type=float,
                        default=0.8,
                        help='proportion of train to all')
    parser.add_argument('-n', '--num_workers', type=int,
                        default=20,
                        help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int,
                        default=12,
                        help='The number of images per batch among all devices')
    parser.add_argument('--force_input_size', type=int,
                        default=None)
    parser.add_argument('--aug_prob', type=float,
                        default=0.5)

    # Model configurations
    parser.add_argument('-c', '--compound_coef', type=int,
                        default=0,
                        help='coefficients of efficientdet')
    parser.add_argument('--head_only', action='store_true',
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    
    # Training configurations
    parser.add_argument('--lr', type=float,
                        default=1e-4)
    parser.add_argument('--optim', type=str,
                        default='adamw',
                        help='select optimizer for training, '
                             'suggest using \'admaw\' until the'
                             ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int,
                        default=500)
    parser.add_argument('--val_interval', type=int,
                        default=1,
                        help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int,
                        default=500,
                        help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float,
                        default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to '
                             'qualify as an improvement')
    parser.add_argument('--es_patience', type=int,
                        default=0,
                        help='Early stopping\'s parameter: number of epochs with no '
                             'improvement after which training will be stopped. Set to '
                             '0 to disable this technique.')
    parser.add_argument('--debug', action='store_true',
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    # Path configurations
    parser.add_argument('--data_path', type=str,
                        default='datasets/',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str,
                        default='logs/')
    parser.add_argument('-w', '--load_weights', type=str,
                        default=None,
                        help='whether to load weights from a checkpoint, set None '
                             'to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str,
                        default='logs/')

    args = parser.parse_args()
    return args
