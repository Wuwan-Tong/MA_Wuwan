import argparse

VQA_NUMCLASS = 3129  # number of VQA class
def get_args():
    parser = argparse.ArgumentParser('training script', add_help=False)

    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_samples_train', default=50000, type=int,
                        help='number of data samples for training')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='epoch where start training, =1 if not load_ckpt')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='update frequency while training parameters')
    parser.add_argument('--count_param_num', default=False, type=bool,
                        help='count the number of parameters in training')

    # learning rate
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')



    # weight decay
    parser.add_argument('--weight_decay', type=float, default=0.2,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD. 
            (Set the same value with args.weight_decay to keep weight decay no change)""")


    # testing
    parser.add_argument('--eval_freq', default=None, type=int,
                        help='test per eval_freq epochs')

    # noise
    parser.add_argument('--train_snr', default=18, type=float,
                        help='snr for training')
    parser.add_argument('--test_snr', default=12, type=float,
                        help='snr for testing')
    parser.add_argument('--noise_scenario', default=1, type=int,
                        help="""scenario 1/2/3/4.
                             1: training with noise, testing without noise; 
                             2: training without noise, testing with noise; 
                             3: training with noise, testing with noise; 
                             4: training without noise, testing without noise""")

    # dataset
    parser.add_argument('--task_name', default='vqa', type=str,
                        help='dataset name or task name')
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    # print and log
    parser.add_argument('--print_freq_testing', default=500, type=int,
                        help='frequency to print testing information')
    parser.add_argument('--logfile_path', default=None, type=str,
                        help='path of txt log file')


    # model and GPU
    parser.add_argument('--model', default='ViT-B-32', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1000, type=int)


    # checkpoints
    parser.add_argument('--save_ckpt_path', default=None,
                        help='path to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=None, type=int)
    parser.add_argument('--resume', default=True, type=bool,
                        help='resume from checkpoint')
    parser.add_argument('--load_latest', default=True, type=bool,
                        help='load the latest checkpoint')
    parser.add_argument('--load_ckpt_path', default='', type=str,
                        help='path to load the previous checkpoint')











    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for data')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--update_freq', default=1, type=int)
            
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Augmentation parameters 
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')






    return parser.parse_args()