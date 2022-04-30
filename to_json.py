import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
parser.add_argument('--dataset', type=str, nargs='?', default='Epinion',
                    help='dataset name')

# Experimental settings.
parser.add_argument('--world_size', type=int, nargs='?', default=2,
                    help='distributed scale')
parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help='# epochs')
parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                    help='Validation frequency (in epochs)')
parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                    help='Testing frequency (in epochs)')
parser.add_argument('--batch_size', type=int, nargs='?', default=128,
                    help='Batch size (# nodes)')
parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                help='True if one-hot encoding.')
parser.add_argument("--data_str", type=str, default='pyg',
                    help="framework to construct the graph")
parser.add_argument("--model", type=str, default='DySAT',
                    help="Baseline models")
# 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.

# Tunable hyper-params
# TODO: Implementation has not been verified, performance may not be good.
parser.add_argument('--residual', type=bool, nargs='?', default=True,
                    help='Use residual')
parser.add_argument("--interval_ratio", type=int, default=0,
                        help="num of interval to keep")               

# model hyperparameters
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.001,
                    help='Initial learning rate for self-attention model.')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                    help='Initial learning rate for self-attention model.')

args = vars(parser.parse_args())

b = json.dumps(args)
f1 = open('parameters.json', 'w')
f1.write(b)
f1.close()