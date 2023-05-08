"""Parser options."""

import argparse
import numpy as np
def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    # Central:
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dtype', default='float', type=str, help='Data type used during reconstruction [Not during training!].')


    parser.add_argument('--trained_model', action='store_true', help='Use a trained model.')
    parser.add_argument('--epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')

    parser.add_argument('--accumulation', default=0, type=int, help='Accumulation 0 is rec. from gradient, accumulation > 0 is reconstruction from fed. averaging.')
    parser.add_argument('--num_images', default=1, type=int, help='How many images should be recovered from the given gradient.')
    parser.add_argument('--target_id', default=None, type=int, help='Cifar validation image used for reconstruction.')
    parser.add_argument('--label_flip', action='store_true', help='Dishonest server permuting weights in classification layer.')

    # Rec. parameters
    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false', help='Do not used signed gradients.')
    parser.add_argument('--boxed', action='store_false', help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str, help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')

    # SCDP
    # quantization arguments
    parser.add_argument('--privacy',type=bool, default=True,
                        help="whether to preserve privacy")
    parser.add_argument('--privacy_noise', type=str, default='jopeq_vector',
                        choices=['laplace', 't', 'jopeq_scalar', 'jopeq_vector'],
                        help="types of PPNs to choose from")
    parser.add_argument('--epsilon', type=float, default=4,
                        help="privacy budget (epsilon)")
    parser.add_argument('--sigma_squared', type=float, default=0.2,
                        help="scale for t-dist Sigma (identity matrix)")
    parser.add_argument('--nu', type=float, default=4,
                        help="degrees of freedom for t-dist")

    parser.add_argument('--quantization', type=bool, default=True,
                        help="whether to perform quantization")
    parser.add_argument('--binary', type=bool, default=True,
                        help="whether to perform binary convert")
    parser.add_argument('--lattice_dim', type=int, default=2,
                        choices=[1, 2], 
                        help="perform scalar (lattice_dim=1) or lattice (lattice_dim=2) quantization ")
    parser.add_argument('--R', type=int, default=16,
                        help="compression rate (number of bits per sample)")
    parser.add_argument('--gamma', type=float, default=set_gamma(parser.parse_args()),
                        help="quantizer dynamic range")
    parser.add_argument('--vec_normalization', type=bool, default=set_vec_normalization(parser.parse_args()),
                        help="whether to perform vectorized normalization, otherwise perform scalar normalization")



    # Files and folders:
    parser.add_argument('--save_image', type=bool,default=True, help='Save the output to a file.')

    parser.add_argument('--image_path', default='images_new/', type=str)
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')

    # Defense strategy:
    parser.add_argument('--defense', default='scdp', type=str, help='defense strategy.')
    parser.add_argument('--pruning_rate', default=100, type=float, help='pruning rate for defense.')
    
    return parser

def set_gamma(args):
    eta = 1.5
    gamma = 1
    if args.privacy:
        if args.lattice_dim == 1:
            gamma += 2 * ((2 / args.epsilon) ** 2)
        else:
            gamma += args.sigma_squared * (args.nu / (args.nu - 2))
    return eta * np.sqrt(gamma)


def set_vec_normalization(args):
    vec_normalization = False
    if args.quantization:
        if args.lattice_dim == 2:
            vec_normalization = True
    if args.privacy:
        if args.privacy_noise == 't' or args.privacy_noise == 'jopeq_vector':
            vec_normalization = True
    return vec_normalization
