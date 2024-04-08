import argparse

from generate_data_for_training import generate_data

parser = argparse.ArgumentParser(description='')

parser.add_argument('--saving_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN",
                    help='Directory path for saving output files.')
parser.add_argument('--image_path', type=str, default=None,
                    help='Path to the input image file. Default is <saving_path>/exp_data/input_beam.tiff')

parser.add_argument('--resolution_in', type=int, default=512,
                    help='Input resolution.')
parser.add_argument('--resolution_out', type=int, default=256,
                    help='Output resolution.')

parser.add_argument('--number_of_n2', type=int, default=10,
                    help='Number of N2 instances.')
parser.add_argument('--number_of_power', type=int, default=10,
                    help='Number of power instances.')
parser.add_argument('--number_of_isat', type=int, default=10,
                    help='Number of ISAT instances.')

parser.add_argument('--is_from_image', action='store_true',
                    help='Whether the input is from an image.')
parser.add_argument('--visualize', action='store_true',
                    help='Enable visualization.')
parser.add_argument('--expension', action='store_true',
                    help='Enable expension.')
parser.add_argument('--generate', action='store_true',
                    help='Enable generation.')
parser.add_argument('--single_power', action='store_true',
                    help='Enable generation.')
parser.add_argument('--multiple_power', action='store_true',
                    help='Enable generation.')

parser.add_argument('--delta_z', type=float, default=1e-3,
                    help='Delta Z value.')
parser.add_argument('--trans', type=float, default=0.01,
                    help='Trans value.')
parser.add_argument('--length', type=float, default=20e-2,
                    help='Length value.')
parser.add_argument('--factor_window', type=int, default=55,
                    help='Factor window value.')

parser.add_argument('--training', action='store_true',
                    help='Enable training.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--accumulator', type=int, default=10,
                    help='Number of accumulation steps')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='Number of epochs')

# Parse the arguments
args = parser.parse_args()

# Set the default for image_path if not specified
if args.image_path is None:
    args.image_path = f'{args.saving_path}/exp_data/input_beam.tiff'

# You can now use args to access the values of the arguments
resolutions = args.resolution_in, args.resolution_out
numbers = args.number_of_n2, args.number_of_power, args.number_of_isat

labels, values = generate_data(args.saving_path, args.image_path, resolutions, numbers, 
                                args.is_from_image, args.generate, args.visualize, args.expension, args.single_power,
                                    args.multiple_power, args.factor_window, args.delta_z, args.length, 
                                        args.trans)

if args.training:
    print("-- TRAINING --")
    if args.single_power and args.multiple_power:
        print("-- SINGLE - MULTIPLE --")

        labels_all_single, labels_all_multiple = labels
        values_all_single, values_all_multiple = values
        
        print("-- SINGLE --")
        from single_power.n2_finder_resnet_single_power import lauch_training
        lauch_training(numbers, labels_all_single, values_all_single, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator)
        
        print("-- MULTIPLE --")
        from multiple_power.n2_finder_resnet_multiple_power import lauch_training
        lauch_training(numbers, labels_all_multiple, values_all_multiple, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator)
    
    elif args.single_power:
        print("-- SINGLE --")

        from single_power.n2_finder_resnet_single_power import lauch_training
        lauch_training(numbers, labels, values, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator)
    
    elif args.multiple_power:

        print("-- MULTIPLE --")
        from multiple_power.n2_finder_resnet_multiple_power import lauch_training
        lauch_training(numbers, labels, values, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator)
