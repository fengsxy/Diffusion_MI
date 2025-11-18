import os
import json
import argparse
import numpy as np
import pytorch_lightning as pl
from sklearn import preprocessing


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate MMG estimator with optional MOE")
    
    # Model architecture parameters
    parser.add_argument('--use_moe', action='store_true',
                        help='Use Mixture of Experts (MOE) architecture with separate low/high SNR models')
    parser.add_argument('--snr_threshold', type=float, default=5.0,
                        help='Threshold for low/high SNR routing (only used with MOE)')
    
    # Task parameters
    parser.add_argument('--task_type', type=str, default='multinormal',
                        choices=['multinormal', 'half_cube', 'spiral'],
                        help='Type of task to evaluate')
    parser.add_argument('--dim', type=int, default=3,
                        choices=[3, 5, 10, 20, 50],
                        help='Dimension for the task')
    parser.add_argument('--strength', type=int, default=2000,
                        choices=[200, 650, 1100, 1550, 2000],
                        help='Strength parameter for the task')
                        
    # Dataset parameters
    parser.add_argument('--train_sample_num', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--test_num', type=int, default=10000,
                        help='Number of test samples')
                        
    # Model parameters                    
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Hidden dimension for the model')
    parser.add_argument('--logsnr_loc', type=float, default=2.0,
                        help='Location parameter for logSNR distribution')
    parser.add_argument('--logsnr_scale', type=float, default=3.0,
                        help='Scale parameter for logSNR distribution')
                        
    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Batch size for testing')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (if None, will be set based on dimensionality)')
    parser.add_argument('--mi_estimation_interval', type=int, default=500,
                        help='Interval for MI estimation during training')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use exponential moving average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Decay rate for EMA')
    parser.add_argument('--preprocess', type=str, default='rescale',
                        choices=['rescale', 'none'],
                        help='Data preprocessing method')
                        
    # Utility parameters
    parser.add_argument('--load_ckpt', action='store_true',
                        help='Load from checkpoint instead of training')
    parser.add_argument('--results_file', type=str, default=None,
                        help='File to save results (auto-generated if None)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all combinations of dim and strength values')
    
    return parser.parse_args()


def get_task(task_type, dim, strength, bmi):
    """Create and return a task based on the specified parameters"""
    # Create base task
    base_task = bmi.benchmark.tasks.task_multinormal_sparse(dim_x=dim, dim_y=dim, strength=strength)
    
    # Apply transformations based on task type
    if task_type == 'multinormal':
        task = base_task
        task_name = f"multinormal_sparse_{strength}_dim_{dim}"
    elif task_type == 'half_cube':
        task = bmi.benchmark.tasks.transform_half_cube_task(base_task)
        task_name = f"half_cube_multinormal_sparse_{strength}_dim_{dim}"
    elif task_type == 'spiral':
        task = bmi.benchmark.tasks.transform_spiral_task(base_task)
        task_name = f"spiral_multinormal_sparse_{strength}_dim_{dim}"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Set task name
    task.task_name = task_name
    
    return task


def get_estimator_class(use_moe):
    """Import and return the appropriate estimator class"""
    if use_moe:
        from MMG_Unet_MOE import MMGEstimator
        print("Using Mixture of Experts (MOE) architecture")
    else:
        from MMG_Unet import MMGEstimator
        print("Using unified single model architecture")
    
    return MMGEstimator


def run_experiment(args, task, bmi):
    """Run a single experiment with the specified task"""
    # Set random seed for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    
    # Set learning rate based on dimensionality if not specified
    if args.learning_rate is None:
        if task.dim_x <= 5:
            args.learning_rate = 1e-3
            args.batch_size = 128
        else:
            args.learning_rate = 2e-3
            args.batch_size = 256
            
    # Set max epochs based on dimensionality if not specified
    if args.max_epochs is None:
        if task.dim_x <= 5:
            args.max_epochs = 500
        else:
            args.max_epochs = 750
    
    # Set results filename if not specified
    if args.results_file is None:
        model_type = "moe" if args.use_moe else "unified"
        args.results_file = f'results_{model_type}_{args.train_sample_num}.json'
    
    # Sample data
    print(f"Sampling {args.train_sample_num + args.test_num} data points from {task.task_name} task...")
    X, Y = task.sample(args.train_sample_num + args.test_num, seed=args.seed)
    X, Y = X.__array__(), Y.__array__()
    
    # Split into train and test
    X_test = X[args.train_sample_num:args.train_sample_num+args.test_num]
    Y_test = Y[args.train_sample_num:args.train_sample_num+args.test_num]
    X_train = X[:args.train_sample_num]
    Y_train = Y[:args.train_sample_num]

    # Preprocess data if required
    if args.preprocess == 'rescale':
        print("Preprocessing data with standardization...")
        scaler_X = preprocessing.StandardScaler(copy=True)
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        scaler_Y = preprocessing.StandardScaler(copy=True)
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)
    
    # Get the appropriate estimator class
    MMGEstimator = get_estimator_class(args.use_moe)
    
    # Initialize model with conditional parameters
    model_kwargs = {
        'x_shape': (task.dim_x,),
        'y_shape': (task.dim_y,),
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'logsnr_loc': args.logsnr_loc,
        'logsnr_scale': args.logsnr_scale,
        'max_epochs': args.max_epochs,
        'seed': args.seed,
        'task_name': task.task_name,
        'task_gt': task.mutual_information,
        'test_num': args.test_num,
        'mi_estimation_interval': args.mi_estimation_interval,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'test_batch_size': args.test_batch_size,
        'strength': args.strength,
        'dim': args.model_dim
    }
    
    # Add MOE-specific parameters
    if args.use_moe:
        model_kwargs['snr_threshold'] = args.snr_threshold
    
    print(f"Initializing MMG estimator ({'MOE' if args.use_moe else 'Unified'}) with strength={args.strength}, dim={args.model_dim}...")
    estimator = MMGEstimator(**model_kwargs)
    
    # Define checkpoint path
    model_type = "moe" if args.use_moe else "unified"
    ckpt_dir = f'checkpoints/{task.task_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f'{ckpt_dir}/mmg_estimator_{model_type}-{estimator.logger_name}-{args.train_sample_num}.ckpt'
    
    # Either load from checkpoint or train the model
    if args.load_ckpt and os.path.exists(ckpt_path):
        print(f"Loading model from checkpoint: {ckpt_path}")
        estimator = MMGEstimator.load_model(checkpoint_path=ckpt_path)
        print(f"LogSNR loc: {estimator.logsnr_loc}, scale: {estimator.logsnr_scale}")
    else:
        print("Training model...")
        estimator.fit(X_train, Y_train, X_test, Y_test)
        estimator.trainer.save_checkpoint(ckpt_path)
        print(f"Model saved to {ckpt_path}")

    # Estimate mutual information
    print("Estimating mutual information on test set...")
    mi_estimate, mi_orthogonal = estimator.estimate(X_test, Y_test)
    print(f"Ground truth MI: {task.mutual_information}")
    print(f"Estimated MI: {mi_estimate}")
    print(f"Orthogonal MI: {mi_orthogonal}")
    
    # Save results
    result_dict = {
        "task": task.task_name,
        "gt_mi": task.mutual_information,
        "mi_estimate": mi_estimate,
        "mi_estimate_orthogonal": mi_orthogonal,
        "estimator": f"MMGEstimator_{'MOE' if args.use_moe else 'Unified'}",
        "use_moe": args.use_moe,
        "snr_threshold": args.snr_threshold if args.use_moe else None,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_sample_num": args.train_sample_num,
        "test_sample_num": args.test_num,
        "max_epochs": args.max_epochs,
        "model_dim": args.model_dim,
        "task_dim": task.dim_x,
        "task_strength": task.strength if hasattr(task, 'strength') else None,
        "model_strength": args.strength,
        "preprocessing": args.preprocess,
        "use_ema": args.use_ema,
    }
    
    # Append to results file
    append_results(result_dict, args.results_file)
    print(f"Results saved to {args.results_file}")


def append_results(result_dict, filename):
    """Append results to a JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
    else:
        results = []
        
    results.append(result_dict)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def check_existing_results(args, task):
    """Check if results already exist for this configuration"""
    if not args.results_file or not os.path.exists(args.results_file):
        return False
    
    try:
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        for result in results:
            if (result.get("task") == task.task_name and 
                result.get("seed") == args.seed and 
                result.get("max_epochs") == args.max_epochs and 
                result.get("preprocessing") == args.preprocess and
                result.get("use_moe") == args.use_moe and
                result.get("model_strength") == args.strength):
                return True
        
        return False
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    """Main function to run the MI estimation"""
    args = parse_arguments()
    
    try:
        import bmi
    except ImportError:
        print("BMI package not available. Please ensure it is installed.")
        print("Install with: pip install git+https://github.com/cbg-ethz/bmi.git")
        return
    
    if args.run_all:
        # Run all combinations of dimensions and strengths
        dim_list = [3, 5, 10, 20, 50]
        strength_list = [200, 650, 1100, 1550, 2000]
        task_types = ['multinormal', 'half_cube', 'spiral']
        
        total_experiments = len(dim_list) * len(strength_list) * len(task_types)
        current_experiment = 0
        
        for dim in dim_list:
            for strength in strength_list:
                for task_type in task_types:
                    current_experiment += 1
                    print(f"\n{'='*80}")
                    print(f"Experiment {current_experiment}/{total_experiments}")
                    print(f"Running experiment with task_type={task_type}, dim={dim}, strength={strength}")
                    print(f"Architecture: {'MOE' if args.use_moe else 'Unified'}")
                    print(f"{'='*80}\n")
                    
                    # Update args for this experiment
                    args.dim = dim
                    args.strength = strength
                    args.task_type = task_type
                    
                    task = get_task(task_type, dim, strength, bmi)
                    
                    # Check if results already exist
                    if check_existing_results(args, task):
                        print(f"Results already exist for this configuration, skipping...")
                        continue
                    
                    run_experiment(args, task, bmi)
    else:
        # Run a single experiment with the specified parameters
        task = get_task(args.task_type, args.dim, args.strength, bmi)
        
        # Check if results already exist
        if check_existing_results(args, task):
            print(f"Results already exist for this configuration.")
            print(f"Use different parameters or delete the results file to rerun.")
            return
        
        run_experiment(args, task, bmi)


if __name__ == "__main__":
    main()
