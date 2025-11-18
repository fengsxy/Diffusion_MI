import math
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import Denoiser
from utils import EMA


class MMGEstimator(pl.LightningModule):
    """
    Mutual Information Neural Diffusion Estimator using a Hierarchical Mixture of Experts approach
    """
    def __init__(self, 
                 x_shape=(1,), 
                 y_shape=(1,),
                 learning_rate=1e-4, 
                 batch_size=512,
                 logsnr_loc=2., 
                 logsnr_scale=3., 
                 max_epochs=None,
                 seed=42, 
                 task_name="default", 
                 task_gt=-1, 
                 test_num=10000, 
                 mi_estimation_interval=500,
                 use_ema=True,
                 ema_decay=0.999,
                 test_batch_size=1000,
                 snr_threshold=5,
                 strength=1.0,
                 dim=64):
        
        super().__init__()
        self.save_hyperparameters()
        self.d_x = np.prod(x_shape)
        self.d_y = np.prod(y_shape)
        self.h_g = 0.5 * self.d_x * math.log(2 * math.pi * math.e)
        self.left = (-1,) + (1,) * len(x_shape)
        self.mi_estimation_interval = mi_estimation_interval
        self.strength = strength
        self.dim = dim
        
        self.task_name = task_name
        self.logger_name = f"mmg_estimator_{task_name}_seed_{seed}_lr_{learning_rate}_strength_{strength}_dim_{dim}_MOE"
        self.task_gt = task_gt
        self.test_num = test_num
        self.logsnr_scale = logsnr_scale
        self.use_ema = use_ema
        self.snr_threshold = snr_threshold
        
        if max_epochs:
            self.logger_name += f"_max_epochs_{max_epochs}"
        
        self.logsnr_loc = t.tensor(logsnr_loc, device=self.device)

        # Initialize the low and high SNR models
        self.model_low = Denoiser(self.d_x, self.d_y, hidden_dim=self.dim)
        self.model_high = Denoiser(self.d_x, self.d_y, hidden_dim=self.dim)

        # Initialize EMA models if enabled
        self.model_low_ema = EMA(self.model_low, decay=ema_decay) if use_ema else None
        self.model_high_ema = EMA(self.model_high, decay=ema_decay) if use_ema else None

    #-------------------------- CORE METHODS (MOST IMPORTANT) --------------------------#

    def nll(self, x, y=None):
        """Calculate negative log likelihood"""
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse(x, logsnr, y)
        mmse_gap = mses - self.d_x * t.sigmoid(logsnr)
        return self.h_g + 0.5 * (weights * mmse_gap).mean()

    def mse(self, x, logsnr, y=None):
        """Calculate MSE using hierarchical mixture of experts based on SNR threshold"""
        # Get noisy input
        z, eps = self.noisy_channel(x, logsnr)
        
        # Apply SNR-based routing
        low_mask = logsnr < self.snr_threshold
        high_mask = logsnr >= self.snr_threshold
        
        # Initialize total error tensor
        total_error = t.zeros(len(x), device=x.device)
        
        # Process low SNR samples
        if low_mask.any():
            z_low = z[low_mask]
            logsnr_low = logsnr[low_mask]
            eps_low = eps[low_mask]
            y_low = y[low_mask] if y is not None else None
            
            # For low SNR, predict eps directly
            eps_hat_low = self.model_low(z_low, logsnr_low, y_low)
            error_low = (eps_low - eps_hat_low).flatten(start_dim=1)
            total_error[low_mask] = t.einsum('ij,ij->i', error_low, error_low)
        
        # Process high SNR samples
        if high_mask.any():
            z_high = z[high_mask]
            logsnr_high = logsnr[high_mask]
            eps_high = eps[high_mask]
            y_high = y[high_mask] if y is not None else None
            
            eps_hat_high = self.model_high(z_high, logsnr_high, y_high)
            error_high = (eps_high - eps_hat_high).flatten(start_dim=1)
            total_error[high_mask] = t.einsum('ij,ij->i', error_high, error_high)
        return total_error

    def mse_orthogonal(self, x, logsnr, y):
        """Calculate MSE between unconditional and conditional predictions"""
        z, _ = self.noisy_channel(x, logsnr)
        
        # Apply SNR-based routing
        low_mask = logsnr < self.snr_threshold
        high_mask = logsnr >= self.snr_threshold
        
        # Initialize error tensor
        error = t.zeros_like(x).flatten(1)
        
        # Process low SNR samples
        if low_mask.any():
            z_low = z[low_mask]
            logsnr_low = logsnr[low_mask]
            y_low = y[low_mask] if y is not None else None
            
            eps_hat_x_low = self.model_low(z_low, logsnr_low)
            eps_hat_y_low = self.model_low(z_low, logsnr_low, y_low)
            error[low_mask] = eps_hat_x_low - eps_hat_y_low
        
        # Process high SNR samples
        if high_mask.any():
            z_high = z[high_mask]
            logsnr_high = logsnr[high_mask]
            y_high = y[high_mask] if y is not None else None
            
            eps_hat_x_high = self.model_high(z_high, logsnr_high)
            eps_hat_y_high = self.model_high(z_high, logsnr_high, y_high)
            error[high_mask] = eps_hat_x_high - eps_hat_y_high
        
        return t.einsum('ij,ij->i', error, error)

    def estimate(self, X, Y, n_samples=1000) -> float:
        """Estimate mutual information on test data"""
        self.eval()
        tmp_model_low = self.model_low
        tmp_model_high = self.model_high
        
        # Use EMA models if available
        if self.use_ema:
            self.model_low = self.model_low_ema.module
            self.model_high = self.model_high_ema.module

        X = t.tensor(X, dtype=t.float32).to(self.device)
        Y = t.tensor(Y, dtype=t.float32).to(self.device)
        
        with t.no_grad():
            mean_estimate = []
            mean_orthogonal = []
            for _ in range(10):
                nll_x = self.nll(X)
                nll_xy = self.nll(X, Y)
                mi_estimate = nll_x - nll_xy
                mi_estimate_orthogonal = self.estimate_orthogonal(X, Y)
                mean_estimate.append(mi_estimate)
                mean_orthogonal.append(mi_estimate_orthogonal)
            mi_estimate = t.stack(mean_estimate).mean()
            mi_estimate_orthogonal = t.stack(mean_orthogonal).mean()
        
        # Restore original models
        self.model_low = tmp_model_low
        self.model_high = tmp_model_high
        
        return mi_estimate.item(), mi_estimate_orthogonal.item()

    def estimate_orthogonal(self, x, y):
        """Estimate mutual information using mse_orthogonal"""
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse_orthogonal(x, logsnr, y)
        return 0.5 * (weights * mses).mean()

    def noisy_channel(self, x, logsnr):
        """Add noise to input based on log signal-to-noise ratio"""
        logsnr = logsnr.view(self.left)
        eps = t.randn_like(x)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def logistic_integrate(self, npoints, clip=4.):
        """Generate integration points using logistic distribution"""
        loc, scale = self.logsnr_loc, self.logsnr_scale
        loc, scale, clip = map(lambda x: t.tensor(x, device=self.device), [loc, scale, clip])
        ps = t.rand(npoints, device=self.device)
        ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps
        logsnr = loc + scale * (t.log(ps) - t.log(1-ps))
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        return logsnr, weights

    #-------------------------- TRAINING METHODS --------------------------#

    def on_before_backward(self, loss: t.Tensor) -> None:
        """Update EMA models before backward pass if enabled"""
        if self.use_ema:
            self.model_low_ema.update(self.model_low)
            self.model_high_ema.update(self.model_high)
    
    def training_step(self, batch, batch_idx):
        """Perform a single training step"""
        x, y = batch
        # Randomly choose between conditional and unconditional training
        if t.rand(1) < 0.5:
            loss = self.nll(x)            
        else:
            loss = self.nll(x, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log periodically for MI estimation
        if self.global_step % self.mi_estimation_interval == 0:
            with t.no_grad():
                loss_xy_train = self.nll(x, y)
                loss_x_train = self.nll(x)
                self.logger.experiment.add_scalars(
                    'loss',
                    {
                        'train_loss_unconditional': loss_x_train,
                        'train_loss_conditional': loss_xy_train,
                        'train_loss_mean': (loss_x_train + loss_xy_train) / 2
                    },
                    self.global_step
                )
                
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step"""
        self.eval()
        x, y = batch
        loss_x = self.nll(x)
        loss_xy = self.nll(x, y)
        mi_estimate, mi_orthogonal = self.estimate_mi_during_training(x, y)
        
        self.logger.experiment.add_scalars(
                'loss',
                {
                    'estimate': mi_estimate,
                    'orthogonal': mi_orthogonal,
                    'ground_truth': self.task_gt,
                    'validation_loss_unconditional': loss_x,
                    'validation_loss_conditional': loss_xy,
                    'validation_loss_mean': (loss_x + loss_xy) / 2,
                },
                self.global_step)
        
        return loss_x, loss_xy

    def estimate_mi_during_training(self, x, y):
        """Estimate mutual information during training"""
        self.eval()
        with t.no_grad():
            nll_x = self.nll(x)
            nll_xy = self.nll(x, y)
            mi_estimate = nll_x - nll_xy
            mi_estimate_orthogonal = self.estimate_orthogonal(x, y)
        self.train()
        return mi_estimate.item(), mi_estimate_orthogonal.item()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
        }

    def fit(self, X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
        """Fit the model to training data"""
        train_sample_num = len(X)
        
        dataset = TensorDataset(t.tensor(X, dtype=t.float32), t.tensor(Y, dtype=t.float32))
        train_data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True) 
        validation_dataset = TensorDataset(t.tensor(X_test, dtype=t.float32), t.tensor(Y_test, dtype=t.float32))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)

        trainer = self.configure_trainer(train_sample_num)
        trainer.fit(model=self, train_dataloaders=train_data_loader,
                val_dataloaders=validation_dataloader)
        return self

    #-------------------------- UTILITY METHODS --------------------------#
    
    def configure_logger(self, train_sample_num):
        """Configure TensorBoard logger"""
        logger = TensorBoardLogger("lightning_logs", name=self.logger_name+"-"+str(train_sample_num))
        logger.log_hyperparams({
            'train_sample_num': train_sample_num,
            'test_sample_num': self.test_num,
            'strength': self.strength,
            'dim': self.dim
        })
        return logger

    def configure_trainer(self, train_sample_num, **trainer_kwargs):
        """Configure PyTorch Lightning trainer"""
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{self.task_name}',
            filename=f'mmg_estimator-{self.logger_name}-{train_sample_num}',
            save_last=False,
            save_top_k=1,
            monitor='train_loss',
            mode='min',
        )
        logger = self.configure_logger(train_sample_num)
        trainer_kwargs.update({
            'callbacks': [checkpoint_callback],
            'logger': logger,
            'log_every_n_steps': 10,
             "devices":1,
            "strategy":'auto' 

        })
       
        if self.hparams.max_epochs:
            trainer_kwargs['max_epochs'] = self.hparams.max_epochs

        trainer = pl.Trainer(**trainer_kwargs)
        return trainer
    
    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        '''
        Load model from checkpoint
        
        Usage:
        model = MMGEstimator.load_model(checkpoint_path)
        '''
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    
    def on_save_checkpoint(self, checkpoint):
        """Save additional state in checkpoint"""
        if self.use_ema:
            checkpoint['model_low_ema_state_dict'] = self.model_low_ema.state_dict()
            checkpoint['model_high_ema_state_dict'] = self.model_high_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load additional state from checkpoint"""
        if self.use_ema:
            if 'model_low_ema_state_dict' in checkpoint:
                self.model_low_ema.load_state_dict(checkpoint['model_low_ema_state_dict'])
            if 'model_high_ema_state_dict' in checkpoint:
                self.model_high_ema.load_state_dict(checkpoint['model_high_ema_state_dict'])
    
    @staticmethod
    def logsnr_to_weight(logsnr, loc, scale, clip=4.):
        """
        Convert logsnr to corresponding weight
        
        Args:
            logsnr (torch.Tensor): The input logsnr value(s)
            loc (float): Location parameter of the logistic distribution
            scale (float): Scale parameter of the logistic distribution
            clip (float): Clipping value, default is 4.0
            
        Returns:
            torch.Tensor: Corresponding weight(s) for the input logsnr
        """
        # Ensure all inputs are tensors on the same device as logsnr
        loc = t.tensor(loc, device=logsnr.device)
        scale = t.tensor(scale, device=logsnr.device)
        clip = t.tensor(clip, device=logsnr.device)
        
        # Calculate weight using the formula
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        
        return weights
