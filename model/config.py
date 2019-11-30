# checkpoints
logs_hyperparams = {'chkp_dir': 'chkp/',
                    'log_dir': 'logs/train_logs/',
                    # None if training from the scratch
                    'chkp_name_gen': 'gen-epoch-19.chkp',
                    'chkp_name_dis': 'dis-epoch-19.chkp'
}

# check gpu devices
train_mode = {'gpu': True,
              'gpu_devices': '2,3',
              'multi_gpu': True
}

print_summary = True
model_hyperparams = {'clip_norm': 1e-2,
                     'function': 'gaussian_roi', # roi generation function
                     'spectral_norm': True,
                     'dis_n_features': 40 # n_features in discriminator
}

gen_hyperparams = {'init_size': (8, 8),
                    'dest_size': (64, 64),
                    'scale': 2,
                    'input_channels': 128,
                    'kernel_size': 3
}

stabilizing_hyperparams = {'adding_noise': True
}

discriminator_stabilizing_hyperparams = {'fe_matching': False,
                                         'n_layers_fe_matching': list(range(12)),
                                         #'n_layers_fe_matching': [2, 5, 8, 11, 14],
                                         #'wgan_clip_size': 1e-2,
                                         'wgan_clip_size': None,
                                         'loss': 'softplus' # 'ls', 'wgan', 'softplus'
}

generator_stabilizing_hyperparams = {'roi_loss': True,
                                     'vae_loss': True,
                                     'loss': 'softplus' # 'ls', 'wgan', 'softplus' 
}

# train/test hyperparameters
train_hyperparams = {'num_epochs': 20,
                    'batch_size': 128,
                    'sample_interval': 500
}

# add lr-scheduling possibility
optimizator_hyperparams = {#'lr_gen': 0.0001,
                           #'lr_dis': 0.0005
                            'lr_gen': 0.001,
                            'lr_dis': 0.005
}

# dataset constants
dataset = 'CelebA'

datasets_hyperparams = {'CelebA': {'mean': [0.5061, 0.4254, 0.3828],
                                    'std': [0.3043, 0.2838, 0.2833],
                                    'path': 'celeba/',
                                    'img_shape': (64, 64)
                                    }
                        }
