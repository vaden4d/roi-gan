# checkpoints
logs_hyperparams = {'chkp_dir': 'chkp/',
                    'log_dir': 'logs/train_logs/',
                    # None if training from the scratch
                    'chkp_name_gen': None,
                    'chkp_name_dis': None
}

# check gpu devices
train_mode = {'gpu': True,
              'gpu_devices': '0',
              'multi_gpu': False
}

print_summary = False
model_hyperparams = {'clip_norm': 1e-2,
                     'roi_mode': 'duplicate', # 'duplicate' or 'full'
                     'function': 'gaussian_roi', # roi generation function
                     'gen_n_input': 4, # input random resolution
                     'gen_n_spade': 128, # spade
                     'gen_n_kernel': 7,  # n_features in generator
                     'dis_n_features': 128 # n_features in discriminator
}

stabilizing_hyperparams = {'adding_noise': True,
                            'fe_matching': True,
                            'roi_loss': False,
                            'n_layers_fe_matching': [4, 7]
}

# train/test hyperparameters
train_hyperparams = {'num_epochs': 100,
                    'batch_size': 128,
                    'sample_interval': 1
}

# add lr-scheduling possibility
optimizator_hyperparams = {'lr_gen': 0.00001,
                           'lr_dis': 0.00003,
                            'loss': 'ls' # vanilla, ls 
}

# dataset constants
dataset = 'CelebA'

datasets_hyperparams = {'CelebA': {'mean': [0.5061, 0.4254, 0.3828],
                                    'std': [0.3043, 0.2838, 0.2833],
                                    'path': 'celeba/',
                                    'img_shape': (64, 64)
                                    }
                        }
