from helpers import *
import re

config_file = sys.argv[1]

# Sample config file for debugging:
# config_file = 'data/sws2013-sample/test_config.yaml'

# Set up experiment from config file
config     = load_parameters(config_file)
output_dir = setup_exp(config)

# Load datasets
datasets    = load_std_datasets(config['datasets'], config['apply_vad'])
dataloaders = create_data_loaders(datasets, config)

if(config['mode'] == 'eval'):

    for ds_name, ds_loader in dataloaders.items():

        logging.info(" Starting evaluation on dataset '%s'" % (ds_name))
        csv_path = make_results_csv(os.path.join(output_dir, ds_name + '-results.csv'))
        logging.info(" Creating output file at '%s'" % (csv_path))

        # Can supply either a single .pt file or a folder containing many .pt files (e.g. checkpoints)
        # so set up empty array and check what class of object config['model_path'] is (directory or file?)
        model_paths = []

        # If configured to load a single model...
        if os.path.isfile(config['model_path']):
            model_paths.append(config['model_path'])

        # If given a directory of model checkpoints
        elif os.path.isdir(config['model_path']):
            model_paths = [ os.path.join(config['model_path'], m) for m in os.listdir(config['model_path']) ]
            model_paths.sort()

        # Evalulate each model on each dataset
        for model_path in model_paths:

            # Get epoch number from model file name ('model-e010.pt' => 10)
            # There's probably an easier way to do this
            epoch = int(re.search('model-e(\d+).pt', model_path).group(1))

            # Override config model path to current model being evaluated
            config['model_path'] = model_path
            
            model, _, _, scheduler = load_saved_model(config)

            with torch.no_grad():

                run_model(
                    model = model,
                    mode = 'eval',
                    ds_loader = ds_loader,
                    use_gpu = config['use_gpu'],
                    csv_path = csv_path,
                    keep_loss = False,
                    criterion = None,
                    epoch = epoch,
                    optimizer = None
                )

elif(config['mode'] == 'train'):

    # If configured to load a previous model...
    if('model_path' in config.keys()):
        model, optimizer, criterion, scheduler = load_saved_model(config)
    # If no previous model specified then load a new one according to config file
    else:
        model, optimizer, criterion, scheduler = instantiate_model(config)

    # Make CSV output file for training data
    train_csv_path = make_results_csv(os.path.join(output_dir, 'train_results.csv'), headers = 'train')

    # Check if there's a dev set defined in the config file. If so also make a CSV output file for dev data
    if('dev' in datasets.keys() and 'eval_dev_epoch' in config.keys()):
        dev_csv_path = make_results_csv(os.path.join(output_dir, 'dev_results.csv'), headers = 'train')
    else:
        # If no dev set defined, then no need to evaluate on dev set every nth epoch
        config['eval_dev_epoch'] = None

    for epoch in range(1, config['num_epochs'] + 1):

        # Provides a way to supply seperate (unevenly-sized) label files for positive and negative examples
        if isinstance(config['datasets']['train']['labels_csv'], dict):
            # If separate positive and negative label CSV files supplied, re-sample negatives at each epoch
            epoch_train_ds = load_std_datasets({'train' : config['datasets']['train']}, config['apply_vad'])
            epoch_train_dl = create_data_loaders(epoch_train_ds, config)['train']
        else:
            # Single label file for both positive and negative examples (i.e. new negatives not resampled at each epoch)
            epoch_train_dl = dataloaders['train']

        model, optimizer, criterion, loss, mean_loss = run_model(
            model = model,
            mode = 'train',
            ds_loader = epoch_train_dl,
            use_gpu = config['use_gpu'],
            csv_path = train_csv_path,
            keep_loss = True,
            criterion = criterion,
            epoch = epoch,
            optimizer = optimizer
        )

        logging.info(' Epoch: [%d/%d], Train Loss: %.4f' % (epoch, config['num_epochs'], mean_loss))

        if(epoch % config['save_epoch'] == 0):
            save_model(epoch, model, optimizer, loss, output_dir)

        # If this isn't the first epoch and there's a dev file then evaluate on dev set
        if(config['eval_dev_epoch'] is not None and epoch >= 1 and epoch % config['eval_dev_epoch'] == 0):
        
            with torch.no_grad():

                model, optimizer, criterion, loss, mean_loss = run_model(
                    model = model,
                    mode = 'eval',
                    ds_loader = dataloaders['dev'],
                    use_gpu = config['use_gpu'],
                    csv_path = dev_csv_path,
                    keep_loss = True,
                    criterion = criterion,
                    epoch = epoch,
                    optimizer = optimizer
                )

            logging.info(' Epoch: [%d/%d], Dev Loss: %.4f' % (epoch, config['num_epochs'], mean_loss))
        
        scheduler.step()
