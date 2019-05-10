import os
import json
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.translator as module_translator
from config_parser import ConfigParser


def main(config: ConfigParser, resume: str):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        src_path=config['data_loader']['args']['test_src_path'],
        tgt_path=config['data_loader']['args']['test_tgt_path'],
        src_preprocessor_path=config['data_loader']['args']['src_preprocessor_path'],
        tgt_preprocessor_path=config['data_loader']['args']['tgt_preprocessor_path'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        num_workers=1,
    )

    # build model architecture
    model = config.initialize('arch', module_arch)
    print(model)

    # load state dict
    print(f'Loading checkpoint: {resume}')
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    trans_args = {
        'model': model,
        'src_preprocessor': data_loader.src_text_preprocessor,
        'tgt_preprocessor': data_loader.tgt_text_preprocessor,
    }
    translator = config.initialize('translator', module_translator, *trans_args.values())

    # prepare output file
    out_f = (config.test_dir / config['translator']['output']).open('w')

    with torch.no_grad():
        for batch_idx, (src, tgt, lengths, indices) in enumerate(tqdm(data_loader)):
            src, tgt = src.to(device), tgt.to(device)
            pred_batch, _, _ = translator.translate(src, None, lengths, indices)

            for b in range(len(pred_batch)):
                out_f.write(' '.join(pred_batch[b][0]) + '\n')

    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str, required=True,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, required=True,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser.parse(args)

    main(config, args.resume)
