import argparse
import json
import os.path
from zipfile import ZipFile

import gdown
import wget

with open('./asid_paths.json') as f:
    download_paths = json.load(f)

gdrive_prefix = 'https://drive.google.com/uc?id='


def download(source, output_dir, unpack_zips, remove_zips):
    for file in download_paths:
        output = os.path.join(output_dir, download_paths[file]['filename'])
        if source == 'gdrive':
            gdown.download('{}{}'.format(gdrive_prefix,
                                         download_paths[file]['gdrive']),
                           output,
                           quiet=False)
        else:
            wget.download(download_paths[file]['zenodo'], output)

        if download_paths[file]['filename'].endswith('.zip'):
            if unpack_zips:
                unpack_zip(output, output_dir)
            if remove_zips:
                remove_zip(output)


def unpack_zip(output, output_dir):
    print('Unpacking the zip file {} into {}'.format(output, output_dir))
    with ZipFile(output, 'r') as z:
        z.extractall(output_dir)


def remove_zip(output):
    print('Removing the zip file {}'.format(output))
    os.remove(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download script for ASID',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source',
                        default='gdrive',
                        choices=['gdrive', 'zenodo'],
                        help='from which source to download the files. you can either download from Google Drive '
                             '(gdrive) or from Zenodo (zenodo)')
    parser.add_argument('--outputdir',
                        default='./data/',
                        help='directory to store the dataset')
    parser.add_argument('--unpack', action='store_true', help='unpack the zip files')
    parser.add_argument('--remove', action='store_true', help='remove zip files after unpacking')

    args = parser.parse_args()

    if args.source == 'zenodo':
        raise Exception('Currently, we only support downloads from Google Drive.')

    if not os.path.exists(args.outputdir):
        raise Exception('The specified directory for storing the dataset does not exist.')

    download(args.source, args.outputdir, args.unpack, args.remove)