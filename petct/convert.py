import argparse
import pathlib as plb
from tqdm import tqdm
from typing import Type

import h5py
import numpy as np
import nibabel as nib


def normalize_nifti(img, arr_min=-1000, arr_max=1000, dtype=np.float32):
	affine = img.affine
	arr = img.get_fdata()
	arr -= arr_min
	arr *= (1 / (arr_max - arr_min))
	arr = np.clip(arr, 0, 1)
	return nib.Nifti1Image(arr.astype(dtype), affine)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--nii_dir', type=str)
	parser.add_argument('--hdf5', type=str)
	parser.add_argument('--keys', type=str)	
	parser.add_argument('--norm_ct', nargs='+', default=(-1000.0,1000.0))
	parser.add_argument('--norm_pet', nargs='+', default=(0.0, 40.0))
	args = parser.parse_args()

	# parse all subfolders in 'nii_dir', get rsCT.nii.gz & rsSUV.nii.gz
	# normalize ct and petsuv and stack both images 
	# store stacked arrays as hdf5 group ('image') and add the affine matrix as attribute
	# the directory name is used as id  
	print('Collecting nifti files ..')
	root_dir = plb.Path(args.nii_dir)
	with h5py.File(args.hdf5, 'w') as hf:
		image_group = hf.require_group('image')
		for d in tqdm(list(root_dir.glob('*'))):
			if(d.is_dir()):
				id = d.name 
				ct = nib.load(d/'rsCT.nii.gz')
				ct = normalize_nifti(ct, args.norm_ct[0], args.norm_ct[1])
				pet = nib.load(d/'rsSUV.nii.gz') 
				pet = normalize_nifti(pet, args.norm_pet[0], args.norm_pet[1])	
				data = np.stack([pet.get_fdata(), ct.get_fdata()], axis=0).astype(np.float16)
				affine = ct.affine 
				ds = image_group.require_dataset(id, data.shape, data.dtype)
				ds[:] = data
				ds.attrs['affine'] = affine.tolist()

	# create list with all keys and store as files
	with h5py.File(args.hdf5, 'r')  as hf:
		keys = list(hf['image'])
		with open(args.keys, 'w') as f:
			f.writelines([k + '\n' for k in keys])


if __name__ == '__main__':
	main()
