import nibabel as nib


import os
from pathlib import Path

import hydra
import numpy as np
import zarr
import dotenv
import h5py
dotenv.load_dotenv()


@hydra.main(config_path=os.getenv('CONFIG'), strict=False)
def main(cfg):
	image_data = cfg.base.data 
	image_group = cfg.base.image_group
	prediction_data = cfg.prediction.data
	prediction_group = cfg.prediction.group
	postprocessed_data = cfg.postprocessing.data 
	postprocessed_group = cfg.postprocessing.group
	export_dir = Path(cfg.export.directory)

	with zarr.open(store=zarr.ZipStore(prediction_data), mode='r') as zf:
		with zarr.open(store=zarr.DirectoryStore(postprocessed_data), mode='r') as zf_post:
			keys = list(zf[prediction_group])
			for k in keys:
				affine = np.array(zf[prediction_group][k].attrs['affine'])
				data = zf[prediction_group][k][0, :]
				nib.save(nib.Nifti1Image(data, affine), export_dir/f'{k}_prediction.nii.gz')
				data = zf_post[postprocessed_group][k][0, :]
				nib.save(nib.Nifti1Image(data, affine), export_dir/f'{k}_postprocessed.nii.gz')

	with h5py.File(image_data, 'r') as hf:
		keys = list(hf[image_group])
		for k in keys:
			affine = np.array(hf[image_group][k].attrs['affine'])
			data = hf[image_group][k][0, :].astype(np.float32)
			nib.save(nib.Nifti1Image(data, affine), export_dir/f'{k}_image_petsuv.nii.gz')
			data = hf[image_group][k][1, :].astype(np.float32)
			nib.save(nib.Nifti1Image(data, affine), export_dir/f'{k}_image_ct.nii.gz')



if __name__ == '__main__':
    main()