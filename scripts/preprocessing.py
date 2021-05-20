import argparse
import pathlib as plb

import dicom2nifti
import nibabel as nib
import pydicom
import nilearn.image
import numpy as np
from   p_tqdm import p_map

import petct.data.suv as suv
from  petct.data.misc import reorient_nifti


def convert_to_nifti(dicom_dir, nifti_dir, verbose=False):
    # sort pet and ct dicom files in directory
    dicom_files = {'CT': [], 'PT': []}
    for f in dicom_dir.glob('*.dcm'):
        modality = pydicom.dcmread(str(f)).Modality
        assert modality in ['PT', 'CT']
        dicom_files[modality].append(str(f))

    if verbose:
        print(f'{dicom_dir}: {len(dicom_files["PT"])} PET dicom files, {len(dicom_files["CT"])} CT dicom files found')

    # convert dicom direcotry to nifti
    dicom2nifti.convert_directory(dicom_dir, nifti_dir, compression=True, reorient=True)

    if verbose:
        print(f'{dicom_dir}: computing SUV')

    # select sample pet dicom file to read header
    sample_dcm_file = dicom_files['PT'][0]
    suv_corr_factor = suv.calculate_suv_factor(sample_dcm_file)

    raw_pet_nii = next(nifti_dir.glob('*pet*'))
    suv_pet_nii = suv.convert_pet(nib.load(raw_pet_nii), suv_factor=suv_corr_factor)
    nib.save(suv_pet_nii, raw_pet_nii.parent/('suv_' + raw_pet_nii.name ))


def resample_nifti(ct_nii, suv_nii,
                   target_orientation, target_spacing):
    ct = nib.load(ct_nii)
    suv = nib.load(suv_nii)

    # reorient niftis
    ct = reorient_nifti(ct, target_orientation=target_orientation)
    suv = reorient_nifti(suv, target_orientation=target_orientation)

    # resample and align pet/ct
    orig_spacing = np.array(ct.header.get_zooms())
    orig_shape = ct.header.get_data_shape()
    target_affine = np.copy(ct.affine)
    target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ ct.affine[:3, :3]
    target_shape = (orig_shape*(orig_spacing/target_spacing)).astype(int)

    ct_rs = nilearn.image.resample_img(ct, target_affine, target_shape,
                                    interpolation='continuous',
                                    fill_value=-1024)
    ct_rs.set_data_dtype(np.float32)

    suv_rs = nilearn.image.resample_to_img(suv, ct_rs,
                                            interpolation='continuous',
                                            fill_value=0)
    suv_rs.set_data_dtype(np.float32)

    nib.save(ct_rs, ct_nii.parent/('rs_ct.nii.gz'))
    nib.save(suv_rs, suv_nii.parent/('rs_suv.nii.gz')) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom', type=str, default='/mnt/data/datasets/Stanford/dicom')
    parser.add_argument('--nifti', type=str, default='/mnt/data/datasets/Stanford/nifti') 
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--ct_nii_pattern', type=str, default='*gk*.nii.gz')
    parser.add_argument('--suv_nii_pattern', type=str, default='suv*.nii.gz')
    parser.add_argument('--orientation', type=str, default='LAS')
    parser.add_argument('--spacing', nargs='+', type=float, default=[2.0, 2.0, 3.0])
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    
    # get arguments
    verbose = args.verbose
    target_orientation = [t for t in args.orientation]
    target_spacing = args.spacing
    ct_nii_pattern = args.ct_nii_pattern
    suv_nii_pattern = args.suv_nii_pattern

    dicom = plb.Path(args.dicom)
    dicom_dirs = list(dicom.glob('*'))
    nifti = plb.Path(args.nifti)
    nifti.mkdir(exist_ok=True)

    def proc_dir(dicom_dir):
        if verbose:
            print(f'processing {dicom_dir}')

        nifti_dir = nifti/(dicom_dir.name)
        nifti_dir.mkdir(exist_ok=True)

        # convert dicom to nifti
        convert_to_nifti(dicom_dir, nifti_dir, verbose=verbose)

        # align and resample ct/suv niftis    
        ct_nii = next(nifti_dir.glob(ct_nii_pattern))
        suv_nii = next(nifti_dir.glob(suv_nii_pattern)) 
        resample_nifti(ct_nii, suv_nii, target_orientation, target_spacing)

    # version with multiprocessing
    added = p_map(proc_dir, dicom_dirs, num_cpus=args.jobs)

    # version without multiprocessing
    #for d in dicom_dirs:
    #    proc_dir(d)