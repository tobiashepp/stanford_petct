import nibabel as nib
import numpy as np
from nibabel.orientations import ornt_transform, axcodes2ornt, inv_ornt_aff, apply_orientation, io_orientation, aff2axcodes

def reorient_nifti(img,
                   target_orientation=('L','A','S'),
                   verbose=False):
    new_ornt = axcodes2ornt(target_orientation)
    vox_array = img.get_fdata()
    affine = img.affine
    orig_ornt = io_orientation(img.affine)
    ornt_trans = ornt_transform(orig_ornt, new_ornt)
    orig_shape = vox_array.shape
    new_vox_array = apply_orientation(vox_array, ornt_trans)
    aff_trans = inv_ornt_aff(ornt_trans, orig_shape)
    new_affine = np.dot(affine, aff_trans)
    if verbose:
        print(f'{aff2axcodes(affine)} -> {aff2axcodes(new_affine)}')
    new_img = nib.Nifti1Image(new_vox_array, new_affine, img.header)
    return new_img