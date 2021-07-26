

PRJ='/home/thepp/projects/stanford_petct'
export CONFIG=$PRJ'/config/stanford-petct.yaml'

BASE='/mnt/data/datasets/Stanford'
mkdir -p $BASE'/cache'
mkdir -p $BASE'/export'
INPUT=$BASE'/nifti'
HDF5=$BASE'/cache/assembly.h5'
KEYS=$BASE'/cache/keys'
PRED=$BASE'/cache/predictions.zip'
POST=$BASE'/cache/postprocessing.zarr'
EXPORT=$BASE'/export'

MODELDIR='/mnt/data/datasets/tumorvolume/interim/petct/models'
#CKPT=$MODELDIR'/default_petct/18_MED-174/checkpoints/epoch=119.ckpt'
CKPT=$MODELDIR'/default_petct/19_MED-175/checkpoints/epoch=111.ckpt'

# collect niftis files and store as hdf5 + key file 
python petct/convert.py --nii_dir $INPUT --hdf5 $HDF5 --keys $KEYS

# predict semgentation
python $PRJ/external/torch-mednet/examples/predict.py base.data=$HDF5 prediction.test_set=$KEYS prediction.checkpoint=$CKPT prediction.data=$PRED

# post-processing
python $PRJ/petct/postprocessing.py postprocessing.data=$POST base.data=$HDF5 prediction.data=$PRED

# export 
python $PRJ/petct/export_nifti.py postprocessing.data=$POST base.data=$HDF5 prediction.data=$PRED export.directory=$EXPORT