

# specify project directory
PRJ='/home/thepp/projects/stanford_petct'
# define data base directory
INPUT='/mnt/data/datasets/Stanford/nifti'
MODELDIR='/mnt/data/datasets/tumorvolume/interim/petct/models/default_petct'

# define config file
export CONFIG=$PRJ'/config/stanford-petct.yaml'

# create aux directories
BASE=$(dirname $INPUT)
EXPORT=$BASE'/export'
CACHE=$BASE'/cache'
mkdir -p $EXPORT
mkdir -p $CACHE

HDF5=$CACHE'/assembly.h5'
KEYS=$CACHE'/keys'

# collect niftis files and store as hdf5 + key file 
python petct/convert.py --nii_dir $INPUT --hdf5 $HDF5 --keys $KEYS

# iterate over checkpoints
for CKPT in $MODELDIR'/17_MED-169/checkpoints/epoch=145.ckpt' $MODELDIR'/18_MED-174/checkpoints/epoch=119.ckpt'  $MODELDIR'/19_MED-175/checkpoints/epoch=111.ckpt' $MODELDIR'/20_MED-176/checkpoints/epoch=119.ckpt' $MODELDIR'/21_MED-177/checkpoints/epoch=113.ckpt'
do
	PAR="$(dirname -- "$CKPT")"
	PAR="$(dirname -- "$PAR")"	
	PAR="$(basename $PAR)"

	echo "processing checkpoint $CKPT ..."

	CACHE_RUN=$CACHE/$PAR
	EXPORT_RUN=$EXPORT/$PAR
	mkdir -p $CACHE_RUN
	mkdir -p $EXPORT_RUN

	PRED=$CACHE_RUN'/predictions.zip'
	POST=$CACHE_RUN'/postprocessing.zarr'

	# predict semgentation
	python $PRJ/external/torch-mednet/examples/predict.py base.data=$HDF5 prediction.test_set=$KEYS prediction.checkpoint=$CKPT prediction.data=$PRED
	# post-processing
	python $PRJ/petct/postprocessing.py postprocessing.data=$POST base.data=$HDF5 prediction.data=$PRED
	# export to nifti 
	python $PRJ/petct/export_nifti.py postprocessing.data=$POST base.data=$HDF5 prediction.data=$PRED export.directory=$EXPORT_RUN
done



