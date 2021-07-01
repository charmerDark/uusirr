# uusirr

### Steps to train model
- Clone the repository `git clone https://github.com/charmerDark/uusirr.git`
- All dependecies attached in requirements.txt <br>
  `cd uusirr && pip install -r requirements.txt`
- `wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip` downloads SINTEL dataset, please do this in the root folder of the repo.
- unzip packagge using `unzip MPI-Sintel-complete.zip -d uusirr/MPI_Sintel-complete/`
- run `bash scripts/new_model_train.sh` to train the model
<br>Pipline and Training scripts for trainging optical flow models. See `description.md` for details.
<br>
Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [IRR](https://github.com/visinf/irr)

