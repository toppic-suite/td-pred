# TD-Pred

TD-Pred is a deep learning model for predicting top-down MS/MS spectra from
proteoform sequences.  Proteoform sequences are encoded using one-hot encoding
combined with residue mass, positional, and length features. This encoding is
processed by a CNN subnetwork consisting of eight parallel modules with kernel
sizes ranging from 2 to 9, enabling the model to capture local sequence
dependencies up to four residues on each side. The CNN outputs are concatenated
with the original sequence encoding. In addition, meta-information is encoded
and appended to each column of the sequence matrix, allowing the
transformer layers to access global meta-features at every sequence position.
The final representation is input into a transformer architecture comprising
six encoder and six non-autoregressive decoder layers for spectral prediction

## Release 1.3: retraining required

Release 1.3 fixes several bugs in the sequence and meta-information encoding:

- The instrument type is now one-hot encoded (it was always all zeros before).
- The residue position feature now spans [-1, 1] over the sequence (it was offset by one residue).
- Modified residues written in lowercase (e.g. `m` for oxidized Met) are now encoded instead of skipped.

These changes alter the model input. HDF5 training files and model files produced with
release 1.2.x or earlier are **not compatible** with release 1.3. To use release 1.3:

1. Regenerate HDF5 files from the annotated msalign files (Step 1 below).
2. Retrain the model on the regenerated files (Step 2 below).

The pretrained model `td_pred_backbone_model_1.2.1.pth` linked below was trained with the old
encoding and should only be used with release 1.2.x code.

## 1. Generate training MS data
Convert an annotated msalign file to a hdf5 file.

```
python3 td-pred/src/msalign/msalign_anno_to_hdf5.py --msalign spectra_anno_ms2.msalign --out spectra.hdf
```

The training and validation MS data files used in the TopRepo paper can be downloaded  
[here](https://tulane.box.com/s/6pam0vzs618044vbt2y8bjuavxzl7fsn).

## 2. Train the TD-Pred model

Train the TD-Pred model using a training dataset spectra_train.hdf and a validation dataset spectra_val.hdf. The output model is stored in the file td_pred_model.pth

```
python3 td-pred/src/model/train_td_pred.py --train spectra_train.hdf --validate spectra_val.hdf 
```

A trained spectral prediction model can be downloaded [here](https://tulane.box.com/s/6pam0vzs618044vbt2y8bjuavxzl7fsn).


## 3. Predict msalign spectra from sequences

Use a pretrained model to predict msalign spectra from proteoform sequences and MS setting information. You can download a pretrained model td_pred_backbone_model_1.2.1.pth and a tsv file CID_HCD_VAL_1.2.1.tsv with proteoform sequences [here](https://tulane.box.com/s/6pam0vzs618044vbt2y8bjuavxzl7fsn) and use the command below to predict spectra. The predicted spectra are stored in the file spectra_pred.msalign

```
python3 td-pred/src/model/td_pred.py --input CID_HCD_VAL_1.2.1.tsv --model td_pred_backbone_model_1.2.1.pth --output spectra_pred.msalign
```
