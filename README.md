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

Use a pretrained model to predict msalign spectra from proteoform sequences and MS setting information. You can download a pretrained model td_pred_model.pth and a tsv file TopRepo_Fusion_Lumos_Q_Exactive_HF_val_v1.0.0.tsv with proteoform sequences [here](https://tulane.box.com/s/6pam0vzs618044vbt2y8bjuavxzl7fsn) and use the command below to predict spectra. The predicted spectra are stored in the file spectra_pred.msalign

```
python3 td-pred/src/model/td_pred.py --input TopRepo_Fusion_Lumos_Q_Exactive_HF_val_v1.0.0.tsv --model td_pred_model.pth --output spectra_pred.msalign
```

