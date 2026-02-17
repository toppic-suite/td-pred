# TopRepo

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


The training MS data can be found at toprepo.org.

## 2. Train the TD-Pred model


