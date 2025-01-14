For the data, please visit:
https://zenodo.org/records/8163751
https://zenodo.org/records/8170095



# Integrative Deep Learning Analysis Improves Colon Adenocarcinoma Patient Stratification at Risk for Mortality

This repository offers a comprehensive pipeline for survival analysis using machine learning models, specifically utilizing the [Keras library](https://keras.io/) backed by [TensorFlow](https://www.tensorflow.org/). The survival analysis is carried out on patient data which encompasses image features and clinical information. The model makes predictions based on these image features, and the resulting predictions are then analyzed with respect to patient survival times.

## Getting Started

Please follow the instructions in the main.py file to apply the pre-trained model to your data. If you wish to re-train the model or extract image features using a different pre-trained model, please refer to the respective scripts included in the repository.

## Input Data

The data for this project is organized as follows:

- `input/img_features.p`: A Pickle file containing the Inception-V3 extracted features from Hematoxylin and Eosin (H&E) stained images.
- `input/clinical.csv`: A CSV file containing clinical data about the patients.
- `input/pretrained/image_only.h5`: A pre-trained Keras model used for making predictions solely based on image data.

The input for this pipeline includes a pre-trained model, clinical data, and image features extracted using the Inception V3 transfer learning model that is pre-trained on ImageNet. The H&E stained images are normalized using a method proposed by Macenko, Marc, et al. in "A Method for Normalizing Histology Slides for Quantitative Analysis" (2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009). 

## Additional Codes

While the main focus of this repository is the application of the pre-trained model on user's own data, it also includes additional codes: 

- How to extract transfer learning image features using a pre-trained model.
- How to train the survival model based on these extracted features and clinical data.

These additional scripts offer more flexibility for users who might want to adapt this pipeline to their own datasets or use different pre-trained models for feature extraction.

## Note on Reproducibility

Due to the stochastic nature of the model and differences in computing environments, your results may not be exactly the same as ours, but they should be within a similar range.


## Citing this Repository

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{yourPaperIdentifier,
  title={Integrative Deep Learning Analysis Improves Colon Adenocarcinoma Patient Stratification at Risk for Mortality},
  author={Jie Zhou, Ali Foroughi pour, Hany Deirawan, Fayez Daaboul, Thazin Nwe Aung, Rafic Beydoun, Fahad Shabbir Ahmed, Jeffrey H.Chuang},
  journal={eBioMedicine},
  year={2023}
}
