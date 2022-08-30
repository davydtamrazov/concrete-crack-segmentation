# Concrete Crack Segmentation
Semantic segmentation of cracks in concrete using U-Net based Fully Convolutional Network.


## Dataset
Original dataset is 458 high-resolution (4032x3024) concrete crack images obtained from https://data.mendeley.com/datasets/jwsn7tfbrp/1. Dataset used for this project was generated as follows:
1. Image is scaled down and split up into a 224x224 grid.
2. Segmentation mask is used to select patches on the grid that contain target class (i.e. crack is present)


![Pre-processing of the original dataset downscaled by a factor of 0.5](./aux/data_preprocessing_example.png)

To augment the dataset image resolution is downscaled by multiple factors (0.125, 0.25, 0.5), thereby allowing to create images of cracks of various sizes. Additionally, patches are generated with an overlap of 112 pixels, helping to almost double dataset size. As a result generated dataset contains 


# References

Özgenel, Çağlar Fırat (2019), 
“Concrete Crack Segmentation Dataset”, 
Mendeley Data, V1, doi: 10.17632/jwsn7tfbrp.1


