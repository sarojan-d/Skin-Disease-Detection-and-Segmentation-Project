# For Skin disease detection and segmentation on ham10000 as major project 
comparison for yolo5 vs yolo7 detection </br>
comparison for u-net and dlabv3+ segmentation </br>
</br>
`prep.ipynb` first try </br>
`prepr.ipynb` 2nd with normalization at 1000 each done but pathing idk what do</br>
</br>

## for dataset
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
</br>

## for segmented dataset
https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations?resource=download

![image](https://github.com/user-attachments/assets/9ff33fba-4c73-4e98-93dd-9395b89f7ddf)
</br>


## for annotated dataset i think. (not used anymore)
https :// figshare. com /s / c7feb070d066a4ccce19
</br>

## test for `preprocessing test.ipynb` v1
![image](https://github.com/user-attachments/assets/cb866734-9e66-4f0f-b417-3c0cf6153d91)


###  multiple file augmentation test went like  `preprocessingmult.ipynb`
![image](https://github.com/user-attachments/assets/c1c3f14f-6e15-4f63-8596-5c11d7c2f13d)
![image](https://github.com/user-attachments/assets/abac0615-b2a7-42ff-8a1c-f116ecf7a174)

</br>
</br>
and outside this generated for some reason</br>

![image](https://github.com/user-attachments/assets/b6e42896-ea4d-4fdb-a072-d8d63c618377)

`copreprocess3.ipynb` generated preprocessed_images, annotations directoris as follows
![image](https://github.com/user-attachments/assets/ce1f0ba6-41c2-43d2-9243-138d52e8e88e)
but annotations don't match up with either original nor new preprocessed  images
![image](https://github.com/user-attachments/assets/6860552e-a588-4cb5-96d3-4cfd0b347b65)
and proprocessed ones don't ahve the dimensions at all
![image](https://github.com/user-attachments/assets/8a20a639-8eeb-4662-b503-e6e9530de75f)

# Scratch everything ig
after splitting ham10000 dataset into 7 folders by class
using `ypre.ipynb`

### 1 nv
![image](https://github.com/user-attachments/assets/8a362a8c-b381-4384-9fa0-943cec0e9ef0)
![image](https://github.com/user-attachments/assets/b371e90e-e74d-4eae-a785-4be247d9e01f)

### 2 
![image](https://github.com/user-attachments/assets/22d3d6dd-95f4-474e-8d7f-17bf5063638f)
![image](https://github.com/user-attachments/assets/079baa16-63dd-41ed-97fa-41c24a52cfaf)

### 3
![image](https://github.com/user-attachments/assets/b4a91141-e705-40b0-b3e3-2c6afcc20e6d)
![image](https://github.com/user-attachments/assets/35683501-1372-4626-8add-c6e0b7719c18)

### 4 
![image](https://github.com/user-attachments/assets/6b60c233-7eae-4eba-b04f-0d0ae9c3a9a1)
![image](https://github.com/user-attachments/assets/e70c822d-d275-42c3-b46f-cbd2e19ed681)

### 5 
![image](https://github.com/user-attachments/assets/71a1fbc4-b964-4bdf-a75d-cfdd5b638ddc)
![image](https://github.com/user-attachments/assets/b20929d6-e0a5-4f7e-b069-cb28085ea14c)

### 6
![image](https://github.com/user-attachments/assets/abba95f4-90af-4d6f-b898-63f2502adcd4)
![image](https://github.com/user-attachments/assets/c98ed191-6baf-43c3-b79d-89f8e4146706)

### 7 
![image](https://github.com/user-attachments/assets/d81a3920-f296-43a9-a6eb-2cddf1e98cbf)
![image](https://github.com/user-attachments/assets/3c952178-ace7-4368-b0c2-a5fd75c5f689)

## after augmentation as proposed target distribution:

nv: 10000 (original: 6705)
mel: 3000 (original: 1113)
bkl: 3000 (original: 1099)
bcc: 2000 (original: 514)
akiec: 1500 (original: 327)
vasc: 1000 (original: 142)
df: 1000 (original: 115)

### 1
![image](https://github.com/user-attachments/assets/f1085a87-c419-400c-938d-9f7c0ff496ee)
![image](https://github.com/user-attachments/assets/2bf3a5f9-b099-4bfb-bf05-3391a3e519f7)

### 2
![image](https://github.com/user-attachments/assets/f4f3bae6-45b7-47a1-b9a8-3abce6b7041d)
![image](https://github.com/user-attachments/assets/e60e1104-1a78-4361-88f6-3c6f65cd86d5)

### 3
![image](https://github.com/user-attachments/assets/81584f6b-9d5e-4041-a563-c7ec3c0b9b13)
![image](https://github.com/user-attachments/assets/49fe2657-428c-451d-b990-81acbbcc1ff3)

### 4 
![image](https://github.com/user-attachments/assets/8438a153-0119-4388-894e-882865ea4be5)
![image](https://github.com/user-attachments/assets/461d2ed2-8cfc-4ff2-8976-0cd680d2e365)

### 5 
![image](https://github.com/user-attachments/assets/f96f57f7-864a-4ca1-bd9d-dbf5c837e336)
![image](https://github.com/user-attachments/assets/96733c56-b7d9-4edb-9548-3aa30ab7acfa)

### 6
![image](https://github.com/user-attachments/assets/80767f60-e13b-4f76-8fee-e57828f37d9b)
![image](https://github.com/user-attachments/assets/8116cc36-c7de-4522-83b3-a6992553c761)

### 7 
![image](https://github.com/user-attachments/assets/d5932cae-5657-4c4d-b29d-a3b585cdcaa8)
![image](https://github.com/user-attachments/assets/f68925d2-5328-4018-ba92-fc3116a0f742)

</br>
annotations on images

![image](https://github.com/user-attachments/assets/4684875b-005e-41d0-abdc-08bae929a781)

more annotations
![image](https://github.com/user-attachments/assets/a8acb6ba-2390-4a1b-b3da-44f05d474fc7)
![image](https://github.com/user-attachments/assets/59e30f11-72fc-47e8-ac76-6d63e4fdfc69)


</br>

## shifted to colab
</br>
had to change labels from 1-7 to 0-6
</br>
stopped midway coz resources in colab finished

![image](https://github.com/user-attachments/assets/5a848332-0562-4391-a508-8717712796d8)


