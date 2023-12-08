Mosquitoes, recognized as small yet perilous insects, play a crucial role in transmitting diseases that pose significant threats to both humans and the environment. The authors of the **MosquitoAlert 2023: Small Object Detection and Classification Challenge (Phase 2)** emphasize the importance of controlling mosquito populations, with over 3600 species capable of transmitting pathogens leading to diseases such as Zika, Dengue, and Chikungunya. The challenge aims to directly impact public health initiatives by leveraging machine learning and deep learning techniques to automate the image validation process for mosquito identification.

## **Dataset**

The dataset comprises 10,700 real-world images of mosquitoes captured by citizens using mobile phones. Split into *train* (80%) and test (20%) sets, the dataset offers a diverse representation of mosquitoes in different scenarios and locations. Each image is labeled with bounding box coordinates and mosquito class information. <i>Please note, that in the phase 2, test images were [integrated](https://discourse.aicrowd.com/t/important-updates-for-round-2/9078) into the train split</i>

## **Ground Truth**

Accurate annotations are provided by expert entomologists. It's noted that while most images contain a single mosquito with its corresponding bounding box and class label, rare cases may depict multiple mosquitoes. **For consistency, only one bounding box and class label are assigned per image, even if multiple mosquitoes are visible.**