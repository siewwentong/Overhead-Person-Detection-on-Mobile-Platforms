# Overhead-Person-Detection-on-Mobile-Platforms

Person detection is the core element in visual surveillance. This research focuses on the training of an existing object detection model, MobileNetV2 SSD, to perform detection overhead person on mobile platforms in unconstrained environments. The model is trained using a self-collected dataset that undergoes data augmentations via traditional augmentation methods and via StyleGAN2-ADA model. Results show that traditional augmentation techniques help in improving the model performance, but not StyleGAN2-ADA although it is able to generate realistic images. Transfer learning is also applied by freezing the pretrained model layers during training, and it is proved to be able to increase the training speed without affecting the model performance. A precision of over 90% and recall of almost 70% is achieved when tested in the target environment at a deployment speed greater than 30 frame per second.

Demo video (3 minutes) : https://youtu.be/B74YGQFrrkU

Example overhead images generated by StyleGAN2-ADA:
![alt text](EDE_stylegan.png?raw=true)
