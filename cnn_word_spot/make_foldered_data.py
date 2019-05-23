import os
import shutil


def makefolders(folder_name):
    image_classes={}
    for image_name in sorted(os.listdir(folder_name)):
        class_name=image_name.split('_')[0]
        if class_name in image_classes:
            image_classes[class_name].append(image_name)
        else:
            image_classes[class_name]=[image_name]
    if not os.path.isdir('foldered_data'):
        os.makedirs('foldered_data')
        class_list = sorted(list(image_classes.keys()))
        for c_name in class_list:
            os.makedirs('foldered_data/'+c_name)
            images_in_class = image_classes[c_name]
            for i_name in images_in_class:
                shutil.copyfile(folder_name+'/'+i_name,'foldered_data/'+c_name+'/'+i_name)

makefolders('TEST')