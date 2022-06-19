from lib import *

def augment(data_dir):
    
    dir = f'{data_dir}{path_div}train{path_div}1'
    # clear directory
    try:
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    except:
        pass
    # genrate augmented images 
    datagen = ImageDataGenerator(
            rotation_range=4,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            )

    img_path=abspath(f'data{path_div}raw{path_div}download_cleaned.png')
    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)  

    print('INFO: Augmenting...')
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=dir, save_prefix='logo', save_format='jpeg'):
        i += 1
        if i > 500:
            break  # otherwise the generator would loop indefinitely
    source_dir=data_dir+f'{path_div}raw'
    destination_dir=dir
    shutil.copytree(source_dir, destination_dir,dirs_exist_ok=True)
    print('INFO: Augmentation completed')
augment('data')
