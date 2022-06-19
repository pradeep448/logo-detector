from lib import *

def augment(data_dir):
    
    dir = f'{data_dir}{path_div}train{path_div}1'

    try:
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    except:
        pass

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

    # img_path='D:\\CV_master\\comp\\allianz\\REPO\\data\\raw\\download_cleaned.png'
    img_path=abspath(f'data{path_div}raw{path_div}download_cleaned.png')
    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
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