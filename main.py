"""
This is main script which performs following tasks
STEPS:
    Train
    Test
    Evaluate
    Serialize
    Validate
"""

# import library
from lib import *

def train_test_evaluate(data_folder_path):

    ############# TRAIN & TEST ##############
    hists = []  # histogram of Image
    labels = []  # Label of Image
    print('INFO: Training...')
    for imagePath in glob.glob(f'{data_folder_path}{path_div}train{path_div}*{path_div}*'):
        # labe 1 or 0
        # 1 logo present
        # 0 logo absent
        label = imagePath.split(f"{path_div}")[-2]
        image = cv.imread(imagePath)
        try:
            # convert to gray
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # median
            md = np.median(gray)
            # find edges
            sigma = 0.35
            low = int(max(0, (1.0 - sigma) * md))
            up = int(min(255, (1.0 + sigma) * md))
            edged = cv.Canny(gray, low, up)
            # bound logo
            (x, y, w, h) = cv.boundingRect(edged)
            logo = gray[y:y + h, x:x + w]
            logo = cv.resize(logo, (200, 100))
            # get histogram
            hist = feature.hog(
                logo,
                orientations=9,
                pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm="L1"
            )
            # Add value into Lists
            hists.append(hist)
            labels.append(label)
        except cv.error:
            # If Image couldn't be Read
            print(imagePath)
            print("Training Image couldn't be read")

    # modelling
    model=RandomForestClassifier()
    X_train,X_test,y_train,y_test=train_test_split(hists,labels,test_size=0.2)
    # training
    model.fit(hists, labels)
    # prediction
    # train data
    print('INFO: Predicting...')
    y_pred_train=model.predict(X_train)
    # test data
    y_pred_test=model.predict(X_test)
    # evaluate
    print('INFO: Evaluating...')
    acc_train=sklearn.metrics.accuracy_score(y_test,y_pred_train)
    f1_train=sklearn.metrics.f1_score(y_test,y_pred_train)
    acc_test=sklearn.metrics.accuracy_score(y_test,y_pred_test)
    f1_test=sklearn.metrics.f1_score(y_test,y_pred_test)
    print(f'Training: Accuracy = {acc_train}, f1 score = {f1_train}')
    print(f'Testing : Accuracy = {acc_test}, f1 score = {f1_test}')
    # serialize
    print('INFO: Serializing...')
    joblib.dump(model,model_path)
    print(f'INFO: Model saved to {model_path}')

    ######### VALIDATE ##########
    print('INFO: Validating...')
    print('INFO: To stop viewing image predictions, Press ctrl+C or Ctrl+Z on terminal and then close cv image window.')
    for (imagePath) in glob.glob(f'data{path_div}test{path_div}*'):
        # Read Images
        image = cv.imread(imagePath)
        try:
            # Convert to Gray and Resize
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            logo = cv.resize(gray, (200, 100))

            # Calculate Histogram of Test Image
            hist = feature.hog(
                logo,
                orientations=9,
                pixels_per_cell=(10, 10),
                cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm="L1"
            )
            # Predict in model
            predict = model.predict(hist.reshape(1, -1))[0]
            # Make pictures default Height
            height, width = image.shape[:2]
            reWidth = int((300/height)*width)
            image = cv.resize(image, (reWidth, 300))

            # Write predicted label over the Image
            # 1 -> allianz logo present
            # 0 -> logo absent
            cv.putText(image, mapper(predict), (10, 30),
                    cv.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 4)

            # Get Image name and show Image
            imageName = imagePath.split(path_div)[-1]
            cv.imshow(imageName, image)
            cv.waitKey(0)
            # Close Image
            cv.destroyAllWindows()
        except cv.error:
            # If Image couldn't be Read
            print(imagePath)
            print("Test Image couldn't be read")

train_test_evaluate(abspath('data'))

