from lib import *
# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image

model_path='model\\model.pkl'
for imagePath in glob.glob('data\\train\\*\\*'):
    label = imagePath.split("\\")[-2]
    image = cv.imread(imagePath)
    try:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        md = np.median(gray)
        sigma = 0.35
        low = int(max(0, (1.0 - sigma) * md))
        up = int(min(255, (1.0 + sigma) * md))
        edged = cv.Canny(gray, low, up)
        (x, y, w, h) = cv.boundingRect(edged)
        logo = gray[y:y + h, x:x + w]
        logo = cv.resize(logo, (200, 100))
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

# Create model as Nearest Neighbors Classifier
model = KNeighborsClassifier(n_neighbors=5)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(hists,labels,test_size=0.2)

model.fit(hists, labels)

y_pred_train=model.predict(X_train)
y_pred_test=model.predict(X_test)

cm_train=sklearn.metrics.classification_report(y_train,y_pred_train)
print(cm_train)
cm_test=sklearn.metrics.classification_report(y_test,y_pred_test)
print(cm_test)



joblib.dump(model,model_path)
# print()
######################################################################################
# Check Test Images for Model
model=joblib.load('model\\model.pkl')
# def test()
for (imagePath) in glob.glob('data\\test\\*'):
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
        cv.putText(image, predict.title(), (10, 30),
                   cv.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 4)

        # Get Image name and show Image
        imageName = imagePath.split("/")[-1]
        cv.imshow(imageName, image)
        cv.waitKey(0)
        # Close Image
        cv.destroyAllWindows()
    except cv.error:
        # If Image couldn't be Read
        print(imagePath)
        print("Test Image couldn't be read")

