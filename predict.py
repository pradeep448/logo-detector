from lib import *

model=joblib.load('model\\model.pkl')
# print(model.predict(sys.argv[1]))



for (imagePath) in glob.glob('data\\predict\\*'):
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

