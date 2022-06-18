from lib import *
# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image

for imagePath in glob.glob('data/train/*/*'):
    # get label from folder name
    label = imagePath.split("/")[-2]

    image = cv.imread(imagePath)
    try:
        # RGB to Gray
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Calculate Low and Up value to extract Edges
        md = np.median(gray)
        sigma = 0.35
        low = int(max(0, (1.0 - sigma) * md))
        up = int(min(255, (1.0 + sigma) * md))
        # Create Edged Image from Gray Scale
        edged = cv.Canny(gray, low, up)

        # extract only shape in image
        (x, y, w, h) = cv.boundingRect(edged)
        logo = gray[y:y + h, x:x + w]
        logo = cv.resize(logo, (200, 100))

        # Calculate histogram
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
model = KNeighborsClassifier(n_neighbors=1)
model.fit(hists, labels)
