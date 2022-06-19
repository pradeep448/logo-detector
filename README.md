# Aliianz logo detector

## Folder structure
```bash
data            # All images stored here
    predict     # Images can be used for prediction    
    raw         # Raw images used for augmentation
    test        # Images used for validation
    train       # images used for train and test
model           # model directory
    model.pkl   # trained model
app.py          # flask application for web deployment
augment.py      # augmentation
lib.py          # library
main.py         # train, test, evaluate, validate
README.md       # Readme
requirements.txt    # python libraries to be installed
Procfile
```

## Prerequisites
### Note
This project has been tested on Windows 10 OS, Python 3.10.5 interpreter. 
### Classes
1: Logo Present

0: logo absent

## Steps
```bash
# Install Python 3.10.5

# get repo
git clone --branch master https://github.com/pradeep448/logo-detector.git
cd logo-detector 
# install libraries
pip install -r requirements
# create augmented images
python augment.py
# modelling, train, test, evaluate, serialize
python main.py
# run flask app locally (optional)
python app.py # to run flask app locally
```

## Test deployed model on Cloud
Open following URL:

https://allianz-logo-detector.herokuapp.com/



