Aliianz logo detector

Folder structure:

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
main.py         # tarin, test, evaluate, validate
README.md       # Readme
requirements.txt    # python libraries to be installed


Prerequisites:
NOTE: 
This project has been tested on Windows OS, Python 3.10.5 interpreter. 

Steps:
1. Install Python 3.10.5
2. cd REPO 
3. pip install -r requirements
4. python augment.py # create augemented images
5. python main.py # modelling, train, test, evaluate, serialize


Test model on Cloud:
URL: 


