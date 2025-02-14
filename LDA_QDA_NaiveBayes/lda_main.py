'''Note: To run all the models we can run the __init__.py'''
from utils import load_and_prepare_data
from models import LDAModel
import numpy as np

#defining function for RGB Model
def rgb_lda():
    print("----------------------------------------")
    print("*            LDA for RGB               *")
    print("----------------------------------------")
    
    X_train,y_train,X_test,y_test = load_and_prepare_data() #loading the data 

    print("-----> Rgb Model is Training....")

    lda_rgb = LDAModel() #creating instance for the LDAModel Class

    lda_rgb.fit(X_train,y_train) #calling fit method for training data
    
    print("Rgb Model is Trained")

    print("-----> Now the model is Predicting...")
    
    y_pred = lda_rgb.predict(X_test) #calling the predict method for test data

    print("Predictions are completed.")

    if len(y_test) != len(y_pred):
        raise ValueError("Number of true labels does not match number of predicted labels.")
    
    correct_rgb = np.sum(y_test == y_pred)

    accuracy_rgb = correct_rgb / len(y_test) #calculating accuracy
    print("\n-----> Accuracy is: ",accuracy_rgb)

#defining function for Grayscale Model
def gray_lda():
        
    print("------------------------------------------")
    print("*            LDA for Grayscale           *")
    print("------------------------------------------")
    
    #Here 'True' indicates the conversion of RGB images into Grayscale images
    X_gray_train,y_gray_train,X_gray_test,y_gray_test = load_and_prepare_data(True) #loading the data

    print("-----> Grayscale Model is Training....")

    lda_gray = LDAModel() #creating an instance for LDAModel class

    lda_gray.fit(X_gray_train,y_gray_train) #calling fit method for training data

    print("Grayscale Model is Trained")

    print("-----> Now the Model is predicting....")

    y_gray_pred = lda_gray.predict(X_gray_test) #calling predict method test data

    print("The Predictions are completed.")

    if len(y_gray_test) != len(y_gray_pred):
        raise ValueError("Number of true labels does not match number of predicted labels.")
            

    correct_gray =  np.sum(y_gray_test == y_gray_pred)

    accuracy_gray = correct_gray/len(y_gray_test) #calculating accuracy

    print("\n-----> Accuracy is: ",accuracy_gray)

def main():
    print("****************************************")
    print("*         RGB Model Solution           *")
    print("****************************************")
    rgb_lda()

    print("\n******************************************")
    print("*       Grayscale Model Solution         *")
    print("******************************************")
    gray_lda()


if __name__ == "__main__":
    main()