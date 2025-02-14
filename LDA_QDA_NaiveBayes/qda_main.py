'''Note: To run all the models we can run the __init__.py'''
from utils import load_and_prepare_data
from models import QDAModel
import numpy as np

#defining function for RGB Model
def rgb_qda():
    print("----------------------------------------")
    print("*            QDA for RGB               *")
    print("----------------------------------------")
    
    X_train,y_train,X_test,y_test = load_and_prepare_data() #loading the data

    print("-----> Rgb Model is Training....")

    qda_rgb = QDAModel() #creating instance for the QDAModel Class

    qda_rgb.fit(X_train,y_train) #calling fit method for training data

    print("Rgb Model is Trained")

    print("-----> Now the model is Predicting...")
    
    y_pred = qda_rgb.predict(X_test) #calling the predict method for test data

    print("Predictions are completed.")

    if len(y_test) != len(y_pred):
        raise ValueError("Number of true labels does not match number of predicted labels.")
    
    correct_rgb = np.sum(y_test== np.array(y_pred))

    accuracy_rgb = correct_rgb / len(y_test) #calculating accuracy

    print("\n-----> Accuracy is: ",accuracy_rgb)

#defining function for Grayscale Model
def gray_qda():
        
    print("------------------------------------------")
    print("*            QDA for Grayscale           *")
    print("------------------------------------------")
    
     #Here 'True' indicates the conversion of RGB images into Grayscale images   
    X_gray_train,y_gray_train,X_gray_test,y_gray_test = load_and_prepare_data(True)#loading the data

    print("-----> Grayscale Model is Training....")

    qda_gray = QDAModel() #creating an instance for QDAModel class

    qda_gray.fit(X_gray_train,y_gray_train) #calling fit method for training data

    print("Grayscale Model is Trained")

    print("-----> Now the Model is predicting....")

    y_gray_pred = qda_gray.predict(X_gray_test) #calling predict method test data

    print("The Predictions are completed.")

    if len(y_gray_test) != len(y_gray_pred):
        raise ValueError("Number of true labels does not match number of predicted labels.")
            

    correct_gray =  np.sum(np.array(y_gray_test)==y_gray_pred)

    accuracy_gray = correct_gray/len(y_gray_test) #calculating accuracy

    print("\n-----> Accuracy is: ",accuracy_gray)

def main():
    print("****************************************")
    print("*         RGB Model Solution           *")
    print("****************************************")
    rgb_qda()

    print("\n******************************************")
    print("*       Grayscale Model Solution         *")
    print("******************************************")
    gray_qda()


if __name__ == "__main__":
    main()