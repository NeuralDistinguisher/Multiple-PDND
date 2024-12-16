#import simon as sp
#import simeck as sp
import speck as sp
import numpy as np
from tensorflow.keras.models import load_model

'''
To evaluate the models or run real polytopic differences experiment, comment out all sections related to network definition and training in the relevant function of the cipher algorithm analysis code. 
Then, execute the 'eval.py' script in the directory of the desired cipher.
'''

def evaluate_model():
    try:
        # Load the trained model
        model_path = './results_model_round5.h5'
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Generate test data
        round_number = 5  
        test_samples = 10**6  
        s_groups = 1  

        print(f"Generating test data for {test_samples} samples and {round_number} rounds...")
        X_test, Y_test = sp.make_train_data(test_samples, round_number, s_groups=s_groups)
        print("Test data generated successfully.")

        # Verify the shape of the test data
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

        # Evaluate the model
        print("Evaluating the model...")
        results = model.evaluate(X_test, Y_test, verbose=1)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        print(f"True Positives: {results[2]}")
        print(f"True Negatives: {results[3]}")
        print(f"False Positives: {results[4]}")
        print(f"False Negatives: {results[5]}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    evaluate_model()
