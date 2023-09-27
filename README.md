
**UCB Data Analysis Module 21**
# Deep Learning Challenge / Report

---------------
---------------
## To Fund or Not to Fund: Predicting Applicant Success for Alphabet Soup 

### Overview of the Analysis:
The purpose of this analysis is to develop a machine learning model that can predict the success of organizations applying for funding from Alphabet Soup. The goal is to create a binary classifier that can identify successful candidates with 75% (or higher) accuracy based upon metadata collected from previously funded applicants. This predictive model can assist Alphabet Soup in making more informed decisions about which organizations to fund, thereby optimizing their impact and resources.  The following report details the steps taken to achieve this goal.

--------------
## Results:  

### Data Preprocessing 

- Target Variable:
  - The target variable for the model is 'IS_SUCCESSFUL', which indicates whether an organization was successful after receiving funding (1 for successful, 0 for not successful).
- Features:
  - Features include attributes such as APPLICATION_TYPE, AFFILIATION (primarily company-sponsored or independent), CLASSIFICATION (government org classification), USE_CASE, ORGANIZATION (corporate, trust, co—op or association), STATUS, INCOME_AMT and ASK_AMT (amount requested).
- Variables Removed:
  - Two non-beneficial columns, 'EIN' and 'NAME', were removed from the input data as they did not provide meaningful information for prediction.

### Compiling, Training, and Evaluating the Model  

- The optimized neural network model has three hidden layers. Adding a third layer resulted in a marginal improvement and, thus, fulfilled my criteria for retention.  Note: This criteria is discussed in greater detail below.
  - 1st Layer: 16 neurons using "tanh" activation function.
  - 2nd Layer: 6 neurons with a "relu" activation function.
  - 3rd Layer: 4 neurons with a "relu" activation function (added for optimization).
- The model makes use of two activation functions, “tanh” and “relu”.  These are two of the most commonly used activation functions.
  - **"tanh"** (hyperbolic tangent) squashes the output and is often used in the first hidden layer as it helps to normalize the input data and can help with training stability.
  - **"relu"** (rectified linear unit) is capable of handling sparse activations and is often used in subsequent hidden layers as it allows the model to learn non-linear relationships in the dataset.

- #### Model Performance:
  - Initially, the model's accuracy was around 72.83%, which did not meet the target performance of >75%. Following multiple attempts at optimization, the accuracy was only marginally increased to 73.24 %.
 
- #### Optimization Attempts:
  Numerous strategies were implemented in an attempt to attain target performance for the model:  
    1. **Attempt_00: Accuracy: 0.732478141784668**  
       Ironically, my first attempt earned the highest accuracy (73.24%) but I was careless in my approach, implementing multiple adjustments in one step.  As such, it was unclear as to which variable(s) were having a positive impact so I resorted to a more methodical approach.
    2. Starting with the original model, I systematically made small adjustments to the input data and recorded the accuracy for each.  If the accuracy improved, even marginally, the adjustment was retained and implemented in each subsequent attempt.  If the accuracy decreased, the adjustment was rolled back.
    3. **Attempt_01: Accuracy: 0.6468804478645325**  
       - Details:  Drop more columns from the input data.  Specifically, 'EIN', 'NAME', 'SPECIAL_CONSIDERATIONS', 'AFFILIATION', and 'USE_CASE'.
       - Outcome:  REJECT - Performance decreased significantly.
    5. **Attempt_02: Accuracy: 0.6468804478645325**
       - Details:  Bin fewer (rare occurrence) columns. Specifically, only bin columns 'T25', 'T14', 'T29', 'T15' and 'T17'.
       - Outcome:  REJECT - No impact on performance.
    6. **Attempt_03: Accuracy: 0.6468804478645325**  
       - Details:  Decrease # of values per bin from <1000 to <150.
       - Outcome: REJECT - Slight decrease in performance.
    8.	**Attempt_04: Accuracy: 0.7294460535049438**
       - Details: Increase neurons in 2nd hidden layer from 4 to 6.
      	- Outcome: ACCEPT - Slight increase in performance.  
    10.	**Attempt_05: Accuracy: 0.7282798886299133**
        - Details: Bin more (rare occurrence) columns. Specifically, bin columns 'T10', 'T9', 'T13', 'T12', 'T2', 'T25', 'T14', 'T29', 'T15' and 'T17'.
       	- Outcome: REJECT - No impact on performance.
    12.	**Attempt_06: Accuracy: 0.7315452098846436**
        - Details: Increase neurons in 1st hidden layer from 8 to 16.
       	- Outcome: ACCEPT - Slight increase in performance.
    14.	**Attempt_07: Accuracy: 0.7289795875549316**
        - Details: Add 3rd hidden layer with ‘relu’ activation.
       	- Outcome: ACCEPT – While exported (.h5) accuracy shows no significant improvement, I recorded the accuracy of two prior runs***** at 73.30% and 73.00%. As such, I elected to retain this adjustment.
    16.	**Attempt_08: Accuracy: 0.727580189704895**
        - Details: Change activation function of 1st hidden layer from ‘relu’ to ‘tanh’.
       	- Outcome: ACCEPT - While exported (.h5) accuracy shows no significant improvement, I recorded the accuracy of a prior run***** at 73.14%. As such, I elected to retain this adjustment.
    18.	**Attempt_09: Accuracy: 0.7241982221603394**
        - Details: Increase # of epochs in training regimen from 100 to 140.
       	- Outcome: REJECT - No performance improvement.
    20.	**Attempt_10: Accuracy: 0.7300291657447815**
        - Details: Attempting to leverage a prior successful optimization, I increased the neurons in 2nd hidden layer from 6 to 8 AND increased the # of epochs from 100 to 125. The latter was an experiment.
       	- Outcome: REJECT - No performance improvement.

*****While working in Google Colaboratory, I did not understand that the exported HDF5 files are stored in the Colab Hosted (temporary) VM. When this notebook times out (a.k.a. "Runtime Disconnected"), the VM is destroyed along with any files created/exported.  For this reason, I actually ran the code for each optimization attempt a total of 3 times.  Since I lost the HDF5 exports for the first two runs, I am only able to submit the third and final run for each.  However, for instances in which I recorded improved performance in both of the prior runs, I opted to retain that attempt at optimization.  None of my attempts to reach target performance were successful and I figured it couldn’t hurt to try.  

--------------
## Summary:  

In summary, despite multiple optimization attempts, I was unable to achieve the target performance of >75% accuracy with the neural network model.
While the first optimization model improved accuracy (73.24%) compared to the initial version (72.83%), further enhancements may be necessary to meet the desired threshold. It would be a worthwhile endeavor to consider alternative approaches.  

### Recommendations

- Adjust Batch Size:  A smaller batch size during training can sometimes improve upon the performance of a model.
- Regulate Class Imbalance (if any): may want to explore techniques like oversampling or undersampling to balance the dataset.
- Different Optimizers: Experiment with different optimizers, such as Nadam, in addition to Adam for improved convergence.

In conclusion, while the neural network model showed promise, it did not reach target accuracy. Implementing alternative training strategies, preprocessing techniques or optimizers could help to achieve the desired accuracy for Alphabet Soup's predictive algorithm.


---------------
---------------
#### Contents of Repository:
- Original NN Model (non-optimized)
  - 1 x .ipynb python notebook, (deep_learning_challenge.ipynb)
  - 1 x HDF5 export file (AlphabetSoupCharity.h5)
- Optimization Attempts
  - 11 x Folders titled "Optimization_Attempt_xx" (numbered 00 to 10)
    - Inside each folder:
      - 1 x .ipynb python notebook, (deep_learning_challenge_Optimization_xx.ipynb)
      - 1 x HDF5 export file (AlphabetSoupCharity_Optimization_xx.h5)
- 1 x README file

-------------------
#### Contributors:
N/A

------------------
#### License:
[MIT](https://choosealicense.com/licenses/mit/)





