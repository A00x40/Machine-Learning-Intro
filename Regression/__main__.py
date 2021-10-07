import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1234
NUM_SAMPLES = 50

# Set seed for reproducibility
np.random.seed(SEED)

# Generate synthetic data
def generate_data(num_samples):
    """
    Generate dummy data for linear regression
    Create roughly linear data (y = 3.5X + noise)
    """
    x = np.array( range(num_samples) )
    random_noise = np.random.uniform(-10, 20, size=num_samples)

    y = 3.5*x + random_noise # add some noise
    return x, y

# Generate random (linear) data
x, y = generate_data(NUM_SAMPLES)

data = np.vstack([x, y]).T
#print (data[:5])

df = pd.DataFrame(data, columns=["x", "y"])
#print(df)
x = df[["x"]].values
y = df[["y"]].values
df.head()

# Scatter plot
plt.title("Generated data")
plt.scatter(x=df["x"], y=df["y"])

# Random Data Ready
#plt.show()

#---------------------------------------------------------------------------------------------
'''
To determine the efficacy of our models, we need to have an unbiased measuring approach.
To do this, we split our dataset into :
    training, validation, and testing data splits.

With Less training data, your parameter estimates have greater variance.
With less testing data, your performance statistic will have greater variance. 
'''

train_size = 0.8
val_size   = 0.1
test_size  = 0.1

# Shuffle Data
'''
Sometimes data is ordered by some columns and when you split your data 
to ratio of 75% vs 25% you are blind for some values that exists in the last 25% split. 
so you learn everything except the values that exists in the test (last 25% rows). 
Thats why the best is to shuffle
'''

indices = list(range(NUM_SAMPLES))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Split indices
train_start = 0
train_end = int(train_size*NUM_SAMPLES)

val_start = train_end 
val_end = int((train_size + val_size)*NUM_SAMPLES)

test_start = val_end 

# Split Data
x_train = x[train_start:train_end]
y_train = y[train_start:train_end]

x_val   = x[val_start:val_end]
y_val   = y[val_start:val_end]

x_test  = x[test_start:]
y_test  = y[test_start:]

#---------------------------------------------------------------------------------------------
'''
Standardize data
We need to standardize our data (zero mean and unit variance) 
so a specific feature's magnitude doesn't affect how the model learns its weights.

We need to treat the validation and test sets as if they were hidden datasets. 
So we only use the train set to determine the mean and std to avoid biasing our training process.
'''
def standardize_data(data, mean, std):
    return (data - mean)/std
# Determine means and stds
x_mean = np.mean(x_train)
x_std  = np.std(x_train)
y_mean = np.mean(y_train)
y_std  = np.std(y_train)

x_train = standardize_data(x_train, x_mean , x_std)
y_train = standardize_data(y_train, y_mean , y_std)
x_val   = standardize_data(x_val, x_mean, x_std)
y_val   = standardize_data(y_val, y_mean, y_std)
x_test  = standardize_data(x_test, x_mean, x_std)
y_test  = standardize_data(y_test, y_mean, y_std)
#---------------------------------------------------------------------------------------------
'''
Weights
Our goal is to learn a linear model y_pred
that models Y given X using weights W and bias B

y_pred = B + W * X

'''
INPUT_DIM = x_train.shape[1] # x is 1-dimensional
OUTPUT_DIM = y_train.shape[1] # y is 1-dimensional

# Initialize random weights
w = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, 1))
print (f"W: {w.shape}",w)
print (f"b: {b.shape}",b)

LEARNING_RATE = 1e-1
NUM_EPOCHS = 100

# Training loop
for epoch_num in range(NUM_EPOCHS):

    # Forward pass [NX1] · [1X1] = [NX1]
    y_pred = np.dot(x_train, w) + b

    # Loss
    '''
    Compare the predictions with the actual target values  using the objective (cost) function 
    to determine the loss . 
    A common objective function for linear regression is mean squared error (MSE)
    '''
    N = len(y_train)
    loss = (1/N) * np.sum((y_train - y_pred)**2)

    # Show progress
    if epoch_num%10 == 0:
        print (f"Epoch: {epoch_num}, loss: {loss:.3f}")

    # Backpropagation
    db = -(2/N) * np.sum((y_train - y_pred) * 1)
    dw = -(2/N) * np.sum((y_train - y_pred) * x_train)

    # Update weights
    b += -LEARNING_RATE * db
    w += -LEARNING_RATE * dw

#---------------------------------------------------------------------------------------------
'''
Evaluation
Now we're ready to see how well our trained model will perform on our test data split. 
This will be our best measure on how well the model would perform on the real world,
given that our dataset's distribution is close to unseen data.
'''
# Predictions
pred_train = w*x_train + b
pred_test = w*x_test + b

# Train and test MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print (f"train_MSE: {train_mse:.2f}, test_MSE: {test_mse:.2f}")

# Figure size
plt.figure(figsize=(15,5))

# Plot train data
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(x_train, y_train, label="y_train")
plt.plot(x_train, pred_train, color="red", linewidth=1, linestyle="-", label="model")
plt.legend(loc="lower right")

# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(x_test, y_test, label='y_test')
plt.plot(x_test, pred_test, color="red", linewidth=1, linestyle="-", label="model")
plt.legend(loc="lower right")

# Show plots
plt.show()

# Unscaled weights
w_unscaled = w * (y_std/x_std)
b_unscaled = b * y_std + y_mean - np.sum(w_unscaled*x_mean)
print ("[actual] y = 3.5X + noise")
print (f"[model] y_hat = {w_unscaled[0][0]:.1f}X + {b_unscaled[0][0]:.1f}")