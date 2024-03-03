import pandas as pd
import numpy as np

# Define the dataset
data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Function to calculate entropy
def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Function to calculate information gain
def calculate_info_gain(data, split_attribute_name, target_name):
    # Calculate the total entropy
    total_entropy = calculate_entropy(data[target_name])
    
    # Calculate the values and corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    # Calculate the weighted entropy of the subsets
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * calculate_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Calculate entropy of the target variable (Play Tennis)
target_entropy = calculate_entropy(df['Play Tennis'])
print("Entropy of the target variable (Play Tennis):", target_entropy)

# Calculate entropy and information gain for each attribute
for column in df.columns[1:-1]:  # Exclude the 'Day' and 'Play Tennis' columns
    entropy = calculate_entropy(df[column])
    info_gain = calculate_info_gain(df, column, 'Play Tennis')
    print("\nAttribute:", column)
    print("Entropy:", entropy)
    print("Information Gain:", info_gain)
