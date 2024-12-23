import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def evaluate_knn_from_file(test_size=0.2, num_trials=7, k_range=(1, 50)):
    """
    Load a dataset from a file, evaluate kNN performance over multiple trials, and save results to a CSV file.

    Parameters:
    - test_size (float): Proportion of the dataset to include in the test split
    - num_trials (int): Number of trials to perform
    - k_range (tuple): Range of k values to evaluate (inclusive)
    """
    # Ask the user for the file path
    file_path = input("Enter the path to the dataset file (CSV format): ")
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Assume the last column is the target variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Display dataset information
        print(f"Dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")

        # Initialize the results DataFrame
        k_values = range(k_range[0], k_range[1] + 1)
        results = pd.DataFrame(index=k_values, columns=[f"Trial {i + 1}" for i in range(num_trials)])

        # Perform trials
        for trial in range(num_trials):
            # Split data into train/test with a different random state for each trial
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=trial)

            # Evaluate each k value
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.loc[k, f"Trial {trial + 1}"] = accuracy

        # Save results to a CSV file
        output_file = "custom_kNN_results.csv"
        results.to_csv(output_file)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the function
evaluate_knn_from_file()
