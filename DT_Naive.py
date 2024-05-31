import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class NaiveBayesClassifier:
    ##Learning Phase
    def train(self, train_data, train_labels):
        class_probs = {}
        feature_probs = {}
        total_samples = len(train_data)
        class_probs[0] = (train_labels == 0).sum() / total_samples
        class_probs[1] = (train_labels == 1).sum() / total_samples
        for col in train_data.columns:
            feature_probs[col] = {}
            if train_data[col].dtype == 'int64' or train_data[col].dtype == 'float64':
                # For continuous variables
                mean_0 = train_data[train_labels == 0][col].mean()
                std_0 = train_data[train_labels == 0][col].std()
                mean_1 = train_data[train_labels == 1][col].mean()
                std_1 = train_data[train_labels == 1][col].std()
                feature_probs[col] = {'mean': {0: mean_0, 1: mean_1}, 'std': {0: std_0, 1: std_1}}
            else:
                # For categorical variables
                for value in train_data[col].unique():
                    for cls in [0, 1]:
                        num = ((train_data[col] == value) & (train_labels == cls)).sum()
                        denom = (train_labels == cls).sum()
                        feature_probs[col][(value, cls)] = num / denom
        return class_probs, feature_probs
    ##Testing Phase
    def predict(self, test_data, class_probs, feature_probs):
        predictions = []
        for _, row in test_data.iterrows():
            prob_0 = class_probs[0]
            prob_1 = class_probs[1]
            for col in test_data.columns:
                if isinstance(row[col], (int, float)):
                    # For continuous variables, calculate probability using Gaussian distribution
                    prob_0 *= self.gaussian_prob(row[col], feature_probs[col]['mean'][0], feature_probs[col]['std'][0])
                    prob_1 *= self.gaussian_prob(row[col], feature_probs[col]['mean'][1], feature_probs[col]['std'][1])
                else:
                    # For categorical variables
                    prob_0 *= feature_probs[col].get((row[col], 0), 0.01)
                    prob_1 *= feature_probs[col].get((row[col], 1), 0.01)
            predictions.append(0 if prob_0 >= prob_1 else 1)
        return predictions
    
    def gaussian_prob(self, x, mean, std):
        # Calculate Gaussian probability density function
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:   ##All Labels are the Same
            return np.unique(y)[0]
        if len(X) == 0:             ##No Features Left to Split On
            return np.unique(y)[np.argmax(np.bincount(y))]
        ##Initialization
        best_gain = -1
        best_feature = None
        best_value = None
        ##Iterating Over Features
        for feature in range(X.shape[1]):
            ##Iterating Over Unique Values of Each Feature
            values = np.unique(X[:, feature])
            for value in values:
                mask = X[:, feature] <= value      ##Threshold
                y_left = y[mask]
                y_right = y[~mask]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature         ##In the first iteration it will assign the best root for the Tree
                    best_value = value

        if best_gain == 0:
            return np.unique(y)[np.argmax(np.bincount(y))]

        mask = X[:, best_feature] <= best_value
        X_left, X_right = X[mask], X[~mask]
        y_left, y_right = y[mask], y[~mask]

        left = self._build_tree(X_left, y_left)
        right = self._build_tree(X_right, y_right)

        return {'feature': best_feature, 'value': best_value, 'left': left, 'right': right}

    def _information_gain(self, parent, left_child, right_child):
        p = len(left_child) / len(parent)
        return self._entropy(parent) - (p * self._entropy(left_child) + (1 - p) * self._entropy(right_child))

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        p0 = len(y[y == 0]) / len(y)
        p1 = len(y[y == 1]) / len(y)
        if p0 == 0 or p1 == 0:
            return 0
        return -p0 * np.log2(p0) - p1 * np.log2(p1)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if isinstance(tree, np.int64):
            return tree
        feature, value = tree['feature'], tree['value']
        if x[feature] <= value:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

def read_data(file_path, sample_size):
    data = pd.read_csv(file_path)
    num_records = int(len(data) * (sample_size / 100))
    df_sampled = data.sample(n=num_records, random_state=42)
    return df_sampled

def preprocess_data(data):
    def recategorize_smoking(smoking_status):
        if smoking_status in ['never', 'No Info']:
            return 'non-smoker'
        elif smoking_status == 'current':
            return 'current'
        elif smoking_status in ['ever', 'former', 'not current']:
            return 'past_smoker'
        else:
            return 'unknown'

    data['smoking_history'] = data['smoking_history'].apply(recategorize_smoking)
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['smoking_history'] = data['smoking_history'].map({'non-smoker': 0, 'past-smoker': 1, 'current': 2, 'unknown': np.nan})
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    scaler = MinMaxScaler()
    numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

def split_data(data, train_percentage):
    train_size = int(len(data) * (train_percentage / 100))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

def evaluate(test_labels, predictions):
    accuracy = np.mean(predictions == test_labels)
    return accuracy

class DiabetesClassifierGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Diabetes Risk Prediction")
        self.master.geometry("800x400")
        self.file_path = None
        self.sample_size = tk.StringVar()
        self.train_percentage = tk.StringVar()
        self.nb_classifier = NaiveBayesClassifier()
        self.dt_classifier = DecisionTreeClassifier()

        self.file_frame = tk.Frame(self.master)
        self.file_frame.pack(pady=10)
        self.file_label = tk.Label(self.file_frame, text="Select Dataset:")
        self.file_label.grid(row=0, column=0)
        self.browse_button = tk.Button(self.file_frame, text="Browse", command=self.browse_file, width=10)
        self.browse_button.grid(row=0, column=1, padx=10)

        self.sample_size_frame = tk.Frame(self.master)
        self.sample_size_frame.pack(pady=5)
        self.sample_size_label = tk.Label(self.sample_size_frame, text="Percentage of Data to Use:")
        self.sample_size_label.grid(row=0, column=0)
        self.sample_size_entry = tk.Entry(self.sample_size_frame, textvariable=self.sample_size, width=10)
        self.sample_size_entry.grid(row=0, column=1)

        self.percentage_frame = tk.Frame(self.master)
        self.percentage_frame.pack(pady=5)
        self.percentage_label = tk.Label(self.percentage_frame, text="Percentage of Data to Use for Training:")
        self.percentage_label.grid(row=0, column=0)
        self.percentage_entry = tk.Entry(self.percentage_frame, textvariable=self.train_percentage, width=10)
        self.percentage_entry.grid(row=0, column=1)

        self.run_button = tk.Button(self.master, text="Run Classifier", command=self.run_classifier, width=20)
        self.run_button.pack(pady=10)

        self.output_frame = tk.Frame(self.master)
        self.output_frame.pack(expand=True, fill="both")
        self.output_scroll = ScrolledText(self.output_frame, wrap=tk.WORD, width=100, height=20)
        self.output_scroll.pack(expand=True, fill="both")

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            self.file_label.config(text=self.file_path)

    def run_classifier(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a dataset file.")
            return
        try:
            sample_size = float(self.sample_size.get())
            train_percentage = float(self.train_percentage.get())
            if not 0 < sample_size <= 100 or not 0 < train_percentage <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid percentage. Please enter values between 0 and 100.")
            return

        data = read_data(self.file_path, sample_size)
        data = preprocess_data(data)
        train_data, test_data = split_data(data, train_percentage)

        nb_train_labels = train_data['diabetes']
        nb_class_probs, nb_feature_probs = self.nb_classifier.train(train_data.drop('diabetes', axis=1), nb_train_labels)
        nb_predictions = self.nb_classifier.predict(test_data.drop('diabetes', axis=1), nb_class_probs, nb_feature_probs)
        nb_accuracy = evaluate(test_data['diabetes'], nb_predictions)

        dt_train_labels = train_data['diabetes']
        dt_train_features = train_data.drop('diabetes', axis=1).values
        self.dt_classifier.fit(dt_train_features, dt_train_labels)
        dt_predictions = self.dt_classifier.predict(test_data.drop('diabetes', axis=1).values)
        dt_accuracy = evaluate(test_data['diabetes'], dt_predictions)

        formatted_predictions = ""
        for i, (_, row) in enumerate(test_data.iterrows(), start=1):
            features_processed = "\n".join([f"{col}: {row[col]}" for col in data.columns])
            formatted_predictions += f"Record {i}:\n{features_processed}\n"
            formatted_predictions += f"Actual Label: {row['diabetes']}\n"
            formatted_predictions += f"Naive Bayes Predicted Label: {nb_predictions[i-1]}\n"
            formatted_predictions += f"Decision Tree Predicted Label: {dt_predictions[i-1]}\n\n"

        self.output_scroll.insert(tk.END, f"Naive Bayes Accuracy: {nb_accuracy:.2f}\n")
        self.output_scroll.insert(tk.END, f"Decision Tree Accuracy: {dt_accuracy:.2f}\n")
        self.output_scroll.insert(tk.END, f"Predictions for test data:\n{formatted_predictions}")

        best_model = "Naive Bayes" if nb_accuracy > dt_accuracy else "Decision Tree"
        self.output_scroll.insert(tk.END, f"Best Model: {best_model}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesClassifierGUI(root)
    root.mainloop()
