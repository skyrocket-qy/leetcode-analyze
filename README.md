# LeetCode Question Topic Predictor

This project analyzes LeetCode questions to predict their topics (e.g., "Array", "Hash Table", "Dynamic Programming") based on their content. It uses a Decision Tree classifier trained on the title and description of LeetCode "Algorithm" questions.

## Features

- **LeetCode Scraper**: Fetches all questions from LeetCode using their GraphQL API.
- **Data Preprocessing**: Cleans HTML from question content and prepares the data for model training.
- **Multi-Label Classification**: Handles questions that have multiple topics.
- **TF-IDF Vectorization**: Converts the text content of questions into numerical features.
- **Oversampling with KL-Divergence**: Balances the dataset by oversampling under-represented topics until the class distribution is close to uniform, measured by Kullback-Leibler divergence.
- **Decision Tree Model**: Trains a Decision Tree to predict topics from the question's content.
- **Result Logging**: Saves detailed reports, including accuracy, classification metrics, and class counts for each run.

## How it Works

The project follows these steps:

1.  **Data Collection**: The `leetcode/getAllQuestions.py` script sends a GraphQL query to LeetCode to fetch all available questions.
2.  **Filtering**: The `main.py` script filters the fetched questions to keep only those in the "Algorithms" category.
3.  **Preprocessing**:
    - The question's title and content are combined to form the input features (X).
    - The associated topics are the target labels (y).
    - `TfidfVectorizer` is used to convert the text features into a sparse matrix of TF-IDF features.
    - `MultiLabelBinarizer` transforms the list of topics for each question into a binary vector format suitable for multi-label classification.
4.  **Oversampling**:
    - The script calculates the distribution of topics in the training set.
    - It iteratively oversamples the rarest topics until the KL-divergence between the current topic distribution and a uniform distribution falls below a set threshold (`KL_BOUNDARY`). This helps the model learn from less frequent topics.
5.  **Training**:
    - A Decision Tree classifier (`DecisionTreeClassifier`) is trained on the preprocessed and oversampled data.
6.  **Evaluation**:
    - The trained model is evaluated on a held-out test set.
    - The accuracy score and a detailed classification report are generated and saved.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/leetcode-analyze.git
    cd leetcode-analyze
    ```

2.  **Install dependencies:**
    This project requires Python 3 and the following libraries:
    - `scikit-learn`
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `requests`

    It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

    Install the required packages:
    ```bash
    pip install scikit-learn numpy scipy matplotlib requests
    ```

3.  **Run the project:**
    Execute the main script to start the process:
    ```bash
    python main.py
    ```
    The script will create a new directory in `records/` containing the results of the run.

## Results

After each run, a new directory is created under `records/` with a timestamped name (e.g., `records/2024-10-19:22-16-53/`). This directory contains the following files:

-   `report.json`: A summary of the run, including accuracy, number of questions processed, and total execution time.
-   `classification_report.json`: A detailed report with precision, recall, and F1-score for each topic.
-   `class_counts.json`: The count of questions for each topic in the dataset before oversampling.
-   `decision_tree.png` (optional): If uncommented in `decisionTree/train.py`, this file contains a visualization of the trained decision tree.

## Future Work

The `CNN/` directory contains a placeholder for a Convolutional Neural Network model, which could be implemented as an alternative to the Decision Tree for topic prediction.
