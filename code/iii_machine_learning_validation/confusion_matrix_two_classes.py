import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("https://tinyurl.com/y2cocoo7")

# grab independent variable column
inputs = data.iloc[:, :-1]

# grab dependent variable column
output = data.iloc[:, -1]

# build logistic regression, note CVLogisticRegression is also recommended to use cross-validation
fit = LogisticRegression().fit(inputs, output)

# Plot confusion matrix with and without normalization
titles_options = [("Confusion Matrix", None),
                  ("Normalized Confusion Matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(fit, inputs, output,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
