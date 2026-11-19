
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load saved model metrics
metrics = joblib.load("model_metrics.pkl")

# Get confusion matrix
cm = metrics["confusion_matrix"]

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Smart Spam Detection - Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("Confusion matrix saved successfully!")
