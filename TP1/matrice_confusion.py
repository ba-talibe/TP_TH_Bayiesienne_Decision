from validation import y_pred, y_valid
import pandas as pd

df = pd.DataFrame()
df["y_pred"] = y_pred
df["y_valid"] = y_valid

true_positive = df[(df.y_pred == 0) & (df.y_valid == 0)].size
faux_positive = df[(df.y_pred == 0) & (df.y_valid == 1)].size

true_nagative = df[(df.y_pred == 1) & (df.y_valid == 0)].size
faux_nagative = df[(df.y_pred == 1) & (df.y_valid == 1)].size

print(" {:<30}| {:<20}| {:<20}|".format("prediction\\resultats terrain", "positive classe 0", "negative classe 1"))
print(" {:<30}| {:<20}| {:<20}|".format("possitive classe 0", true_positive, faux_positive))
print(" {:<30}| {:<20}| {:<20}|".format("possitive classe 1", true_nagative, faux_nagative))