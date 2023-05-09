import sys
import pandas as pd

full_df = pd.read_csv(sys.argv[1])

#print(df)


for file_path in sys.argv[2:]:
    cur_row = pd.read_csv(file_path)
    full_df=pd.concat([full_df, cur_row], axis=0)
print("\n\n\\\\\\\\\\\\\\\\\\\n")
print(full_df)
print("\n")
print("---> mean:",full_df["test_accuracy_mean"].mean())
print("---> std:",full_df["test_accuracy_mean"].std())

print("\n\n\\\\\\\\\\\\\\\\\\\n")
#print(file_path)
#print('Number of arguments:', len(sys.argv), 'arguments.')
#print('Argument List:', str(sys.argv))