import pandas as pd
import os

print('--- Sample Submission Check ---')
sub = pd.read_csv(r'C:\BTech\Semester_5\ML\Assignment1\sample_submission.csv')
print(f'Columns: {list(sub.columns)}')
print(f'Shape: {sub.shape}')
print(f'Head:\n{sub.head(2)}')

base_dir = r'C:\BTech\Semester_5\ML\Assignment1\BT2024230'
os.makedirs(os.path.join(base_dir, 'src'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'submissions'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'report'), exist_ok=True)

print('\n--- EDA ---')
v1 = pd.read_csv(f'{base_dir}\\BT2024230_train_var1.csv')
v2 = pd.read_csv(f'{base_dir}\\BT2024230_train_var2.csv')

print('Var1 Target Stats:')
print(v1['y'].describe())
print('\nVar2 Target Stats:')
print(v2['y'].describe())
