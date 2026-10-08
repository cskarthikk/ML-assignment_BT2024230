import pandas as pd

try:
    df1 = pd.read_csv(r'C:\BTech\Semester_5\ML\var1_cv_results_refined.csv')
    df1 = df1[df1['prep'] != 'A_OLS']
    best1 = df1.loc[df1.groupby('degree')['cv_mse'].idxmin()]
    terms1 = {1:6, 2:27, 3:83, 4:209, 5:461, 6:923, 7:1715, 8:3002, 9:5004, 10:8007}
    t1 = '\\begin{table}[H]\n\\centering\n\\begin{tabular}{rrrrr}\n\\toprule\nDegree & Terms & Best $\\alpha$ & CV MSE & CV $R^2$ \\\\\n\\midrule\n'
    for _, r in best1.iterrows():
        d = int(r['degree'])
        t1 += f"{d} & {terms1.get(d,0)} & {r['alpha']:.4g} & {r['cv_mse']:.4f} & {r['cv_r2']:.4f} \\\\\n"
    t1 += '\\bottomrule\n\\end{tabular}\n\\caption{\\texttt{var1}: Best ridge $\\alpha$ for all ten permitted degrees.}\n\\end{table}\n'
except Exception as e:
    t1 = f'Error var1: {e}'

try:
    df2 = pd.read_csv(r'C:\BTech\Semester_5\ML\var2_cv_results_refined.csv')
    df2 = df2[df2['prep'] != 'A_OLS']
    best2 = df2.loc[df2.groupby('degree')['cv_mse'].idxmin()]
    terms2 = {1:3, 2:9, 3:19, 4:34, 5:55, 6:83, 7:119, 8:164, 9:219, 10:285, 11:363, 12:454, 13:559, 14:679, 15:815, 16:968, 17:1139, 18:1329, 19:1539, 20:1770}
    t2 = '\\begin{table}[H]\n\\centering\n\\begin{tabular}{rrrrr|rrrrr}\n\\toprule\nDeg & Terms & $\\alpha$ & CV MSE & CV $R^2$ & Deg & Terms & $\\alpha$ & CV MSE & CV $R^2$ \\\\\n\\midrule\n'
    for i in range(1, 11):
        r1 = best2[best2['degree'] == i].iloc[0]
        r2 = best2[best2['degree'] == i+10].iloc[0]
        t2 += f"{i} & {terms2.get(i,0)} & {r1['alpha']:.4g} & {r1['cv_mse']:.4f} & {r1['cv_r2']:.4f} & {i+10} & {terms2.get(i+10,0)} & {r2['alpha']:.4g} & {r2['cv_mse']:.4f} & {r2['cv_r2']:.4f} \\\\\n"
    t2 += '\\bottomrule\n\\end{tabular}\n\\caption{\\texttt{var2}: Best ridge $\\alpha$ for all twenty permitted degrees.}\n\\end{table}\n'
except Exception as e:
    t2 = f'Error var2: {e}'

with open('latex_tables.txt', 'w') as f:
    f.write(t1 + '\n\n' + t2)
