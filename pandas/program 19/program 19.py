import pandas as pd
details = {'Name':['a','b','c','d','e'],'Occupation':['A1','A1','A1','B1','B1'],'Salary':[20,30,40,27,23]}
df=pd.DataFrame(details)
print(df)
occ_average_salary = df.groupby('Occupation')['Salary'].mean()
print("Average salary per occupation:")
print(occ_average_salary)