import pandas as pd
import matplotlib.pyplot as plt

N = [50, 75, 100, 125]

def read_and_filter_csv(df, N_value):   
    diag          = df[df['N'] == N_value]['diag']
    diag_error    = df[df['N'] == N_value]['diag_error']
    offdiag       = df[df['N'] == N_value]['offdiag']
    offdiag_error = df[df['N'] == N_value]['offdiag_error']
    
    return diag, diag_error, offdiag, offdiag_error

df = pd.read_csv('matrix_data.csv', skipinitialspace=True)
k = df[df['N'] == 100]['k']

diag = []
diag_error = []
offdiag = []
offdiag_error = []

for i in N:
    a,b,c,d = read_and_filter_csv(df, i)
    diag.append(a.to_numpy())
    diag_error.append(b)
    offdiag.append(c)
    offdiag_error.append(d)

fig, ax = plt.subplots(2,2)
for i in range(len(N)):
    first = int(i/2); second = i%2
    ax[first][second].scatter(k, diag[i], color='red', s=10)
    ax[first][second].errorbar(k, diag[i], diag_error[i], linestyle='None', color='red')
    ax[first][second].set_title('N={}'.format(N[i]), size=14)
    ax[first][second].set_ylabel(r"$\chi_\text{diagonal}$", size=18, color='red')
    ax[first][second].set_xlabel('t', size=18)
    
    sax = ax[first][second].twinx()
    sax.scatter(k, offdiag[i], color='blue', s=10)
    sax.errorbar(k, offdiag[i], offdiag_error[i], linestyle='None', color='blue')
    sax.set_ylabel(r"$\chi_\text{off-diagonal}$", size=18, color='blue')

plt.show()
