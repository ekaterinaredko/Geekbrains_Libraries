#1
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]
plt.plot(x ,y)
plt.show()
plt.scatter(x, y)
plt.show()
#2
t = np.linspace(0, 10, 51)
print(t)
f = np.cos(t)
print(f)
plt.plot(t, f, color='green')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show()
#3
x = np.linspace(-3, 3, 51)
print(x)

y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax1.set_title('График y1')
ax2.set_title('График y2')
ax3.set_title('График y3')
ax4.set_title('График y4')
ax1.set_xlim([-5, 5])
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
#4
import pandas as pd
plt.style.use('fivethirtyeight')
creditcard = pd.read_csv('C:/Users/redkoer/Documents/Python Scripts/creditcard.csv')
class_list = creditcard['Class'].value_counts()
print(class_list)
class_list.plot(kind='barh')
plt.show()
class_list.plot(kind='barh', logx=True)
plt.show()
class0 = creditcard.loc[creditcard['Class'] == 0, ['V1']]
class1 = creditcard.loc[creditcard['Class'] == 1, ['V1']]
plt.hist(class0['V1'], bins=20, density=True, alpha=0.5, label='Class 0', color='grey')
plt.hist(class1['V1'], bins=20, density=True, alpha=0.5, label='Class 1', color='red')
plt.legend()
plt.show()