import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

w = np.linspace(-0.5, 2.5, 100)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
J1 = (w - 1) ** 2
J2 = (w - 1) ** 2 + w ** 2
plt.plot(w, J1, label=r'$J_1=(w-1)^2$')
plt.scatter(1, 0, s=20)
plt.plot(w, J2, linestyle='--', label=r'$J_2=(w-1)^2+w^2$')
plt.xlabel('w')
plt.scatter(0.5, 0.5, s=20)

plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
w = np.linspace(-0.5, 2.5, 100)
J1 = 2 * (w - 1) ** 4
J2 = 2 * (w - 1) ** 4 + w ** 2
plt.plot(w, J1, label=r'$J_1=2(w-1)^4$')
plt.scatter(1, 0, s=20)
plt.plot(w, J2, linestyle='--', label=r'$J_2=2(w-1)^4+w^2$')
plt.scatter(0.5, 3 / 8, s=20)
plt.xlabel('w')
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()
