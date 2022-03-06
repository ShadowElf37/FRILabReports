from math import *
import numpy as np
import matplotlib.pyplot as plot
import numpy.linalg as la
from scipy.optimize import curve_fit


def latexify_float(x):
    # turns "e+N" into "\cdot10^N"
    num = str(x).split('e+')
    if len(num) == 1:
        return num[0]
    else:
        return num[0] + '\cdot 10^' + str(int(num[1]))

def sigfigs(x, n):
    # gives a string with n sig figs of x
    # still has trouble adding extra 0s
    if n is None:
        return str(x)
    return latexify_float('{:.{p}g}'.format(float('{:.{p}g}'.format(x, p=n)), p=n))

class DataSet:
    # a nice class that holds data and its uncertainties and makes it easy to do operations on it and get latex of its values

    def __init__(self, data, uncertainty=None, name='Data', n_sigfigs=None, dn_sigfigs=None):
        self.data = data
        self.count = len(data)
        self.uncertainties = [0]*self.count if uncertainty is None else [uncertainty]*self.count if type(uncertainty) in (int, float) else uncertainty
        self.name = name

        self.n_sigfigs = n_sigfigs
        self.dn_sigfigs = dn_sigfigs

    def latex(self):
        return (rf'${x} \pm {dx}$' for x, dx in zip(self.data, self.uncertainties))
    def latex_sigfigs(self, n: int=None, dn: int=None):
        return (rf'${sigfigs(x, n)} \pm {sigfigs(dx, dn)}$' for x, dx in zip(self.data, self.uncertainties))
    def latex_auto(self):
        return self.latex_sigfigs(self.n_sigfigs, self.dn_sigfigs)

    @staticmethod
    def transform(f, df, *datasets, name='', n_sigfigs=None, dn_sigfigs=None):
        """For datasets 1, 2, 3...
        Accepts f of the format x1, x2, x3... and df of the format x1, x2, x3... dx1, dx2, dx3..."""
        data = [ds.data for ds in datasets]
        uncertainties = [ds.uncertainties for ds in datasets]
        return DataSet([f(*x) for x in zip(*data)], [df(*x, *dx) for x, dx in zip(zip(*data), zip(*uncertainties))], name=name, n_sigfigs=n_sigfigs, dn_sigfigs=dn_sigfigs)

table_counter = 0
def make_table(title, *datasets):
    # makes a latex table from some datasets! super epic function right here

    global table_counter
    table_counter += 1
    ncols = len(datasets)

    if ncols == 0:
        raise ValueError('Can\'t make table with no data!')

    rows = [list() for _ in range(datasets[0].count)]
    latex_generators = [ds.latex_auto() for ds in datasets]

    for r in range(datasets[0].count):
        for ds in latex_generators:
            rows[r].append(next(ds))
    data_rows = ['\t\t\t'+' & '.join(row) for row in rows]

    table = r"""
\begin{table}[H]
	\caption{%TITLE%}
	\label{tab:table%TABLECOUNTER%}
    \begin{center}
        \begin{tabular}{%COLUMNFORMAT%}
            %DATAHEADERS%
%DATA%
        \end{tabular}
    \end{center}
\end{table}"""\
        .replace('%COLUMNFORMAT%', '|'.join(['c']*ncols))\
        .replace('%TABLECOUNTER%', str(table_counter))\
        .replace('%TITLE%', title)\
        .replace('%DATA%', (r'\\'+'\n').join(data_rows))\
        .replace('%DATAHEADERS%', ' & '.join(d.name for d in datasets)+'\\\\\n\t\t\t\\hline')

    print(table)
    return table


def plot_line(m, b, x_lower_bound, x_upper_bound, *args, **kwargs):
    # plots a line quickly
    plot.plot(np.linspace(x_lower_bound, x_upper_bound, 5), np.linspace(x_lower_bound, x_upper_bound, 5)*m+b, *args, **kwargs)

def plot_line_with_error_band(m, b, dm, db, x_lower_bound, x_upper_bound, *args, **kwargs):
    plot_line(m, b, x_lower_bound, x_upper_bound, *args, **kwargs)
    plot.fill_between(np.linspace(x_lower_bound, x_upper_bound, 5), np.linspace(x_lower_bound, x_upper_bound, 5) * (m - dm) + b - db,
                      np.linspace(x_lower_bound, x_upper_bound, 5) * (m + dm) + b + db, alpha=0.2, color='red')

def plot_f(f, x_lower_bound, x_upper_bound, *args, resolution=20, **kwargs):
    # plots a function quickly
    partitions = int(ceil(resolution*(x_upper_bound - x_lower_bound)))
    plot.plot(np.linspace(x_lower_bound, x_upper_bound, partitions), list(map(f, np.linspace(x_lower_bound, x_upper_bound, partitions))), *args,
              **kwargs)

def analytic_regression(xds, yds: DataSet):
    # does linear regression and also spits out a shit ton of latex to "show your work" even though you did no work
    # needs some work to get reasonable numbers in the latex, right now it just gives whatever floats are calculated

    xdata = xds.data
    xerr = xds.uncertainties
    ydata = yds.data
    yerr = yds.uncertainties

    invsqs = list(map(lambda x: x ** -2, yerr))
    x2invsqs = list(map(lambda x, dy: x ** 2 * dy, xdata, invsqs))
    xinvsqs = list(map(lambda x, dy: x * dy, xdata, invsqs))
    yinvsqs = list(map(lambda y, dy: y * dy, ydata, invsqs))
    #x2y2invsqs = list(map(lambda x, y, du: x ** 2 * y ** 2 * du, xdata, ydata, invsqs))
    xyinvsqs = list(map(lambda x, y, du: x * y * du, xdata, ydata, invsqs))

    a = sum(invsqs)
    b = sum(x2invsqs)
    c = sum(xinvsqs)
    d = sum(yinvsqs)
    #e = sum(x2y2invsqs)
    f = sum(xyinvsqs)

    regression_matrix = np.matrix([[b, c], [c, a]])
    regression_vector = np.matrix([[f], [d]])
    M = la.inv(regression_matrix) * regression_vector

    m, b, dm, db = M[0, 0], M[1, 0], sqrt(a / (a * b - c ** 2)), sqrt(b / (a * b - c ** 2))

    print('REGRESSION RESULTS')
    latex = r"""
    These values were calculated with the following formulae:

    \begin{gather}
        m\sumin \frac{x_i^2}{\Delta y_i ^ 2} + b\sumin \frac{x_i}{\Delta y_i ^ 2} = \sumin \frac{x_iy_i}{\Delta y_i ^ 2}\\
        m\sumin \frac{x_i}{\Delta y_i ^ 2} + b\sumin \frac{1}{\Delta y_i ^ 2} = \sumin \frac{y_i}{\Delta y_i ^ 2}\\
        \Delta m = \sqrt{\frac{\sumin \frac{1}{\Delta y_i ^ 2}}{\left(\sumin \frac{1}{\Delta y_i ^ 2}\right)\left(\sumin \frac{x_i^2}{\Delta y_i ^ 2}\right) - \left(\sumin \frac{x_i}{\Delta y_i ^ 2}\right)^2}}\\
        \Delta b = \sqrt{\frac{\sumin \frac{x_i^2}{\Delta y_i ^ 2}}{\left(\sumin \frac{1}{\Delta y_i ^ 2}\right)\left(\sumin \frac{x_i^2}{\Delta y_i ^ 2}\right) - \left(\sumin \frac{x_i}{\Delta y_i ^ 2}\right)^2}}
    \end{gather}
    
    Some intermediate quantities we get are (this is illustrative, please do not interpret these numbers as having correct precision):
    \begin{gather}
        \sumin \frac{1}{\Delta y_i ^ 2} = %A%\\
        \sumin \frac{x_i^2}{\Delta y_i ^ 2} = %B%\\
        \sumin \frac{x_i}{\Delta y_i ^ 2} = %C%\\
        \sumin \frac{y_i}{\Delta y_i ^ 2} = %D%\\
        \sumin \frac{x_iy_i}{\Delta y_i ^ 2} = %F%
    \end{gather}
    
    Using the system of equations defining $m$ and $b$, we can construct a matrix such that
    \begin{gather}
        \begin{bmatrix}
            %B% & %C%\\
            %C% & %A%
        \end{bmatrix} \begin{bmatrix}m\\b\end{bmatrix} = \begin{bmatrix}%F%\\%D%\end{bmatrix}\\
        \Rightarrow \begin{bmatrix}m\\b\end{bmatrix} = \begin{bmatrix}
            %B% & %C%\\
            %C% & %A%
        \end{bmatrix}^{-1}\begin{bmatrix}%F%\\%D%\end{bmatrix} \approx \begin{bmatrix}%M%\\%B%\end{bmatrix}\\
        \Delta m = \sqrt{\frac{%A%}{(%A%)(%B%)-(%C%)^2}} \approx %DM%\\
        \Delta b = \sqrt{\frac{%B%}{(%A%)(%B%)-(%C%)^2}} \approx %DB%
    \end{gather}""".replace('%A%', str(a)).replace('%B%', str(b)).replace('%C%', str(c)).replace('%D%', str(d)).replace('%F%', str(f)).replace('%M%', str(m)).replace('%B%', str(b)).replace('%DM%', str(dm)).replace('%DB%', str(db))

    print(latex)

    print('m =', m, '\nb =', b, '\ndm =', dm, '\ndb =', db)

    #plot.fill_between(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1)*(m-dm)+b-db, np.arange(0, 1, 0.1)*(m+dm)+b+db, alpha=0.2, color='red')

    return m, b, dm, db


def numerical_regression(f_to_fit_to, xds: DataSet, yds: DataSet, *guesses):
    # f_to_fit_to should be of the form f(t, *fit_params), and there should be as many guesses as fit_params
    return curve_fit(f_to_fit_to, xdata=xds.data, ydata=yds.data, p0=guesses)[0]


def standard_graph(title, xds: DataSet, yds: DataSet):
    # makes a nice graph, quickly

    xdata = xds.data
    xerr = xds.uncertainties
    ydata = yds.data
    yerr = yds.uncertainties

    plot.title(title)
    plot.xlabel(xds.name)
    plot.ylabel(yds.name)

    plot.scatter(xdata, ydata, color='blue')
    plot.errorbar(xdata, ydata, yerr=yerr, color='red', capsize=5, fmt='none')


def regression_graph(xds, yds):
    # adds a linear regression to your nice graph, quickly
    # also spits out the regression latex

    xdata = xds.data

    m, b, dm, db = analytic_regression(xds, yds)
    plot_line_with_error_band(m, b, dm, db, xdata[0], xdata[-1], color='green', linestyle='--')
    return m, b, dm, db




# LAB 4 SPECIFIC DATA AND CODE
m = 0.2
dm = 0.0005

dx = 0.005
dt = 0.005

x0 = 0.17  # eq

xdata = (np.array([0.50, -0.10, 0.38, 0.01, 0.3, 0.07, 0.25, 0.105, 0.22, 0.13, 0.2, 0.145, 0.19, 0.15, 0.18, 0.16, 0.175, 0.163, 0.172, 0.165])-x0)*-1
tdata = [0.00, 0.44, 0.82, 1.25, 1.62, 2.07, 2.44, 2.88, 3.26, 3.70, 4.06, 4.48, 4.89, 5.31, 5.71, 6.14, 6.55, 6.95, 7.38, 7.74]


damped_f = lambda t, A, g, w, phi: A * np.exp(-g/2 * t) * np.cos(w * t + phi)

x = DataSet(xdata, dx, 'Displacement (m)', n_sigfigs=3, dn_sigfigs=1)
t = DataSet(tdata, dt, 'Time (s)', n_sigfigs=3, dn_sigfigs=1)

standard_graph('Spring Displacement from Equilibrium', t, x)
plot_f(lambda t: damped_f(t, 0.33, 0.91, 7.57, pi), 0, tdata[-1], resolution=50)


make_table('Spring Displacement Data', t, x)


fA, fg, fw = numerical_regression(lambda t, A, g, w: damped_f(t, A, g, w, pi), t, x, 0.33, 0.91, 7.14)

print(fA, fg, fw)

plot_f(lambda t: damped_f(t, fA, fg, fw, pi), 0, tdata[-1], resolution=50, linestyle='--', color='green')

plot.show()
