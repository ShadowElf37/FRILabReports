from math import *
import numpy as np
import matplotlib.pyplot as plot
import numpy.linalg as la
from scipy.optimize import curve_fit
import sympy as sp


def latexify_float(x):
    # turns "e+N" into "\cdot10^N"
    num = str(x).split('e')
    if len(num) == 1:
        return num[0]
    else:
        return num[0] + '\cdot 10^{' + str(int(num[1])).strip('+') + '}'

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
        self.uncertainties = [None]*self.count if uncertainty is None else [uncertainty]*self.count if type(uncertainty) in (int, float) else uncertainty
        self.name = name

        self.n_sigfigs = n_sigfigs
        self.dn_sigfigs = dn_sigfigs

    def latex(self):
        return (rf'${x} \pm {dx}$' for x, dx in zip(self.data, self.uncertainties))
    def latex_sigfigs(self, n: int=None, dn: int=None):
        return ((rf'${sigfigs(x, n)} \pm {sigfigs(dx, dn)}$' if dx is not None else rf'${sigfigs(x, n)}$') for x, dx in zip(self.data, self.uncertainties))
    def latex_auto(self):
        return self.latex_sigfigs(self.n_sigfigs, self.dn_sigfigs)

    def slice(self, i0, i1=None):
        return DataSet(self.data[i0:i1], self.uncertainties[i0:i1], name=self.name, n_sigfigs=self.n_sigfigs, dn_sigfigs=self.dn_sigfigs)

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
    partitions = int(ceil(abs(resolution*(x_upper_bound - x_lower_bound))))
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


# LAB SPECIFIC DATA AND CODE

Vin = DataSet([5]*3, None, 'Input (V)')
R1 = DataSet([1, 1, 3], None, r'R1 (k$\Omega$)')
R2 = DataSet([2, 1, 3], None, r'R2 (k$\Omega$)')
R3 = DataSet([3, 1, 1], None, r'R3 (k$\Omega$)')
Vout = DataSet([-15, -10, -3.333], 0.05, r'Output (V)')
P = DataSet([-15, -10, -2.5], None, 'Predicted Output (V)')

make_table('Complicated Amplifier Output', Vin, R1, R2, R3, Vout, P)

"""
dv = 0.0005
dt = 0.0005

# raw RC stuff
t0 = 77.635
v = [0., 0.531, 1.541, 3.261, 4.068, 4.539, 4.848, 4.197, 3.048, 2.436, 1.027]  # V
t = np.array([0., 2.115, 7.235, 20.995, 33.475, 47.555, 77.635, 80.195, 86.595, 91.075, 108.355]) - t0 # ms
v0 = 4.848
R = 2000
C = 10**-5

vds = DataSet(np.array(v), dv, 'Voltage (V)', n_sigfigs=4)
tds = DataSet(t, dt, 'Time (ms)', n_sigfigs=6)

#lnvds = DataSet(np.log(np.array(v)/4.848), dv, 'Adjusted Voltage (V)')

# low pass stuff
tau = 51000*3.3*10**-9
w_cutoff = 1/tau
f_cutoff = w_cutoff/2/pi  # 945.6621692923076 Hz

maxv = [0.466, 0.519, 0.584, 0.667, 0.776, 0.927, 1.148, 1.502, 2.179, 3.435, 3.535, 3.817, 4.420, 4.892]  # V
f = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.945662, 0.8, 0.5, 0.2]  # kHz

fds = DataSet(list(reversed(f)), name='Frequency (kHz)')
pds = DataSet.transform(lambda x: 1/x, lambda x, dx: None, fds, name='Period (ms)', n_sigfigs=4, dn_sigfigs=None)
mvds = DataSet(list(reversed(maxv)), dv, 'Max Voltage (V)')


# High pass
# As the frequency goes beyond 2k, there is not a large difference in voltage between frequencies.
# After 3k, the difference is tiny (around 100mV).
# The cutoff seems to be around 1k, and it is letting the higher frequencies through,
# but substantially cutting off anything lower.
# This is as predicted, and matches the low pass experiment

make_table('Voltage Across Capacitor', fds, pds, mvds)

standard_graph('Voltage Across Capacitor', fds, mvds)

#plot_f(lambda T: v0*np.exp(-(T)/1000/(R*C)), 0, t[-1], linestyle='--', color='green')
#plot_f(lambda T: v0*(1-np.exp(-(T+t0)/1000/(R*C))), t[0], 0, linestyle='--', color='green')

#standard_graph('Voltage Across Capacitor', tds, vds)

#F = lambda V: np.log(V/v0)
#dF = lambda V, dV: (1/V)*dV+ 0.00000206

#discharge_tds = tds.slice(6)
#discharge_vds = DataSet.transform(F, dF, vds.slice(6), name='Adjusted Voltage', n_sigfigs=6, dn_sigfigs=2)

#make_table('Adjusted Voltage during Discharge', discharge_tds, discharge_vds)

#standard_graph('Adjusted Voltage Across Capacitor', discharge_tds, discharge_vds)

#m, b, dm, db = analytic_regression(discharge_tds, discharge_vds)

#plot_line_with_error_band(m, b, dm, db, tds.data[6], tds.data[-1])

#standard_graph('Voltage Across Capacitor', tds, lnvds)

#plot.legend()

plot.show()"""