from math import *
import numpy as np
import matplotlib.pyplot as plot
import numpy.linalg as la


def sigfigs(x, n):
    # a function that gives n sig figs of x
    # pretty broken, do not use
    if n is None:
        return str(x)
    return '{:.{p}g}'.format(float('{:.{p}g}'.format(x, p=n)), p=n)

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
def makeTable(title, *datasets):
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

def analytic_regression(xdata, ydata, xerr, yerr: list):
    # does linear regression and also spits out a shit ton of latex to "show your work" even though you did no work
    # needs some work to get reasonable numbers in the latex, right now it just gives whatever floats are calculated

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



def standard_graph(title, xlabel, ylabel, xdata, ydata, xerr, yerr):
    # makes a nice graph, quickly

    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)

    plot.scatter(xdata, ydata, color='blue')
    plot.errorbar(xdata, ydata, yerr=yerr, color='red', capsize=5, fmt='none')


def regression_graph(xdata, ydata, xerr, yerr):
    # adds a linear regression to your nice graph, quickly
    # also spits out the regression latex

    m, b, dm, db = analytic_regression(xdata, ydata, xerr, yerr)
    plot_line_with_error_band(m, b, dm, db, xdata[0], xdata[-1], color='green', linestyle='--')
    return m, b, dm, db


# LAB 3 SPECIFIC DATA AND CODE

dV = 0.005
dA = 0.005

# R = 1 ohm
V1 = [1, 2, 3]
A1 = [1, 2, 3]

# R = 1 ohm, parallel
V2 = [1, 2, 3]
A2 = [2, 4, 6]
R1V2 = [1, 2, 3]
R2V2 = [1, 2, 3]
R1A2 = [1, 2, 3] # lower resistor
R2A2 = [1, 2, 3] # upper resistor

# R = 1 ohm, series
V3 = [1, 2, 3]
A3 = [0.5, 1, 1.5]
R1V3 = [0.5, 1, 1.5]
R2V3 = [0.5, 1, 1.5]
R1A3 = [0.5, 1, 1.5]
R2A3 = [0.5, 1, 1.5]

# R = 1 ohm, wire resistivity
inputV4 = [1,2,3]
V4 = [0.33, 0.68, 1.02]
A4 = [0.33, 0.67, 1.00]
# we measured as close to the resistor as possible to minimize its effect

LdV = 0.5
LdA = 0.0005
LV = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
LA = [569.863, 615.237, 660.134, 704.491, 748.212, 791.171, 833.210, 874.123, 913.624, 951.291]


v1ds = DataSet(V1, dV, 'Voltage (V)')
a1ds = DataSet(A1, dA, 'Current (A)')
r1ds = DataSet.transform(lambda V, I: V/I, lambda V, I, dV, dI: dV/I + V*dI/I**2, v1ds, a1ds, name=r'Resistance (\Omega)')

makeTable('Circuit Voltage and Current Data', DataSet(V1, dV, name='Input Voltage (V)'), v1ds, a1ds, r1ds)
makeTable('Circuit Voltage and Current Data', DataSet(V2, dV, name='Input Voltage (V)'), DataSet(V2, dV, 'Voltage (V)'), DataSet(A2, dA, 'Current (A)'))
makeTable('Circuit Voltage and Current Data', DataSet(V3, dV, name='Input Voltage (V)'), DataSet(V3, dV, 'Voltage (V)'), DataSet(A3, dA, 'Current (A)'))
makeTable('Circuit Voltage and Current Data', DataSet(V2, dV, name='Input Voltage (V)'), DataSet(R1V2, dV, 'R1 Voltage (V)'),
          DataSet(R2V2, dV, 'R2 Voltage (V)'), DataSet(R1A2, dA, 'R1 Current (A)'), DataSet(R2A2, dA, 'R2 Current (A)'))
makeTable('Circuit Voltage and Current Data', DataSet(V3, dV, name='Input Voltage (V)'), DataSet(R1V3, dV, 'R1 Voltage (V)'),
          DataSet(R2V3, dV, 'R2 Voltage (V)'), DataSet(R1A3, dA, 'R1 Current (A)'), DataSet(R2A3, dA, 'R2 Current (A)'))

makeTable('Circuit Voltage and Current Data', DataSet(inputV4, dV, name='Input Voltage (V)'), DataSet(V4, dV, 'Voltage (V)'), DataSet(A4, dA, 'Current (A)'))

#LVDS = DataSet(LV, LdV, 'Voltage (V)')
#LCDS = DataSet(LA, LdA, 'Current (mA)')

#makeTable('Lamp Voltage and Current Data', LVDS, LCDS)

#standard_graph('Lamp Voltage by Current', 'Voltage (V)', 'Current (mA)', LV, LA, LVDS.uncertainties, LCDS.uncertainties)
#m, b, dm, db = regression_graph(LV, LA, LVDS.uncertainties, LCDS.uncertainties)

#plot.show()
