# DCASE 2016::Domestic Audio Tagging / Baseline System
# Copyright (C) 2016 Peter Foster (p.a.foster@qmul.ac.uk) / QMUL
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from sklearn import metrics

def compute_eer(y_score, y_true):
    """Compute the equal error rate (EER) from the plot of the false negative rate
        versus the false positive rate.

    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_score,drop_intermediate=True)
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]

    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]
    else:
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)
    return EER
