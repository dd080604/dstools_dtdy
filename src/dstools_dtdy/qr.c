#include <math.h>
#include <stdlib.h>

void householder_qr(int m, int n, double *A, double *Q, double *R)
{
    int i, j, k;

    // Initialize R = A
    for (i = 0; i < m*n; i++)
        R[i] = A[i];

    // Initialize Q = Identity
    for (i = 0; i < m*m; i++)
        Q[i] = 0.0;

    for (i = 0; i < m; i++)
        Q[i*m + i] = 1.0;

    for (j = 0; j < n; j++) {
        int len = m - j;
        double normx = 0.0;
        for (i = j; i < m; i++)
            normx += R[i*n + j] * R[i*n + j];
      
        normx = sqrt(normx);
        if (normx == 0) continue;

        double rho = (R[j*n + j] >= 0) ? -1.0 : 1.0;
        double u1 = R[j*n + j] - rho * normx;
        double *u = malloc(len * sizeof(double));

        for (i = 0; i < len; i++)
            u[i] = R[(j+i)*n + j] / u1;

        u[0] = 1.0;
        double beta = -rho * u1 / normx;

        // Update R
        for (k = j; k < n; k++) {

            double s = 0.0;
            for (i = 0; i < len; i++)
                s += u[i] * R[(j+i)*n + k];

            s *= beta;

            for (i = 0; i < len; i++)
                R[(j+i)*n + k] -= s * u[i];
        }

        // Update Q
        for (k = 0; k < m; k++) {

            double s = 0.0;
            for (i = 0; i < len; i++)
                s += Q[k*m + (j+i)] * u[i];

            s *= beta;

            for (i = 0; i < len; i++)
                Q[k*m + (j+i)] -= s * u[i];
        }

        free(u);
    }
}
