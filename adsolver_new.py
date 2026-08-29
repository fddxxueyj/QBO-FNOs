import torch

from . import utils
from . import stochastic_forcing


class ADSolver_new:
    """Class to solve a one-dimensional advection-diffusion equation with a
    source term.

    Parameters
    ----------
    z_min : float, optional
        Bottom boundary [m], by default 17e3
    z_max : float, optional
        Top boundary [m], by default 35e3
    dz : int, optional
        Vertical grid spacing [m], by default 250
    t_min : float, optional
        Initial time [s], by default 0
    t_max : float, optional
        Final time [s], by default 360*12*86400
    dt : float, optional
        Timestep [s], by default 86400
    w : float/tensor, optional
        Vertical advection [:math:`\mathrm{m \, s^{-1}}`], by default 0.
        Can be constant (scalar), time-dependent (tensor with size time.shape),
        or time- and height-dependent (tensor of size time.shape, z.shape).
    kappa : float, optional
        Constant diffusivity [:math:`\mathrm{m^{2} \, s^{-1}}`], by default 3e-1
    initial_condition : tensor, optional
        The unknown profile at t_min, by default None

    Attributes
    ----------
    z : tensor
        Vertical grid [m]
    time : tensor
        Temporal grid [s]
    D1 : tensor
        Differentiation matrix for centered (second-order accurate) first derivative
    D2 : tensor
        Differentiation matrix for centered (second-order accurate) second derivative
    """

    def __init__(self, z_min=17e3, z_max=35e3, dz=250, t_min=0,
    t_max=360*12*86400, dt=86400, w=0, kappa=3e-1, initial_condition=None):

        self.z_min = z_min
        self.z_max = z_max
        self.dz = dz
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt
        self.kappa = kappa
        if torch.is_tensor(w):
            self.w = w
        else:
            self.w = torch.tensor(w)

        self.z = torch.arange(z_min, z_max + self.dz, self.dz)
        self.nlev, = self.z.shape

        self.time = torch.arange(t_min, t_max + self.dt, self.dt)
        self.current_time = torch.tensor([t_min])

        self.initial_condition = initial_condition
        if self.initial_condition is None:
            self.initial_condition = (lambda z:
            -14/81e6 * (z - z_min) * (z - z_max))

        # create differentiation matrices
        self.D1 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev -1):
            self.D1[i, i + 1] = 1
            self.D1[i, i - 1] = -1
        self.D1 /= 2 * self.dz

        # create second derivative matrix
        self.D2 = torch.zeros((self.nlev, self.nlev))
        for i in range(1, self.nlev - 1):
            self.D2[i, [i - 1, i + 1]] = 1
            self.D2[i, i] = -2
        # for zero flux (Neumann) BC at the top uncomment following two lines
        # self.D2[self.nlev - 1, self.nlev - 2] = 2
        # self.D2[self.nlev - 1, self.nlev - 1] = -2
        self.D2 /= self.dz ** 2


    def solve(self, nsteps=None, source_func=None):
        """Integrates the model for a given number of steps starting from the
        initial conditions.

        Parameters
        ----------
        nsteps : int, optional
            number of time steps to take in the integration, by default None
        source_func : function, optional
            The source term as an explicit function of the unknown only, by default None

        Returns
        -------
        tensor
            Tensor of shape (n_steps, m), where m is the number of levels, whose
            entries are the zonal wind profiles at each time.
        """

        if nsteps is None:
            nsteps, = self.time.shape

        if source_func is None:
            # source_func, _, _ = utils.make_source_func(self)
            source_func = utils.load_model(self)

        # t = 0*dt
        #---------
        u = torch.zeros((nsteps, self.nlev))
        s = torch.zeros((nsteps, self.nlev))

        u[0] = self.initial_condition(self.z)

        if self.w.ndim == 0:
            # if w is constant LHS can be inverted only once

            # t = 1*dt
            #---------
            D = self.dt * (self.w * self.D1 - self.kappa * self.D2)

            # LHS
            B = torch.eye(self.nlev) + D

            Q, self.R = torch.linalg.qr(B)
            self.QT = Q.T

            # a single forward Euler step
            source = source_func(u[0])
            s[0] = source

            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(u[n])
                s[n] = source

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - D, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

                u[n + 1,0] = 0
                u[n + 1,-1] = 0

        elif self.w.ndim == 1:
            # w is time-dependent (constant with height)

            # t = 1*dt
            #---------
            D = self.dt * (self.w[0] * self.D1 - self.kappa * self.D2)

            # a single forward Euler step
            source = source_func(u[0])
            s[0] = source

            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(u[n])
                s[n] = source

                DL = self.dt * (self.w[n+1] * self.D1 - self.kappa * self.D2)
                DR = self.dt * (self.w[n-1] * self.D1 - self.kappa * self.D2)

                # LHS
                B = torch.eye(self.nlev) + DL

                Q, self.R = torch.linalg.qr(B)
                self.QT = Q.T

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - DR, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

                u[n + 1,0] = 0
                u[n + 1,-1] = 0

        elif self.w.ndim == 2:
            # w is time- and height-dependent (w[time, height])

            # t = 1*dt
            #---------
            D = self.dt * (torch.matmul(torch.diag(self.w[0]), self.D1) -
            self.kappa * self.D2)

            # a single forward Euler step
            source = source_func(u[0])
            s[0] = source

            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(u[n])
                s[n] = source

                DL = self.dt * (torch.matmul(torch.diag(self.w[n+1]), self.D1) -
                self.kappa * self.D2)
                DR = self.dt * (torch.matmul(torch.diag(self.w[n-1]), self.D1) -
                self.kappa * self.D2)

                # LHS
                B = torch.eye(self.nlev) + DL

                Q, self.R = torch.linalg.qr(B)
                self.QT = Q.T

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - DR, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

                u[n + 1,0] = 0
                u[n + 1,-1] = 0

        return u, s


    def emulate(self, nsteps=None, source_func=None,
    sfe=3.7e-3, sfv=9e-8, cwe=32, cwv=256, corr=0.75, seed=int(21*7+7+1),
    sf=None, cw=None):
        """The difference compared to solve is that for the emulation
        we took source term to be a function of (u, sf, cw), so the source
        requires an augmented input.
        """

        if nsteps is None:
            nsteps, = self.time.shape

        if source_func is None:
            # source_func, _, _ = utils.make_source_func(self)
            source_func = utils.load_model(self)

        if sf is None and cw is None:
            sf, cw = stochastic_forcing.sample_sf_cw(n=nsteps,
            sfe=sfe, sfv=sfv, cwe=cwe, cwv=cwv, corr=corr, seed=seed)

        # t = 0*dt
        #---------
        u = torch.zeros((nsteps, self.nlev))
        u[0] = self.initial_condition(self.z)

        if self.w.ndim == 0:
            # if w is constant LHS can be inverted only once

            # t = 1*dt
            #---------
            D = self.dt * (self.w * self.D1 - self.kappa * self.D2)

            # LHS
            B = torch.eye(self.nlev) + D

            Q, self.R = torch.linalg.qr(B)
            self.QT = Q.T

            # a single forward Euler step
            source = source_func(torch.hstack((u[0], sf[0], cw[0])))
            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(torch.hstack((u[n], sf[n], cw[n])))

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - D, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

        elif self.w.ndim == 1:
            # w is time-dependent (constant with height)

            # t = 1*dt
            #---------
            D = self.dt * (self.w[0] * self.D1 - self.kappa * self.D2)

            # a single forward Euler step
            source = source_func(torch.hstack((u[0], sf[0], cw[0])))
            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(torch.hstack((u[n], sf[n], cw[n])))

                DL = self.dt * (self.w[n+1] * self.D1 - self.kappa * self.D2)
                DR = self.dt * (self.w[n-1] * self.D1 - self.kappa * self.D2)

                # LHS
                B = torch.eye(self.nlev) + DL

                Q, self.R = torch.linalg.qr(B)
                self.QT = Q.T

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - DR, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

        elif self.w.ndim == 2:
            # w is time- and height-dependent (w[time, height])

            # t = 1*dt
            #---------
            D = self.dt * (torch.matmul(torch.diag(self.w[0]), self.D1) -
            self.kappa * self.D2)

            # a single forward Euler step
            source = source_func(torch.hstack((u[0], sf[0], cw[0])))
            u[1] = (torch.matmul(torch.eye(self.nlev) - D, u[0]) -
            self.dt * source)

            # t = n*dt
            #---------
            for n in range(1, nsteps - 1):

                self.current_time += self.dt
                source = source_func(torch.hstack((u[n], sf[n], cw[n])))

                DL = self.dt * (torch.matmul(torch.diag(self.w[n+1]), self.D1) -
                self.kappa * self.D2)
                DR = self.dt * (torch.matmul(torch.diag(self.w[n-1]), self.D1) -
                self.kappa * self.D2)

                # LHS
                B = torch.eye(self.nlev) + DL

                Q, self.R = torch.linalg.qr(B)
                self.QT = Q.T

                # RHS multiplied by QT on the left
                b = torch.matmul(self.QT, (
                    torch.matmul(torch.eye(self.nlev) - DR, u[n - 1]) -
                    2 * self.dt * source
                )).reshape(-1, 1)

                u[n + 1] = torch.triangular_solve(b, self.R).solution.flatten()

        return u