import paddle
from torchdiffeq import odeint


class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = paddle.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = paddle.randn(x.shape) #.to(x)
        t = paddle.ones(x.shape[0])*t #.to(x) * t
        dw = w_cur * paddle.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + paddle.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = paddle.randn(x.shape) #.to(x)
        dw = w_cur * paddle.sqrt(self.dt)
        t_cur = paddle.ones(x.shape[0])*t #.to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + paddle.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with paddle.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples



# def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
#     """Integrate a system of ordinary differential equations.

#     Solves the initial value problem for a non-stiff system of first order ODEs:
#         ```
#         dy/dt = func(t, y), y(t[0]) = y0
#         ```
#     where y is a Tensor or tuple of Tensors of any shape.

#     Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

#     Args:
#         func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
#             into a Tensor of state derivatives with respect to time. Optionally, `y`
#             can also be a tuple of Tensors.
#         y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
#             can also be a tuple of Tensors.
#         t: 1-D Tensor holding a sequence of time points for which to solve for
#             `y`, in either increasing or decreasing order. The first element of
#             this sequence is taken to be the initial time point.
#         rtol: optional float64 Tensor specifying an upper bound on relative error,
#             per element of `y`.
#         atol: optional float64 Tensor specifying an upper bound on absolute error,
#             per element of `y`.
#         method: optional string indicating the integration method to use.
#         options: optional dict of configuring options for the indicated integration
#             method. Can only be provided if a `method` is explicitly set.
#         event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
#             event_fn evaluates to zero. If this is not None, all but the first elements of
#             `t` are ignored.

#     Returns:
#         y: Tensor, where the first dimension corresponds to different
#             time points. Contains the solved value of y for each desired time point in
#             `t`, with the initial value `y0` being the first element along the first
#             dimension.

#     Raises:
#         ValueError: if an invalid `method` is provided.
#     """

#     shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

#     solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

#     if event_fn is None:
#         solution = solver.integrate(t)
#     else:
#         event_t, solution = solver.integrate_until_event(t[0], event_fn)
#         event_t = event_t.to(t)
#         if t_is_reversed:
#             event_t = -event_t

#     if shapes is not None:
#         solution = _flat_to_shape(solution, (len(t),), shapes)

#     if event_fn is None:
#         return solution
#     else:
#         return event_t, solution


class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"
        self.drift = drift
        self.t = paddle.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        
        def _fn(t, x):
            t = paddle.ones(x[0].shape[0]) * t if isinstance(x, tuple) else paddle.ones(x.shape[0]) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
