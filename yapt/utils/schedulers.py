class LinearScheduler:

    def __init__(self, init_step, final_step, init_value, final_value):
        assert final_step >= init_step
        self.init_step = init_step
        self.final_step = final_step
        self.init_value = init_value
        self.final_value = final_value

    def get_value(self, step):

        if step < self.init_step:
            return self.init_value
        elif step >= self.final_step:
            return self.final_value
        else:
            if self.init_step == self.final_step:
                return self.final_value

            rate = (float(self.final_value - self.init_value) /
                    float(self.final_step - self.init_step))
            return self.init_value + rate * (step - self.init_step)

class Scheduler:
    def __init__(self, parameters):

        self.parameters = parameters
        self.last_epoch = -1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        for init_val, param in zip(state_dict['parameters'], self.parameters):
            param.fill_(init_val)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for val, param in zip(self.next_vals(), self.parameters):
            param.fill_(val)

    def next_vals(self):
        raise NotImplementedError


class LambdaScheduler(Scheduler):

    def __init__(self, parameters, lambdas):
        super().__init__(parameters)
        self.lambdas = lambdas

    def next_vals(self):
        return [lmbd(self.last_epoch) for lmbd in self.lambdas]


class StepScheduler(Scheduler):

    def __init__(self, parameters, step_size, gammas=0.1):
        super().__init__(parameters)
        self.step_size = step_size
        self.gammas = self._extend(gammas, len(parameters))

    @staticmethod
    def _extend(val, length):
        if isinstance(val, list):
            assert len(val) == length
        else:
            val = [val] * length
        return val

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.step_size = state_dict['step_size']
        self.gammas = state_dict['gammas']

    def next_vals(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return self.parameters
        return [param * gamma for param, gamma in zip(self.parameters, self.gammas)]
