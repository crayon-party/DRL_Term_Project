import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp

class BatchEnv(gym.Env):
    metadata = {'render_fps': 30}
    def __init__(self, seed: int = None):
        super().__init__()
        self.simulator = Simulator(seed)
        self.action_space = spaces.Discrete(1 + 3*3) # 1 action (stop & reset) + 3*3 actions (feed A and B)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 20, 2.5, 1.4], dtype=np.float32), 
                                            high=np.array([10, 10, 10, 10, 100, 17.5, 6.6], dtype=np.float32), shape=(7,))
        self.render_mode = None
        self.cum_A_fed = 0.0 #mol
        self.cum_B_fed = 0.0 #mol
        self.gamma = 0.95
        self.has_fed = getattr(self, "has_fed", False)

        self.step_count = 0
        self.max_steps = 30

    def _get_obs(self):
        return self.simulator._get_obs()

    def reset(self, seed: int = None, options: dict = None):
        self.simulator = Simulator(seed)
        state = self.simulator.reset()
        self.cum_A_fed = 0.0
        self.cum_B_fed = 0.0
        self.profit_tracker = 0.0
        self.has_fed = False
        self.step_count = 0

        return state, {'time': self.simulator.time}

    def step(self, action: int):
        P_A, P_B, P_D = self.simulator.P_A, self.simulator.P_B, self.simulator.P_D
        prev_state = self._get_obs()

        # CALCULATE PURITY BEFORE reset (for STOP actions)
        current_purity = None
        if action == 0:
            current_state = self.simulator.state
            current_purity = current_state[2] / (current_state[2] + current_state[3] + 1e-6)
            #print(f"PRE-STOP purity={current_purity:.3f}, D={current_state[2]:.2f}, U={current_state[3]:.2f}")

        reward = 0.0
        terminated = truncated = False

        # Action decoding + simulation (same as before)
        if action == 0:
            clean = True;
            F_A = F_B = 0.0
        else:
            clean = False
            F_A_bin = F_B_bin = [0.0, 5.0, 10.0]
            F_A = F_A_bin[(action - 1) % 3]
            F_B = F_B_bin[(action - 1) // 3]

        next_state, info = self.simulator.step(clean, F_A, F_B)

        # Feed handling (non-STOP)
        if not clean:
            cost = info['used_A'] * P_A + info['used_B'] * P_B
            self.profit_tracker -= cost
            self.cum_A_fed += info['used_A']
            self.cum_B_fed += info['used_B']
            self.has_fed = True

            # Shaping
            phi_prev = prev_state[6] * prev_state[2] * prev_state[4] - 0.5 * self.cum_A_fed - prev_state[
                5] * self.cum_B_fed
            phi_next = next_state[6] * next_state[2] * next_state[4] - 0.5 * self.cum_A_fed - next_state[
                5] * self.cum_B_fed
            shaping = 0.01 * (self.gamma * phi_next - phi_prev)
            reward += shaping

        # Illegal STOP
        if action == 0 and not self.has_fed:
            return prev_state, -5.0, False, False, {"illegal_stop": True}

        # TERMINAL - USE PRE-CALCULATED PURITY
        if action == 0:
            terminated = True
            if current_purity >= 0.7:  # Use PRE-STOP purity!
                true_profit = self.profit_tracker + info['sold_D'] * P_D
                bonus = 0.02 * abs(true_profit)  # 2% bonus
                reward += true_profit
                #print(
                #    f"CLEAN true_profit={true_profit:.1f}$ (D={info['sold_D']:.0f}, costs={self.profit_tracker:.0f}$)")
            self.profit_tracker = 0.0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        return next_state, reward, terminated, truncated, info


class Simulator:
    """
    Semi-Batch Reactor simulator
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

        # Parameters
        self.V_max = 100.0       # L, maximum volume of reactor
        self.V_0 = 20.0          # L, initial volume
        self.C_A_0 = 10.0        # mol/L, concentration of A in raw material A
        self.C_B_0 = 10.0        # mol/L, concentration of B in raw material B
        self.dt = 2.0            # hr, 1 timestep

        self.k_D = 0.100        # L/(mol·hr), main reaction rate constant (D)
        self.k_U = 0.025        # L^2/(mol^2·hr), side reaction rate constant (U)

        self.P_A = 0.5           # $/mol, price of A (fixed)

        self.P_B_mu = 1.0
        self.P_B_phi = 0.06
        self.P_B_sigma_eps = 0.2
        self.P_B_stat_sigma = self.P_B_sigma_eps / np.sqrt(1 - self.P_B_phi**2)

        self.P_D_mu = 4.0
        self.P_D_phi = 0.7
        self.P_D_sigma_eps = 0.6
        self.P_D_stat_sigma = self.P_D_sigma_eps / np.sqrt(1 - self.P_D_phi**2)

        # sampling initial market prices
        P_B = self.rng.normal(self.P_B_mu, self.P_B_stat_sigma)
        P_D = self.rng.normal(self.P_D_mu, self.P_D_stat_sigma)

        self.P_B = np.maximum(0.1, P_B)
        self.P_D = np.maximum(0.1, P_D)

        # internal variables used during simulation
        self.state = None
        self.time = 0

    def _get_obs(self):
        return self.state.copy()

    def _get_new_prices(self):
        P_B = (1 - self.P_B_phi) * self.P_B_mu + self.P_B_phi * self.P_B + self.P_B_sigma_eps * self.rng.normal(0, 1)
        P_D = (1 - self.P_D_phi) * self.P_D_mu + self.P_D_phi * self.P_D + self.P_D_sigma_eps * self.rng.normal(0, 1)

        self.P_B = np.maximum(0.1, P_B)
        self.P_D = np.maximum(0.1, P_D)

    def _model(self, t, y, F_A, F_B, dV):
        """
        define ODE system for scipy.integrate.solve_ivp
        y = [N_A, N_B, N_D, N_U, V] (vector of moles and volume)
        """
        N_A, N_B, N_D, N_U, V = y

        # prevent volume from being too small
        if V < 1e-6:
            return [0, 0, 0, 0, F_A + F_B]

        C_A = N_A / V
        C_B = N_B / V

        # reaction rates
        r_D = self.k_D * C_A * C_B
        r_U = self.k_U * C_A * C_B**2

        # Mole Balance
        dN_A_dt = -(r_D + r_U) * V + F_A * self.C_A_0
        dN_B_dt = -(r_D + r_U) * V + F_B * self.C_B_0
        dN_D_dt = r_D * V
        dN_U_dt = r_U * V
        dV_dt = dV

        if V >= self.V_max:
            V_out = dV
            dV_dt = 0.0
            dN_A_dt -= V_out * C_A
            dN_B_dt -= V_out * C_B
            dN_D_dt -= V_out * N_D / V
            dN_U_dt -= V_out * N_U / V

        return [dN_A_dt, dN_B_dt, dN_D_dt, dN_U_dt, dV_dt]

    def reset(self):
        """
        initialize reactor state
        """
        self.state = np.array([
            self.C_A_0,  # C_A
            0.0,         # C_B
            0.0,         # C_D
            0.0,         # C_U
            self.V_0,    # V
            self.P_B,    # P_B
            self.P_D     # P_D
        ])

        return self._get_obs()

    def _stop(self):
        """
        Helper: handle 'Stop & Reset' action
        """
        current_V = self.state[4]
        current_D = self.state[2]
        total_D = float(current_V * current_D)
        
        if self.time == 0:
            self._get_new_prices()

        next_state = self.reset()
        info = {'info': 'Batch stopped by agent.', 'time': self.time, 
        'sold_D': total_D, 'used_A': self.V_0 * self.C_A_0, 'used_B': 0.0,
        'overflow': 0.0}

        return next_state, info

    def step(self, clean: bool, F_A: float, F_B: float):
        """
        Receive action from agent and simulate for 1 timestep (self.dt)
        clean: bool
        if clean action is taken, F_A and F_B is ignored.
        F_A: float
        F_B: float
        :return: next_state, info
        """
        # update time
        self.time = (self.time + self.dt) % 24

        # --- 1. handle 'Stop & Reset' action ---
        if clean:
            return self._stop()

        # --- 2. ODE simulation ---

        # current state (concentration) -> moles
        C_A, C_B, C_D, C_U, V, P_B, P_D = self.state
        y0 = np.array([
            C_A * V,  # N_A
            C_B * V,  # N_B
            C_D * V,  # N_D
            C_U * V,  # N_U
            V         # V
        ])

        dV = F_A + F_B
        # NOISE
        F_A_noisy = F_A * self.rng.normal(1, 0.05)
        F_B_noisy = F_B * self.rng.normal(1, 0.05)

        sol = solve_ivp(lambda t, y: self._model(t, y, F_A_noisy, F_B_noisy, dV), (0, self.dt), y0, method='BDF')
        N_A_end, N_B_end, N_D_end, N_U_end, V_end = sol.y[:, -1]

        overflow = np.maximum(0, V + dV * self.dt - self.V_max)
        V_end_clipped = np.clip(V_end, 1e-6, self.V_max)
        N_A_end = N_A_end * V_end_clipped / V_end
        N_B_end = N_B_end * V_end_clipped / V_end
        N_D_end = N_D_end * V_end_clipped / V_end
        N_U_end = N_U_end * V_end_clipped / V_end
        V_end = V_end_clipped

        # --- 3. define next state ---
        if self.time == 0:
            self._get_new_prices()

        self.state = np.array([
            N_A_end / V_end,  # new C_A
            N_B_end / V_end,  # new C_B
            N_D_end / V_end,  # new C_D
            N_U_end / V_end,  # new C_U
            V_end,            # new V
            self.P_B,         # P_B
            self.P_D          # P_D
        ])
        
        info = {'time': self.time, 'sold_D': 0.0, 'used_A': F_A * self.dt * self.C_A_0, 'used_B': F_B * self.dt * self.C_B_0, 'overflow': overflow}

        return self._get_obs(), info

# --- test code for simulator ---
if __name__ == "__main__":
    action_list = [4, 4, 1, 2, 1, 1, 1, 1, 0, 4, 4, 2, 5, 1, 1, 4, 1, 1, 0] 
    undiscounted_cumulative_return = 0.0
    env = BatchEnv()
    state, info = env.reset(seed=42)
    print(f"Time: {info['time']} hr")
    print(f"A: {state[0]:.2f} mol/L, B: {state[1]:.2f} mol/L, D: {state[2]:.2f} mol/L, U: {state[3]:.2f} mol/L, V: {state[4]:.2f} L, P_B: {state[5]:.2f} $/mol, P_D: {state[6]:.2f} $/mol")
    reward_list = []
    for a in action_list:
        action = a
        next_state, reward, terminated, truncated, info = env.step(action)
        print("")
        if action == 0:
            print("Clean")
            
        print(f"A Feed: {info['used_A']} mol, B Feed: {info['used_B']} mol")
        if info['sold_D'] > 0:
            print(f"Sold D: {info['sold_D']:.2f} mol")
        if info['overflow'] > 0:
            print(f"Overflow: {info['overflow']:.2f} L")
        print(f"Reward: {reward:.2f} $")
        undiscounted_cumulative_return += reward
        reward_list.append(reward)

        print("--------------------------------")
        print(f"Time: {info['time']} hr")
        print(f"A: {next_state[0]:.2f} mol/L, B: {next_state[1]:.2f} mol/L, D: {next_state[2]:.2f} mol/L, U: {next_state[3]:.2f} mol/L, V: {next_state[4]:.2f} L, P_B: {next_state[5]:.2f} $/L, P_D: {next_state[6]:.2f} $/mol")

    print(f"Undiscounted cumulative return: {undiscounted_cumulative_return:.2f} $")
    print(f"Average daily return: {undiscounted_cumulative_return / (len(reward_list) / 12):.2f} $ / day")