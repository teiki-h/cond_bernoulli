import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import geom
import time
from scipy.stats import chisquare
from scipy.stats import qmc


class sampler:
    def __init__(self, p, I):
        self.p = np.array(p)
        self.I = I
        self.N = len(p)
        self.g = self._pdf_bernoulli()
        self.f = self._pdf_cb()
        self.sample = None
        
        if self.I <= 0 or self.I >= self.N :
            raise ValueError("I out of bound ( I does not belong to [1,...,N-1])")
        if np.count_nonzero(self.p) < self.I:
            raise ValueError("p is such that it's impossible to simulate a random variable following CB(p,I)")
        pass
    
    def _calculate_expectation(self):
        omega = self._generate_sequences()
        res = 0
        for x in omega:
            res += self.f(x) * x
        return res
    
    def _generate_sequences(self, N = None,I = None):
        '''get all the sequences of size N with I ones and the rest are zeros
        '''
        if N is None:
            N = self.N
        if I is None:
            I = self.I
            
        positions = itertools.combinations(range(N), I)
        sequences = []
        for pos in positions:
            seq = np.zeros(N, dtype=int)  # Start with all zeros
            seq[list(pos)] = 1  # Set the specified positions to 1
            sequences.append(seq)
        return np.array(sequences)
    
    def _pdf_bernoulli(self, p=None):
        ''' density function of independents bernouilli'''
        if p is None :
            p = self.p
        
        def g(x):
            return np.prod(np.where(x == 1, p, 1 - p))
        return g
    
    def _pdf_cb(self, p = None, I = None):
        ''' density function of conditionnal bernoulli'''
        if p == None:
            p = self.p
        if I == None:
            I = self.I
        g = self._pdf_bernoulli(p)
        N = len(p)
        
        proba = 0
        for x in self._generate_sequences(N,I):
            proba += g(x)

        def f(x):
            return g(x) / proba if np.sum(x) == I else 0  # Use np.sum(x)

        return f

    def chi_squared_adequation(self,print_result = True):
        if self.sample is None:
            raise Exception("sample before using chi_squared")
        L = len(self.sample)
        omega = self._generate_sequences()
        omega = omega.tolist()
        
         # Calculate observed frequency
        omega_to_index = {tuple(val): idx for idx, val in enumerate(omega)}
        observed_freq = np.zeros(len(omega), dtype=int)
        for x in self.sample:
            observed_freq[omega_to_index[tuple(x.tolist())]] +=1
            
        #calculate expected frequency
        index_to_omega =  {v: k for k, v in omega_to_index.items()}
        expected_freq = [L * self.f(np.array(index_to_omega[i])).tolist() for i in range(len(omega))]
        
        chi2_stat, p_value = chisquare(observed_freq, expected_freq)
        if print_result:
            print("Pour le test du Chi2 avec hypothèse nulle H0 = 'notre sample suit une loi Conditionnal Bernoulli de paramètres (p,I)'")
            print(f"statistique du chi2 : {chi2_stat}")
            print(f"p-valeur : {p_value}")
            print(f"l'hypothèse nulle {'est rejeté' if p_value<=0.1 else "n'est pas rejeté"} à 10%")
        return p_value

class rejection_sampler(sampler):
    def __init__(self, p, I):
        super().__init__(p, I)
        self.M = self._compute_M()
        self.acceptances = None
        pass
    
    def _compute_M(self):
        res = 1
        for seq in self._generate_sequences(self.N,self.I):
            pr = self.f(seq) / self.g(seq) 
            if pr > res:
                res = pr
        return res
    
    def sample_generator(self, L = 1):
        samples = []
        acceptances = []
        while len(samples) < L:
            attempt = 0
            while True:
                attempt += 1 
                X = np.array([np.random.binomial(1, p_i) for p_i in self.p])
                U = np.random.uniform(0,1)
                if U<=(self.f(X)/(self.g(X)*(self.M))):
                    samples.append(X)
                    acceptances.append(attempt)
                    break
        self.sample=np.array(samples)
        self.acceptances = np.array(acceptances)
    
    def naive_sample(self, L = 1):
        samples = []
        acceptances = []
        while len(samples) < L:
            attempt = 0
            while True:
                attempt +=1
                X = np.array([np.random.binomial(1, p_i) for p_i in self.p])
                if X.sum() == self.I:
                    samples.append(X)
                    acceptances.append(attempt)
                    break
        self.sample=np.array(samples)
        self.acceptances = np.array(acceptances)
    
    def plot_acceptance_density(self):
        """Plots the empirical density function and the theoretical Geometric(1/M) density."""
        if self.acceptances is None:
            raise Exception("No samples generated yet. Run sample() first.")

        unique, counts = np.unique(self.acceptances, return_counts=True)
        empirical_density = counts / np.sum(counts)  # Normalize to get probability values

        # Theoretical geometric distribution with success probability 1/M
        k_values = np.arange(1, max(unique) + 1)  # Range of values for plotting
        theoretical_density = geom.pmf(k_values, 1 / self.M)  # Geometric PMF

        plt.figure(figsize=(8, 5))
        
        # Empirical density as a bar plot
        plt.bar(unique, empirical_density, width=0.8, color='b', alpha=0.7, edgecolor='black', label="Empirical Density")
        
        # Theoretical geometric density as a line plot
        plt.plot(k_values, theoretical_density, 'r-o', markersize=4, label=f"Theoretical Geom(1/{self.M:.2f})", linewidth=1)

        plt.xlabel("Number of Attempts Before Acceptance")
        plt.ylabel("Probability Density")
        plt.title("Empirical vs Theoretical Density of Acceptance Attempts")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

class exact_sampler(sampler):
    def __init__(self, p, I):
        super().__init__(p, I)
        self.q = self._compute_q()
        pass
    
    def _compute_q(self, N = None, I = None):
        if I is None:
            I = self.I
        if N is None:
            N = self.N
        
        q = np.zeros((I+1,N))
        
        for n in range(N):
            q[0][n] = np.prod( 1 - self.p[n:])
        
        q[1][N - 1] = self.p[N - 1]
        
        for n in range(N - 2, -1, -1):
            for i in range(1, min(I, N - n) + 1):
                q[i][n] = self.p[n] * q[i - 1][n + 1] + (1 - self.p[n]) * q[i][ n + 1 ]
        
        return q
    
    def _sample_one(self):
        X = np.zeros(self.N, dtype = int)
        i_n_min_1 = 0
        for n in range(self.N - 1):
            prob = self.p[n] * self.q[self.I - i_n_min_1 - 1][n+1] / self.q[self.I - i_n_min_1][n]
            if np.random.rand() <= prob:
                X[n] = 1
                i_n_min_1 += 1
                if i_n_min_1 == self.I:
                    break
        if i_n_min_1 != self.I:
            X[-1] = 1
        return X
    
    def sample_generator(self, L = 1):
        res = []
        for _ in range(L):
            res.append(self._sample_one())
        self.sample = np.array(res)

class RQMC_sampler(exact_sampler):
    def __init__(self, p, I):
        super().__init__(p, I)
        pass
    
    def _sample_one(self, U):
        X = np.zeros(self.N, dtype = int)
        
        i_n_min_1 = 0
        for n in range(self.N - 1):
            prob = self.p[n] * self.q[self.I - i_n_min_1 - 1][n+1] / self.q[self.I - i_n_min_1][n]
            if U[n] <= prob:
                X[n] = 1
                i_n_min_1 += 1
                if i_n_min_1 == self.I:
                    break
        if i_n_min_1 != self.I:
            X[-1] = 1
        return X
    
    def sample_generator(self, L=1, sequence = 0, RQMC = False, scramble_seq = True):
        ''' sequence :  0 if Sobol
                        1 if Halton
                        2 if true random 
            scramble_seq:   True if we scramble the sequence
                            False if we do not
            RQMC:   True if we do Randomized Quasi Monte Carlo
                    False if we do Quasi Monte Carlo'''
        res = []
                # code la partie (Randomized) Quasi Monte Carlo
        U = None
        low_discrep_sampler = None
        match sequence:
            case 0:
                low_discrep_sampler = qmc.Sobol(d=self.N, scramble = scramble_seq)
                U = low_discrep_sampler.random(n = L)
            case 1:
                low_discrep_sampler = qmc.Halton(d=self.N, scramble = scramble_seq)
                U = low_discrep_sampler.random(n = L)
            case 2:
                U = np.random.uniform(0, 1, (L,self.N))
        
        if RQMC:
            U = ( U + np.random.uniform(0, 1, (L,self.N)) ) % 1
        
        for i in range(L):
            res.append(self._sample_one(U[i]))
        self.sample = np.array(res)