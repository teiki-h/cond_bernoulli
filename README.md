# cond_bernoulli

This project is based on https://arxiv.org/pdf/2012.03103, a paper that discusses how
to sample the distribution of Bernoulli variables conditional on their sum; i.e. generate
N independent variables Xn ∼Bernoulli(pn) conditional on N
n=1 Xn = k, for a certain
integer k.



We will answer this 4 questions : 

1) Show that you can use rejection sampling to sample from such a distribution,
and implement the corresponding algorithm for different values of N and p=
(p1,...,pN ). What is the main drawback of this approach? Study the acceptance
rate as a function of N (and/or different vectors p).

2) The aforementioned paper proposes an exact algorithm (see Appendix A). Imple-
ment it and explain why it has complexity O(N2).

3)  Implement the MCMC sampler proposed at the beginning of the paper, and explain
why it is valid (i.e. why it leaves invariant the target distribution). How to do
you propose to assess the mixing of this algorithm in this case (given that the
distribution of each component is discrete)?

4) Can you adapt the exact algorithm (point 2) to make use of (randomised) quasi-
Monte Carlo? Explain, and compare the performance of this RQMC approach with
the one you implemented in point 2.
