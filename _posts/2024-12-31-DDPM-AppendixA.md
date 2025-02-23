---
title: 'DDPMs: Appendix A - Notations'
date: 2024-12-31
permalink: /posts/2024/12/blog-post-DDPM-AppendixA/
tags:
  - DDPM
---

Here, notations used in the DDPM blog series are defined. Please keep in mind, some notations are introduced later in text and it's OK to not understand them all right away
(e.g., when you just start reading).


<hr>

**Notation for input and latents**

- $$\mathbf{x}_0$$ - noise-free input images.
Together input images form a dataset $$\mathbf{x}_0=\{\mathbf{x}_0^i\}_{i=0:N-1}$$ - a collection of observation points $$\mathbf{x}_0^i$$ (one point - one image) 
with $$i=0:N-1$$ denoting a sample number. $N$ is how many images we have overall. Input images have three color channels (e.g., RGB) and a two-dimensional pixel space (e.g., $$\mathbf{x}_0^i=\{x\}_{jk}^{i}$$, where $$j=0:\text{height}-1$$ and $$k=0:\text{width}-1$$);

- $\mathbf{z}$ - latent variables; depending on the context, can denote both: single-level latent (e.g., in VAE) and an hierarchy;
- $\mathbf{z}_i$ - single latent level: for models with hierarchies of latents index $i$ denotes the level of the latent ($i=0:T$);
  - $\mathbf{z}_T$ - deepest level latent corresponding to last diffusion time $T$ ($\mathbf{z}_i$ with $i=T$);
  - $\mathbf{z}_0$ - shallowest level latent corresponding to diffusion input ($\mathbf{z}_i$ with $i=0$): $\mathbf{z}_0=\mathbf{x}_0$; typically, only $\mathbf{x}_0$-notation is used;
- $\mathbf{z}_{1:T}$ - hierarchy of latents $(\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_T)$. The deepest latent of the hierarchy can also correspond to $i<T$, e.g.,
$(\mathbf{z}_1,\mathbf{z}_2,\mathbf{z}_3)$ hierarchy;

Hierarchical latents can be denoted in various ways. In variational inference we use $$\mathbf{z}$$ to denote latents regardless of a particular form (hierarchical, Markovian or not...). While typically this notation does denote only a single latent, as in a regular VAE ($$\mathbf{z}=\mathbf{z}_1$$), it can also represent a collection or an hierarchy $$\mathbf{z}=(\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_N)$$ used in joint and conditional probabilities of, for instance, MHVAEs. In diffusion, we always assume a hierarchy of latents and use the colon notation above. For instance, for illustration purposes, let's write the conditional probability of latents given a data sample:

$$
\begin{align}
p(\mathbf{z}_{1:T}|\mathbf{x}_0)=\frac{p(\mathbf{z}_{1:T},\mathbf{x}_0)}{p(\mathbf{x}_0)}
\end{align}
$$

with evidence $$p(\mathbf{x}_0)=\int d\mathbf{z}_{1:T}\ p(\mathbf{z}_{1:T},\mathbf{x}_0) = \int d\mathbf{z}_{1}...d\mathbf{z}_T\ p(\mathbf{z}_{1},...,\mathbf{z}_{T},\mathbf{x}_0) $$.

<hr>

**Notation for probabilities**

Probabilities are denoted by $$p(\cdot)$$ and $$q(\cdot)$$. Both $$p(\cdot)$$ and $$q(\cdot)$$ can represent different quantities depending on their arguments and indices.
In particular:

- $p(\mathbf{z} \vert \mathbf{x}_0)$ - true latent posterior density given observations $\mathbf{x}_0$. Used in Variational Inference introduction. Same quantity as $q(\mathbf{z} \vert \mathbf{x}_0)$;

- $$q(\mathbf{z} \vert \mathbf{x}_0)$$ - true latent posterior density given observations $\mathbf{x}_0$. Used in forward diffusion process. Same quantity as $p(\mathbf{z} \vert \mathbf{x}_0)$;
  - $$q(\mathbf{z}_{i+1} \vert \mathbf{z}_{i})$$ - same as above but denotes the posterior density of the next level ($i+1$) latent given the previous level ($i$) latent; denotes a single Gaussian transition (aka a single Brownian jump) between above mentioned latent levels in the forward diffusion process; corresponds to a single noising step in DDPM; has an analytical expression (unlike its approximate counterpart
  $q_\phi(\mathbf{z}_{i+1} \vert \mathbf{z}_i)$ used in MHVAE (see below));

- $$q(\mathbf{z}_{i} \vert \mathbf{z}_{i+1})$$ - true diffusion forward process posterior (denoising); denotes true forward diffusion process,
which is turned back in time, hence the transition from latent level $i+1$ to $i$, and is intractable (see Reverse Diffusion Process section);
  - $$q(\mathbf{z}_{i} \vert \mathbf{z}_{i+1}, \mathbf{x}_0)$$ - same as above except conditioned on an input data sample $\mathbf{x}_0$;
  conditioning on input makes the quantity tractable (see Reverse Diffusion Process section); referred to as forward process posterior in the text;

- $q_\phi(\mathbf{z} \vert \mathbf{x}_0)$ - approximate latent posterior density given observations $\mathbf{x}_0$. Corresponds to a neural network with weights $\phi$; denotes VAE's encoder;
  - $$q_\phi(\mathbf{z}_{i+1} \vert \mathbf{z}_{i})$$ - same as above, except denoting the posterior density of the next level ($i+1$) latent given the previous level ($i$) latent; used in MHVAE's encoder;

- $p_\theta(\mathbf{x}_0 \vert \mathbf{z})$ - approximate likelihood given latents $\mathbf{z}$. Corresponds to a neural network with weights $\theta$; denotes VAE's decoder;
  - $$p_\theta(\mathbf{z}_{i} \vert \mathbf{z}_{i+1})$$ - same as above, except denoting the likelihood of the previous level ($i$) latent given the higher level ($i+1$) latent; denotes MHVAE's decoder;

- $g(\mathbf{z})$ - aggregate approximate posterior latent distribution; aims to reproduce $p(\mathbf{z} \vert \mathbf{x}_0)$; optimal approximation is denoted by $^*$; only used once in Variational Inference section (changed from $q(\mathbf{z})$ (as in <a href="https://arxiv.org/abs/1601.00670" target="_blank">Blei et al., 2018</a>) to $g(\mathbf{z})$ for consistency);

- $p(\mathbf{z})$ - prior probability density of the latent variable $\mathbf{z}$; typically, corresponds to a zero-mean unit-variance Gaussian; used in VAE; 
  - $p(\mathbf{z}_T)$ - same as above but a probability distribution of the deepest level latent (typically, Gaussian); used in DDPM and MHVAE;

- $q(\mathbf{x}_0)$ - true data probability density (notation by <a href="https://arxiv.org/abs/2006.11239" target="_blank">Ho et al., 2020</a>); the quantity we aim to learn (it is unknown to us); our dataset $\mathbf{x}_0$ corresponds to a finite set of its samples $\mathbf{x}_0 \sim q(\mathbf{x}_0)$;

- $p(\mathbf{x}_0)$ - true data probability density as $q(\mathbf{x}_0)$, but infers latents' marginalization $\int d\mathbf{z}\ p(\mathbf{x}_0,\mathbf{z})$ and, thus, corresponds to evidence; intractable due to the high-dimensional integration over all possible values of all possible latent variables (see Variational Inference section);

- $q(\mathbf{z}_1, \mathbf{z}_2, ... , \mathbf{z}_T \vert \mathbf{x}_0)$ - forward diffusion process;
  - $q_\phi(\mathbf{z}_1, \mathbf{z}_2, ... , \mathbf{z}_T \vert \mathbf{x}_0)$ - data encoding process in MHVAE;

- $p_\theta(\mathbf{z}_1, \mathbf{z}_2, ... , \mathbf{z}_T, \mathbf{x}_0)$ - joint distribution between latents and data, which corresponds to a generative model aka process in DDPM and MHVAE (see Generative Model section);
  
- $$p_\theta(\mathbf{x}_0)$$ - approximated data distribution; corresponds to the learnt reverse process marginalized over latents
$$p_\theta(\mathbf{x}_0) = \int d\mathbf{z}_{1:T} p_\theta(\mathbf{z}_{1:T}, \mathbf{x}_0)$$; evidence with respect to latents $\mathbf{z}$. Intractable as its true counterpart $p(\mathbf{x}_0)$;

Distinction between $p(\cdot)$ and $q(\cdot)$
------

- $$q(\cdot)$$ - denotes posterior probabilities (with the exception of $q(\mathbf{x}_0)$, which denotes true data distribution$^{*}$) and corresponds to transformations mapping data to latents and shallower latents to deeper latents (except forward process posterior, which is still related to forward process except turned back in time); denotes quantities associated with the forward diffusion process;

- $$p(\cdot)$$ - denotes likelihoods (with the exception of $p(\mathbf{z}|\mathbf{x}_0)$, $p(\mathbf{x}_0)$
(see $^{*}$) and $p(\mathbf{z})$, $p(\mathbf{z}_T)$) associated with a generative model $p(\mathbf{z},\mathbf{x}_0)$
and corresponds to transformations mapping latents to data and deeper latents to shallower latents; denotes quantities associated with the reverse diffusion process

In other words, $q(\cdot)$ (and associated neural net weights $\phi$,
e.g., VAE's and MHVAE's encoders)
denotes quantities corresponding to operations in the data-to-latent direction;
$p(\cdot)$ (and associated neural net weights $\theta$, e.g., decoders in VAE, MHVAE and reverse diffusion process) denotes quantities
corresponding to operations in the latent-to-data direction.

Obviously, we do not use $p(\cdot)$ and $q(\cdot)$ interchangeably,
and expand the corresponding quantities using a consistent notation
e.g.,

$$
\begin{align}
q(z|x_0)=p(z,x_0)/q(x_0)
\end{align}
$$

is wrong and should be corrected to

$$
\begin{align}
q(z|x_0)=q(z,x_0)/q(x_0)
\end{align}
$$

_$^{*}$ Our input data $$\mathbf{x}_0$$ distribution is denoted by $$\mathbf{x}_0 \sim q(\mathbf{x}_0)$$
(<a href="https://arxiv.org/abs/2006.11239" target="_blank">Ho et al., 2020</a> page 2). While it essentially
describes the same quantity as $$p(\mathbf{x}_0)$$, the latter comes from marginalization of latent variables_

$$
\begin{align}
\int d\mathbf{z}_{1:T}\ p(\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_T,\mathbf{x}_{0}) = p(\mathbf{x}_0)
\end{align}
$$

_Hence, while both $$p(\mathbf{x}_0)$$ and $$q(\mathbf{x}_0)$$ describe data distribution, $$p(\mathbf{x}_0)$$
does it in the context of image recovery/generation from latents and represents model's learned approximation of data.
$$q(\mathbf{x}_0)$$ pertains to raw input data: we, in general, do not know its distribution besides a set of samples in our hands.
Since samples are input in the forward noising process, notation of $q(\cdot)$ is used._

Bayesian Inference notations in the context of MLE
------
As mentioned in the 'Warning' in the Variational Inference section, Bayesian Inference notations above are defined with respect to latents $$\mathbf{z}$$. In particular, $$p(\mathbf{x}_0)$$ and its learnt counterpart $$p_\theta(\mathbf{x}_0)$$ correspond to evidence:
result of marginalization over latents $$\int d\mathbf{z}\ p(\mathbf{x}_0,\mathbf{z})$$ and
$$\int d\mathbf{z}\ p_\theta(\mathbf{x}_0,\mathbf{z})$$ correspondingly. At the same time, with respect to learnt parameters $$\theta$$,
$$p_\theta(\mathbf{x}_0)$$ is likelihood: data probability given a set of neural network parameters. By
maximizing the Evidence Lower Bound (ELBO), we obviously maximize evidence (with respect to latents),
which with respect to $$\theta$$ is likelihood - hence, the MLE objective (see MLE section).

Additional notation clarification for VAE, MHVAE and DDPM
---------

With many notations introduced and interchangeably used in different contexts, let me additionally discuss their utilization in different models: VAE, MHVAE, and, most importantly, diffusion.

<hr>

**Notations related to _forward_ process (VAE, MHVAE encoders)**

$$q(\cdot)$$ denotes forward-process-related probabilities: mapping the input $$\mathbf{x}_0$$ to a latent space $$\mathbf{z}$$.
It typically takes the form of $$q(\mathbf{z}|\mathbf{x}_0)$$ and approximates true but intractable posterior distribution $$p(\mathbf{z}|\mathbf{x}_0)$$
of latents $$\mathbf{z}$$ given a dataset $$\mathbf{x}_0$$.
For VAE, a posterior $$q_{\phi}(\mathbf{z}|\mathbf{x}_0)$$ is learnt, with $$\phi$$ denoting neural network weights. For MHVAE, it takes the form

$$
\begin{align}
q_\phi(\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_T|\mathbf{x}_{0}) = q_\phi(\mathbf{z}_1 \vert \mathbf{x}_{0}) \prod^T_{t=2} q_\phi(\mathbf{z}_{t} \vert \mathbf{z}_{t-1})
\end{align}
$$ 

For diffusion, as mentioned above and shown below, encoding the input into a latent space is not learnt - thus, $$q(\cdot)$$ with no $$\phi$$ subscripts. Moreover, we can _reverse_ the _forward process_ analytically and generate training examples for the actual _reverse process_. Yes, I myself find the latter sentence very confusing but, hopefully, diffusion section sheds some light to show us the exit from the cave. For now, simply note that for two neighbour levels of latents $$\mathbf{z}_i$$ and $$\mathbf{z}_{i+1}$$, we can analytically revert $$q(\mathbf{z}_{i+1}|\mathbf{z}_i)$$
into $$q(\mathbf{z}_{i}|\mathbf{z}_{i+1},\mathbf{x}_0)$$
if we know the input $$\mathbf{x}_0$$. And obviously, since $$q(\mathbf{z}_{i}|\mathbf{z}_{i+1},\mathbf{x}_0)$$
is a reverse of the forward it is still denoted by $$q(\cdot)$$.

<hr>

**Notations related to _reverse_ and _generative_ processes (VAE, MHVAE decoders)**

$$p(\cdot)$$ denotes reverse-process-related and generative modeling probabilities. For mapping latents $$\mathbf{z}$$ back to the input domain $$\mathbf{x}_0$$
it typically takes the form of $$p_{\theta}(\mathbf{x}_0|\mathbf{z})$$. When denoting a joint probability, and thus a generative model, it typically takes the form $$p_\theta(\mathbf{x}_0,\mathbf{z})$$ where subscript $$\theta$$ denotes learnt neural network weights (see, for instance, decoder in the VAE context). Since a generative model is a combination of a decoder with a prior (sampling aggregate posterior) it takes the form: $$p_\theta(\mathbf{x}_0,\mathbf{z})=p_\theta(\mathbf{x}_0|\mathbf{z})p(\mathbf{z})$$. For MHVAE we have the following generative model:

$$
\begin{align}
p_\theta(\mathbf{x}_{0},\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_T) = p(\mathbf{z}_T) p_\theta(\mathbf{x}_{0} \vert \mathbf{z}_1) \prod^T_{t=2} p_\theta(\mathbf{z}_{t-1} \vert \mathbf{z}_t)
\end{align}
$$

DDPM's generative model has the same form. The only difference is $$p(\mathbf{z}_T)$$,
which is Gaussian for DDPM. In MHVAE, the deepest latent $$\mathbf{z}_T$$
is sampled from $$q_\phi(\mathbf{z}_T \vert \mathbf{z}_{T-1})$$
(output of the encoder applied to a previous-to-the-deepest latent $\mathbf{z}_{T-1}$),
which serves as the prior for the generative process.

The joint distribution
pθ(x0:T ) is called the reverse process
Ho Johnathan - adjust throughout the text