<!-- Page 1 -->

# Strong field physics in open quantum systems  

Neda Boroumand, $^{1}$ Adam Thorpe, $^{1}$ Graeme Bart, $^{1}$ Andrew Parks, $^{2}$ Mohamad Toutounji, $^{3}$ Giulio Vampa, $^{4}$ Thomas Brabec, $^{1, *}$ and Lu Wang (汪璐) $^{1, \dagger}$ $^{1}$ Department of Physics, University of Ottawa, Ottawa, Ontario K1N 6N5, Canada $^{2}$ Wyant College of Optical Sciences, University of Arizona, Tucson, Arizona, 85721, USA $^{3}$ College of Science, Department of Chemistry, UAE University, Al- Ain, UAE $^{4}$ Joint Attosecond Science Laboratory, National Research Council of Canada and University of Ottawa, 100 Sussex Drive, Ottawa, Ontario K1A 0R6, Canada  

Dephasing is the loss of phase coherence due to the interaction of an electron with the environment. The most common approach to model dephasing in light- matter interaction is the relaxation time approximation. Surprisingly, its use in intense laser physics results in a pronounced failure, because ionization is highly overestimated. Here, this shortcoming is corrected by developing a strong field model in which the many- body environment is represented by a heat bath. Our model reveals that ionization enhancement and suppression by several orders of magnitude are still possible, however only in more extreme parameter regimes. Our approach allows the integration of many- body physics into intense laser dynamics with minimal computational and mathematical complexity, thus facilitating the identification of novel effects in strong- field physics and attosecond science.  

## Introduction  

Strong laser- matter interaction is commonly modeled as a closed quantum system with a single active electron [1, 2]. While this assumption is well justified for atomic gases, its validity is not so clear for denser materials, such as liquids and solids. A full many- body treatment of the non- perturbative dynamics of all electrons and nuclei is prohibitively difficult. Therefore, it is more practical to model dense materials as a single active electron within an open quantum system, where many- body effects are accounted for by interactions with the environment [3, 4]. Due to its simplicity, the environment in intense laser- driven solids is mostly modeled in the relaxation time approximation [5, 6], where the effect of many- body dynamics is replaced by a dephasing time $T_{2}$ [7- 9]. Dephasing represents the destruction of the coherence between different one- electron eigenstates of the material, as a result of many- body collisions.  

However, a simple calculation for an under- resonantly driven two- level system reveals questionable features of the relaxation time approximation [10]. Throughout this paper, we refer to ionization as the laser- induced excitation of an electron from the valence $|0\rangle$ to the conduction $|1\rangle$ band, as determined by conventional optical field ionization theory [11]. In Fig. 1a the ionization dynamics with dephasing described via the relaxation time approximation (yellow) and without dephasing (blue) are compared. It can be seen that the relaxation time approximation predicts nine orders of magnitude ionization enhancement at very moderate electric field strength. This is clearly unphysical according to experimental measurements. Though this overestimation of ionization may be mitigated by introducing a time- dependent relaxation time [12, 13], the underlying issue is still not resolved. We term the ionization enhancement caused by dephasing as dephasing ionization.  

![](../images/Physics_(2025)_p1_img1.png)

FIG. 1. Illustration of under-resonantly driven, open two-level/band systems. Panel a presents the two-level system (band gap $E_{g} = 3.51 \mathrm{eV}$ ) described by the relaxation time approximation. On the right-hand side ionization with $(T_{2} = 8 \mathrm{fs}$ , yellow curve) and without dephasing $(T_{2} = \infty$ , blue curve) is compared. A moderate electric field strength $\mathrm{E}_{0} = 5 \times 10^{8} \mathrm{V / m}$ with photon energy $\sim 0.39 \mathrm{eV}$ $(\lambda_{0} = 3.2 \mu \mathrm{m})$ is chosen. See Supplement Fig.S3 for details. Panel b shows the two-band system coupled to a heat bath described via the spin-boson model. The heat bath is modeled using boson harmonic oscillator modes. As the temperature rises, boson modes with higher energies are engaged (gray curves).   

In short, dephasing ionization happens when the phase relation between the laser and the two- level system is disturbed. Therefore, the laser- driven virtual population of the excited state is transformed into real excitation i.e. dephasing ionization. Here, the virtual excitation refers to the population that disappears after the laser pulse. The apparent shortcomings of the relaxation time approximation leave a gap between more complex and



<!-- Page 2 -->

computationally demanding many- body approaches and oversimplified dephasing models commonly used in intense light- matter interaction.  

Furthermore, ionization is the first step in all strong field processes, such as material machining [14- 16], petahertz electronics [17, 18], electron acceleration from nano emitters [19], and attosecond spectroscopy in atoms, molecules and solids [2, 20]. Due to the importance of ionization, a deeper understanding of dephasing ionization is essential.  

As such, a more sophisticated model is needed that ideally maintains most of the simplicity and wide applicability of the relaxation time approximation. We borrow inspiration from the field of open quantum systems and adopt one of its key achievements, the spin- boson model, which typically serves as a minimal model to describe the quantum dynamics of an electron under the influence of the environment [21- 23]. Here, the spin- boson model is integrated into the semiconductor Bloch equations governing intense laser solid- state physics. The electron dynamics is represented by a single electron- hole, two- band model which is linearly coupled to its environment via bosonic harmonic oscillator modes, see Fig. 1b. The so- called strong field spin- boson (SFSB) model allows for a closed- form solution of the electron dynamics in an environment and in the presence of an intense laser. We refer to the environment as a heat bath in the rest of the paper.  

The SFSB fixes the pathological ionization behavior displayed by the relaxation time approximation. Nevertheless, numerical analysis of the SFSB equation reveals that ionization enhancement of up to a few orders of magnitude is still possible, but only at high temperatures. Interestingly, in the opposite low- temperature limit the heat bath can suppress ionization by up to a few orders of magnitude, which we term as dephasing suppressed ionization. This occurs when the electron and heat bath interact strongly.  

The SFSB model provides a distinctive approach to uncovering the physics of complex many- body systems with minimal computational and mathematical complexity. The predictive power of the SFSB approach can be progressively refined through either more detailed models or by fine- tuning the heat bath response through comparison with experiments. We anticipate that the SFSB framework will facilitate the discovery of new phenomena in strong- field physics and attosecond science.  

## Theory  

Our analysis starts with a single electron two- band system coupled to a bosonic heat bath via a linear interac  

tion term, [24, 25]  

$$\begin{array}{l}{{H=-\frac{1}{2}\mathcal{E}(\pmb{K}_{t},t)\sigma_{z}+\frac{1}{2}\hbar\Omega(\pmb{K}_{t},t)\sigma_{x}+\sum_{q}\hbar\omega_{q}b_{q}^{\dagger}b_{q}}}\\ {{+\sigma_{z}\sum_{q}g_{q}\left(b_{q}+b_{q}^{\dagger}\right).}}\end{array} \quad (1)$$  

Here, $\mathbf{E}(t)$ is the laser electric field, the vector potential is defined by $- \partial_{t}\mathbf{A} = \mathbf{E}$ , and $\pmb {K}_t = \pmb {K} + e\pmb {A}(t) / \hbar$ . The canonical momentum $\pmb{K}$ belongs to the shifted Brillouin zone $\overline{\mathrm{BZ}}$ . Further, $\Omega (\pmb {K}_t,t) = (2e / \hbar)\mathbf{d}(\pmb {K}_t,t)\mathbf{E}(t)$ is a generalized Rabi frequency, $e > 0$ is the elementary charge and $\hbar$ is the Planck constant; $\mathbf{d}(\pmb {K}_t,t)$ and $\mathcal{E}(\pmb {K}_t,t)$ represent transition dipole and bandgap between conduction $|1\rangle$ and valence $|0\rangle$ band, respectively. The time dependence of these quantities arises from the moving momentum frame. The Pauli matrices are denoted by $\sigma_{j}$ $(j = x,y,z)$ . Finally, $\omega_{q}$ , $b_{q}^{\dagger}$ , $b_{q}$ , and $g_{q}$ are the harmonic oscillator frequency, creation, and annihilation operators, and the coupling coefficient of a mode with momentum $\mathbf{q}$ , respectively.  

The coupling term between the heat bath and the two- band system appears exclusively in the diagonal terms of the Hamiltonian. Thus, it accounts only for dephasing, and not directly for heat- bath driven transitions between bands, i.e the off- diagonal terms. Nevertheless, due to the coupling of laser and heat bath driven dynamics [5, 26], dephasing does influence the overall ionization. In the high- temperature limit, multi- boson transitions between valence and conduction band could become relevant but are ignored here.  

The Hamiltonian shown in Eq.(1) can be further simplified. First, we perform a polaron transformation that diagonalizes the laser- free Hamiltonian [27]. This is followed by a change to the interaction picture, which results in  

$$H_{I} = -\frac{\mathcal{E}(\pmb{K}_{t},t)}{2}\sigma_{z} + \frac{1}{2}\hbar \Omega (\pmb{K}_{t},t)\left(\sigma_{+}D^{\dagger^{2}} + \sigma_{-}D^{2}\right). \quad (2)$$  

For a detailed derivation, see Supplementary Material, Section I. Here, $\sigma_{+} = (\sigma_{x} + i\sigma_{y}) / 2$ and $\sigma_{- } = (\sigma_{x} - i\sigma_{y}) / 2$ . The interactions with laser and heat bath are now described by a single term, with the shift operator defined as $D = \exp \left\{-\sum_{q}g_{q}\left[b_{q}^{\dagger}(t) - b_{q}(t)\right] / (h\omega_{q})\right\}$ .  

The evolution of the density matrix is determined by the integration of the Liouville- Von Neumann equation with the Hamiltonian shown in Eq.(2). Initially, the valence band is fully occupied, the conduction band is empty, and the heat bath is in thermal equilibrium. A closed- form solution is obtained by using a Dyson expansion up to the second order. As we are only interested in the two- band system dynamics, the heat bath degrees of freedom are traced out (see Supplementary Material Sections II- III for details) [8, 24, 25, 28- 35]. We found



<!-- Page 3 -->

that the dominant contribution to ionization is contained in the second order expansion term [33] from which the conduction band population follows as  

$$\begin{array}{l}{n_{c}(\pmb {K},t) = \frac{1}{2}\mathrm{Re}\left\{\int_{-\infty}^{t}\int_{-\infty}^{t_{1}}\Omega^{*}(\pmb{K}_{t_{1}},t_{1})\Omega (\pmb{K}_{t_{2}},t_{2})\right.}\\ {\left.\times \exp \left[iS(t_{1},t_{2}) + C(t_{1} - t_{2})\right]dt_{1}dt_{2}\right\} ,}\\ {n_{c}(t) = \int_{\overline{\mathrm{BZ}}}n_{c}(\pmb {K},t)d\pmb {K},} \end{array} \quad (3)$$  

where the action $\begin{array}{r}{S(t_{1},t_{2}) = \int_{t_{2}}^{t_{1}}d\tau \mathcal{E}_{s}(\pmb {K}_{\tau},\tau) / \hbar} \end{array}$ , and $\begin{array}{r}{\mathcal{E}_{s}(\pmb {K}_{\tau},\tau) = \sqrt{\mathcal{E}(\pmb{K}_{\tau},\tau)^{2} + \hbar^{2}\Omega(\pmb{K}_{\tau},\tau)^{2}}} \end{array}$ is the bandgap shifted by the dynamic Stark effect [33, 36, 37]. One can see from Eq.(3) that the heat bath influences are exclusively included in the correlation function  

$$\begin{array}{l}{C(t_{1} - t_{2})\approx \int_{-\infty}^{\infty}J(\omega)\Big\{i\sin [\omega (t_{1} - t_{2})]}\\ {-\{1 - \cos [\omega (t_{1} - t_{2})]\} \coth \left(\frac{\hbar\omega}{2k_{B}T}\right)\Big\} d\omega ,} \end{array} \quad (5)$$  

where $k_{B}$ is the Boltzmann constant. The temperature $T$ dependence in Eq.(5) is contained only in the $\mathrm{coh}$ term. The $g_{q}$ related terms in Eqs. (1,2) are replaced by a spectral density $J(\omega)$ through a transition from discrete to continuous modes. The spectral density depends on two parameters: coupling strength $j_{o}$ , and cutoff frequency $\omega_{c}$ . There exists a wealth of different models for the spectral density $J(\omega)$ , such as the Debye [32], Ohmic [8], Under- Damped Brownian [25, 34], Gaussian [35], and Shifted- Gaussian models, the definition of which can be found in the Supplementary Material, Sec. IV.  

The relaxation time approximation is recovered for the Debye bath in the high $T$ - limit, $C(t_{1} - t_{2})\rightarrow -(t_{1} - t_{2}) / T_{2}$ with $T_{2} = \hbar /(2\pi k_{B}Tj_{o})$ , as outlined in the Supplementary Material, Section IV.A. By contrast, the high $T$ - limits of the other heat bath models do not exhibit a linear time dependence in the exponent.  

In the context of strong laser solid interaction, the temperature $T$ refers to the local electron or ion temperature. Our approach presents an approximation, as the system, its dependence on laser pulse duration, is not always in thermal equilibrium. This can be analyzed via the well- established two- temperature model, where electrons are first heated by the laser, and then the energy is transferred to the lattice, raising its temperature. Material damage or melting is typically determined by the lattice temperature. For dielectrics, damage occurs around a few thousand K, even though the electron temperature can be much higher, reaching up to $10^{5}\mathrm{K}$ [38- 40]. While our approach can be extended to describe non- equilibrium heat baths, this would go beyond the limit of an initial investigation.  

The cutoff frequency $\omega_{c}$ falls within the Terahertz to the far- infrared range for phonons, and spans the far- infrared to the mid- infrared range for collective electronic  

excitations, such as excitons and plasmons. The coupling strength $j_{o}$ is a dimensionless parameter ranging from $10^{- 3}$ to multiples of unity [24, 25, 41- 44]. For phonons, $j_{o}< 1$ in III- V semiconductors, whereas $j_{o} > 1$ in more polar II- VI compounds [3]. Strong electron- phonon coupling $j_{o} > 1$ typically occurs in very polar materials [43, 45] such as bi- layer graphene [46], single- layer InSe [47] and superconductors [48, 49]. For collective electronic excitations, the coupling strength depends on the electron density [50]. For electron densities above $10^{20}\mathrm{cm}^{- 3}$ and for $\hbar \omega_{c}\sim 1\mathrm{eV}$ the plasmon coupling strength can become comparable to and even exceed the phonon coupling strength.  

## Results  

We have selected zinc oxide (ZnO), a representative and widely studied semiconductor. The crystal momentum $\mathbf{k}$ dependence in the entire 3D Brillouin zone is considered for the two- band system. Material parameters are derived from ab initio calculations [51- 53] (see Supplementary Material Section V, Table I). We find that both 3D and 1D calculations along the $\Gamma$ - M direction yield similar results in terms of relative heat bath induced ionization changes, both quantitatively and qualitatively (see Supplementary Material Fig.S4). Therefore, for computational efficiency, we focus on the 1D Brillouin zone along the $\Gamma$ - M direction throughout the following calculations.  

A driving laser with the center wavelength $\lambda_{0} = 3.2\mu \mathrm{m}$ is selected. The center frequency is defined as $\omega_{0} =$ $2\pi c / \lambda_{0}\approx 2\pi \times 10^{14}\mathrm{Hz}$ $\hbar \omega_{0}\sim 0.39\mathrm{eV}$ with $c$ the vacuum light velocity. The energy of the laser photons is much lower than the resonance energy of ZnO (with a band gap of $\mathcal{E}_{g} = 3.51\mathrm{eV}$ ), meaning that at least 9 photons are required to excite an electron from the valence band to the conduction band. We choose a linearly polarized electric field defined as $\mathbf{E} = \mathbf{E}_{x} =$ $\mathrm{E}_{0}\exp \left(- t^{2} / \tau^{2}\right)\cos (\omega_{0}t)$ , where $\tau = 20\mathrm{fs}$ . The electric field strength $\mathrm{E}_{0} = 1.5\times 10^{9}\mathrm{V / m}$ is well below the single pulse damage threshold of ZnO [54]. These parameter values are used throughout the paper unless otherwise stated.  

The change of ionization due to the heat bath is characterized by calculating the ionization ratio with and without the heat bath,  

$$\eta = \frac{n_{c}(j_{o}\neq 0)}{n_{c}(j_{o} = 0)}\bigg|_{t = \infty}, \quad (6)$$  

where $n_{c}(t)$ is defined in Eq.(4).  

In Fig.2a, the ionization ratio $\log_{10}(\eta)$ is plotted versus $T$ for Ohmic, Under- Damped Brownian, Gaussian, and Shift- Gaussian spectral densities, all of which follow a similar trend and yield comparable results. Thus, without loss of generality, we have chosen the Ohmic spectral



<!-- Page 4 -->

![Figure_1](../images/Physics_(2025)_p4_img1.png)
![Figure_3](../images/Physics_(2025)_p4_img3.png)
![Figure_4](../images/Physics_(2025)_p4_img4.png)
![Figure_6](../images/Physics_(2025)_p4_img6.png)
![Figure_7](../images/Physics_(2025)_p4_img7.png)
![Figure_8](../images/Physics_(2025)_p4_img8.png)
![Figure_9](../images/Physics_(2025)_p4_img9.png)
![Figure_10](../images/Physics_(2025)_p4_img10.png)

![](../images/Physics_(2025)_p4_img1.png)

FIG. 2. Panel a presents the ionization ratio versus temperature $T$ for various heat baths. Panel b shows the ionization ratio for the Debye heat bath and relaxation time approximation versus $T$ . The insets in a and b show details in the low $T$ regime. The relaxation time $T_{2} = \hbar /(2\pi k_{B}T_{j_{o}})$ obtained from the Debye spectral density, is plotted in c as a function of $T$ . The heat bath parameters are $\omega_{c} = 0.1\omega_{0}$ , $j_{o} = 0.1$ .   

density throughout the entire numerical analysis. The ionization ratio is plotted in $\log_{10}$ scale, where the positive (negative) numbers of $\log_{10}(\eta)$ correspond to the order of magnitude of enhancement (suppression) of ionization. Figure 2b shows that the Debye spectral density  

converges to the relaxation time approximation at very high temperatures. The temperature dependence of $T_{2}$ , obtained from the Debye spectral density in the high $T$ limit above, is presented in Fig. 2c. Both Debye and relaxation time approximation show an unrealistic rise of $\eta$ at low $T$ and therefore do not represent realistic heat bath models. This is to be expected, due to the unphysically long high- frequency tail of the Debye spectral density [45, 55]. Finally, by comparing the zoomed- in sections of Figs. 2a and b, one can see that the relaxation time approximation substantially overestimates ionization at low temperatures, while all the other heat baths in a show negligible changes in ionization, as detected by experiments.  

In Fig.3a, $\log_{10}(\eta)$ is scanned over a wide range of $T$ and $j_{o}$ for three representative values of $\omega_{c}$ , referring to various collective lattice or electron excitations; (i) $\omega_{c} = 0.01\omega_{0}$ , (ii) $\omega_{c} = 0.1\omega_{0}$ , and (iii) $\omega_{c} = 2.1\omega_{0}$ . Acoustic and optical phonons span the range from (i) to (ii), whereas collective electronic excitations span the range from (ii) to (iii) with excitons for ZnO around (ii) [56] and laser- excited plasmons in the spectral range around (iii) and above [57, 58]. Although electrons are fermions, their collective excitations can, to a good approximation, be treated as bosons [25, 45, 50]. As such, they can be directly modeled via the spin- boson Hamiltonian shown in Eq.(1). While all $T$ and $\omega_{c}$ ranges can be realized in intense laser- driven ZnO, the shown $j_{o}$ - dependence is not ZnO specific. We explore the typical range of $j_{o}$ defined above.  

![](../images/Physics_(2025)_p4_img2.png)

FIG. 3. Panel a shows ionization ratio $\log_{10}(\eta)$ as a function of local temperature $T \in [1, 3 \times 10^{4}] \mathrm{K}$ and coupling coefficient $j_{o} \in [0, 5]$ . The three panels represent different cutoff frequencies, (i) $\omega_{c} = 0.01\omega_{0}$ , (ii) $\omega_{c} = 0.1\omega_{0}$ and (iii) $\omega_{c} = 2.1\omega_{0}$ . b Ionization versus time for two data points $n_{c1}$ and $n_{c2}$ in panel (iii) of a; black dotted curve shows ionization in the absence of a heat bath. c same as plots for $n_{c1}$ and $n_{c2}$ in b only with setting the imaginary part of the heat bath response $C(t)$ [defined in Eq.(5)] to zero.   

The panel (i) of Fig.3a represents acoustic phonons, suggesting the heat bath has little effect on ionization. The influence of the heat bath increases with $\omega_{c}$ , as seen  

in (ii) and (iii), which embody optical phonons and electronic excitations. In the high- $T$ limit, ionization is increased by several orders of magnitude, which we call



<!-- Page 5 -->

![Figure_1](../images/Physics_(2025)_p5_img1.png)
![Figure_2](../images/Physics_(2025)_p5_img2.png)
![Figure_3](../images/Physics_(2025)_p5_img3.png)

dephasing ionization. On the other hand, at moderate $T$ with strong coupling $(j_{o} > 1)$ , ionization is suppressed by multiple orders of magnitude, which we have termed dephasing suppressed ionization. These two limits are represented by data points $n_{c1}$ , $n_{c2}$ in panel (iii) for which, the temporal evolution of ionization is plotted in Fig. 3b. The black dotted curve represents ionization in the absence of a heat bath $n_{c}(j_{o} = 0)$ .  

The increase and decrease of ionization can be explained by the real and imaginary parts of the correlation function $C(t)$ . With a given $\omega_{c}$ , at extremely high temperatures, the correlation function approaches a delta function (instantaneous) in time, leading to the Markovian limit [59]. In this limit, the real part of the correlation function dominates, and one may neglect the imaginary contribution. This is why the relaxation time approximation using $T_{2}$ as a purely real number remains a valid approximation at high temperatures. On the  

other hand, at low temperatures, the correlation function is non- Markovian with a wider distribution in time. In this case, the phase of the correlation function acts as a dynamic addition to the bandgap, increasing the original material bandgap, and thereby resulting in dephasing suppressed ionization. The importance of the heat bath phase becomes clear from a comparison of 3b and 3c. In Fig.3c the imaginary part of the $C(t)$ is set to zero, as a result of which ionization at $T = 300\mathrm{K}$ changes from suppression into enhancement.  

To gain further insight into the parameter dependence, ionization ratios are presented as a function of cutoff frequency $\omega_{c}$ in Fig.4a and of peak electric field strengths $\mathrm{E}_{0}$ in Figs.4b,c. We have chosen two different temperatures: $300\mathrm{K}$ (represented by the cold color dashed curves) and $2\times 10^{4}\mathrm{K}$ (represented by the warm- colored curves). The curves are color- coded to indicate different coupling strengths $j_{o}$ , with the values of $j_{o}$ denoted in the same color.  

![](../images/Physics_(2025)_p5_img1.png)

FIG. 4. Ionization ratio as a function of cutoff frequency $\omega_{c}$ (panel a) and of peak electric field strength $\mathrm{E}_{0}$ (panels b,c) are presented. We have chosen different values of $j_{o} \in \{0.1,1,5\}$ denoted by different colors beside each curve. The cold-colored dashed curves are for $T = 300\mathrm{K}$ ; warm-colored full curves refer to $T = 2\times 10^{4}\mathrm{K}$ . In a, the ionization without the heat bath is $n_{c}(j_{o} = 0,t = \infty) = 2\times 10^{-6}$ . Panels b,c are calculated by $\omega_{c} = 0.4\omega_{0}$ . The relaxation time used in panel c is calculated by $T_{2} = \hbar /2\pi k_{B}j_{o}T$ . The pink curve plotted on the right $y$ axis shows the ionization $n_{c}(j_{o} = 0,t = \infty)$ in the absence of the heat bath.   

Figure 4a confirms that dephasing ionization only occurs at high temperatures, while dephasing suppression ionization happens exclusively at low temperatures. Figure 4b indicates that the heat bath only plays a role at moderate electric field strengths. This can be explained by the multi- photon and tunneling ionization channels. When the electric field is strong, the Keldysh parameter $\gamma = \omega_{0}\sqrt{m^{*}\mathcal{E}_{g} / (e\mathrm{E}_{0})}$ where $m^{*}$ is the effective mass and $\mathcal{E}_{g}$ is the band gap energy, becomes smaller than 1, suggesting the tunneling effects dominate [60]. With our choice of parameters, $\gamma = 1$ corresponds to $E_{0} \approx 1.2\mathrm{V / nm}$ . Since tunneling $(\gamma < 1)$ occurs much more rapidly than multiphoton absorption [61, 62], the heat bath cannot follow the ionization process and thus has negligible influence at large $\mathrm{E}_{0}$ . In addition, while optical field ionization scales exponentially with $\mathrm{E}_{0}$ , dephasing ionization scales proportional to the laser intensity [10]. As a result, the relative importance of dephasing ionization drops for increasing laser fields. The multiphoton ionization $(\gamma > 1)$ develops over an optical cycle  

and thus is more sensitive to the non- Markovian heat bath, making it more sensitive to heat bath influences.  

In order to relate the relative ionization changes to absolute values, ionization in the absence of the heat bath $n_{c}(j_{o} = 0,t = \infty)$ , is shown as a function of $E_{0}$ in Fig.4c. At the highest field strength, ionization is approaching saturation. Moreover, the ionization ratio calculated via the relaxation time approximation is also presented. Comparing Figs.4b,c, one can see that the relaxation time approximation predicts orders of magnitude higher ionization compared to that predicted by our model.  

## Discussion  

So far, we have seen that the environment can modify ionization by orders of magnitude in the extreme limits of high $T$ or strong coupling $j_{o}$ . The environment in intense laser- solid interaction is difficult to control. There are various ways in which the environment can be engi



<!-- Page 6 -->

neered for more controlled experiments on dephasing and dephasing suppressed ionization.  

First, light modes in high- quality micro and nano- cavities can be controlled to vary from sub- poissonian, super- poissonian, poissonian, and squeezed vacuum to thermal distributions; from weak to strong coupling with electrons [63, 64]. As such, they can serve as an artificial, strongly coupled environment in which the modification of strong field processes by ionization can be investigated.  

Second, collective electron oscillations can be created in tailor- made experiments. The conduction band can be populated by doping semiconductors, or with a pump pulse in a pump- probe experiment. Ionization changes are probed with a second pulse or with transient absorption spectroscopy. As some of the effects observed here depend on strong coupling with the environment, control of the coupling strength is important. Coupling strength increases when going from bulk to 2D and 3D nano- scale materials, such as in nano- resonators and - cavities [65, 66].  

The possibility of engineering ionization has potential practical impacts. First, dephasing ionization increases ionization and thus, allows material micro- machining and - modification at lower laser intensities. This could be instrumental in generating highly charged ion states in high- density plasmas with lower pump pulse energy, contributing to the improvement of table- top X- ray sources. Second, the transition between perturbative nonlinear optics and strong field physics is marked by the onset of ionization. Dephasing suppressed ionization shifts this onset and permits probing dynamics in materials under excitation conditions previously unattainable.  

## Conclusion  

The relaxation time approximation is frequently used in intense laser field physics to account for the many- body coupling between a single electron and its environment, which consists of lattice, impurities, and remaining electrons. This work aimed to understand the failure of the relaxation time approximation and to correctly describe ionization in an open quantum system. Ionization in the presence of the relaxation time approximation is enhanced by orders of magnitude over a wide range of parameters, which is termed dephasing ionization.  

To decide whether dephasing ionization holds physical significance or is simply a failure of the relaxation time approximation, we have developed a more comprehensive model that captures more physics and still retains much of the simplicity of the relaxation time approximation.  

Our results confirmed that ionization enhancement through dephasing ionization still persists, but only in fairly extreme parameter ranges. Very little enhancement is found for acoustical phonon frequencies. For optical phonons and collective electronic excitations dephasing  

ionization becomes dominant in the limit of high temperatures. Our analysis has also revealed the possibility that a heat bath can suppress ionization by orders of magnitude, which we have named dephasing suppressed ionization.  

We presented a novel framework here to model intense laser many- body processes in a low- cost, semiphenomenological way. Future research will entail finding optimum heat baths and complementing the current framework with more physics. In addition, a simple fermionic heat bath. Though the SFSB presents a good approximation to a large class of collective excitations of electrons and lattice, it does not account for electron- electron scattering which requires an extended approach with a fermionic heat bath [67]. Besides, heat bath parameters, such as material temperature, change during intense laser interaction. As such, the ionization dynamics investigated here present only approximate snapshots. For a full treatment of laser material interaction, a dynamically evolving heat bath will have to be considered.  

## Acknowledgements  

L. W. would like to thank Fluffy B. for being the motivation to complete this project.  

\* Thomas.brabec@uottawa.ca  † lu.wangTHz@outlook.com  [1] F. Krausz and M. Ivanov, Rev. Mod. Phys. 81, 163 (2009).  [2] E. Goulielmakis and T. Brabec, Nature Photonics 16, 411 (2022).  [3] H. Haug and S. W. Koch, Quantum theory of the optical and electronic properties of semiconductors (world scientific, 2009).  [4] V. May and O. Kühn, Charge and energy transfer dynamics in molecular systems (John Wiley & Sons, 2023).  [5] G. Vampa, C. McDonald, G. Orlando, D. Klug, P. Corkum, and T. Brabec, Physical review letters 113, 073901 (2014).  [6] T.- Y. Du and C. Ma, Physical Review A 105, 053125 (2022).  [7] W. M. Witzel, M. S. Carroll, A. Morello, L. Cywinski, and S. Das Sarma, Physical review letters 105, 187602 (2010).  [8] J. P. Paz, S. Habib, and W. H. Zurek, Physical Review D 47, 488 (1993).  [9] T. Yu and J. Eberly, Physical Review B 68, 165322 (2003).  [10] C. McDonald, A. B. Taher, and T. Brabec, Journal of Optics 19, 114005 (2017).  [11] L. V. Keldysh, Zh. Eksperim. i Teor. Fiz. 47 (1964).  [12] S. Y. Kruchinin, Physical Review A 100, 043839 (2019).  [13] X. Cai, Scientific reports 10, 88 (2020).  [14] R. R. Gattass and E. Mazur, Nature photonics 2, 219 (2008).



<!-- Page 7 -->

[15] M. F. Yanik, H. Cinar, H. N. Cinar, A. D. Chisholm, Y. Jin, and A. Ben- Yakar, Nature 432, 822 (2004).[16] M. Farsari and B. N. Chichkov, Nature photonics 3, 450 (2009).[17] A. Schiffrin, T. Paasch- Colberg, N. Karpowicz, V. Apalkov, D. Gerster, S. Mühlbrandt, M. Korbman, J. Reichert, M. Schultze, S. Holzner, et al., Nature 493, 70 (2013).[18] T. Boolakee, C. Heide, A. Garzón- Ramírez, H. B. Weber, I. Franco, and P. Hommelhoff, Nature 605, 251 (2022).[19] T. Chlouba, R. Shiloh, S. Kraus, L. Brückner, J. Litzel, and P. Hommelhoff, Nature 622, 476 (2023).[20] V. Korolev, T. Lettau, V. Krishna, A. Croy, M. Zuerch, C. Spielmann, M. Waechtler, U. Peschel, S. Graefe, G. Soavi, et al., arXiv preprint arXiv:2401.12929 (2024).[21] D. Segal and A. Nitzan, Physical review letters 94, 034301 (2005).[22] A. J. Leggett, S. Chakravarty, A. T. Dorsey, M. P. Fisher, A. Garg, and W. Zwerger, Reviews of Modern Physics 59, 1 (1987).[23] M. Thorwart, E. Paladino, and M. Grifoni, Chemical Physics 296, 333 (2004).[24] C. K. Lee, J. Moix, and J. Cao, The Journal of chemical physics 136 (2012), https://doi.org/10.1063/1.4722336. [25] N. Lambert, S. Ahmed, M. Cirio, and F. Nori, Nature communications 10, 3721 (2019).[26] L. Wang, M. F. Ciappina, T. Brabec, and X. Liu, Physical Review Letters 133, 113804 (2024).[27] G. D. Mahan, Many- particle physics (Springer Science & Business Media, 2013).[28] A. Würger, Physical Review B 57, 347 (1998).[29] L. Nicolin and D. Segal, The Journal of chemical physics 135 (2011), https://doi.org/10.1063/1.3655674. [30] A. Morreau and E. Muljarov, Physical Review B 100, 115309 (2019).[31] M. Bundgaard- Nielsen, J. Mork, and E. V. Denning, Physical Review B 103, 235309 (2021).[32] H. Liu, L. Zhu, S. Bai, and Q. Shi, The Journal of chemical physics 140 (2014), https://doi.org/10.1063/1.4870035. [33] A. Thorpe, N. Boroumand, A. Parks, E. Goulielmakis, and T. Brabec, Physical Review B 107, 075135 (2023).[34] C. Meier and D. J. Tannor, The Journal of chemical physics 111, 3365 (1999).[35] D. M. Rouse, E. M. Gauger, and B. W. Lovett, Physical Review B 105, 014302 (2022).[36] A. Tóth, S. Borbély, Y. Zhou, and A. Csehi, Physical Review A 107, 053101 (2023).[37] B. J. Sussman, American Journal of Physics 79, 477 (2011).[38] J. Chen, D. Tzou, and J. Beraun, International journal of heat and mass transfer 49, 307 (2006).[39] M. Mozafariyard, Y. Liao, Q. Nian, and Y. Wang, International Journal of Heat and Mass Transfer 202, 123759 (2023).[40] E. Carpene, Physical Review B—Condensed Matter and Materials Physics 74, 024301 (2006).[41] T. Yamamoto, Y. Tokura, and T. Kato, Physical Review B 106, 205419 (2022).[42] N. Anto- Sztrikacs and D. Segal, New Journal of Physics 23, 063036 (2021).[43] C. Franchini, M. Reticcioli, M. Setvin, and U. Diebold, Nature Reviews Materials 6, 560 (2021).  

[44] L. Magazzù, P. Forn- Díaz, R. Belyansky, J.- L. Orgiazzi, M. Yurtalan, M. R. Otto, A. Lupascu, C. Wilson, and M. Grifoni, Nature communications 9, 1403 (2018).[45] J. T. Devreese and A. S. Alexandrov, Reports on Progress in Physics 72, 066501 (2009).[46] C. Chen, K. P. Nuckolls, S. Ding, W. Miao, D. Wong, M. Oh, R. L. Lee, S. He, C. Peng, D. Pei, et al., Nature 636, 342 (2024).[47] A. Lugovskoi, M. Katsnelson, and A. Rudenko, Physical Review Letters 123, 176401 (2019).[48] Y. Wu, X. Yu, J. Hasaien, F. Hong, P. Shan, Z. Tian, Y. Zhai, J. Hu, J. Cheng, and J. Zhao, Nature communications 15, 9683 (2024).[49] I. Errea, F. Belli, L. Monacelli, A. Sanna, T. Koretsune, T. Tadano, R. Bianco, M. Calandra, R. Arita, F. Mauri, et al., Nature 578, 66 (2020).[50] F. Caruso and F. Giustino, Physical Review B 94, 115208 (2016).[51] M. Goano, F. Bertazzi, M. Penna, and E. Bellotti, Journal of Applied Physics 102 (2007), https://doi.org/10.1063/1.2794380. [52] G. Vampa, C. McDonald, G. Orlando, P. Corkum, and T. Brabec, Physical Review B 91, 064302 (2015).[53] G. Vampa, T. Hammond, N. Thire, B. Schmidt, F. Légaré, C. McDonald, T. Brabec, D. Klug, and P. Corkum, Physical review letters 115, 193603 (2015).[54] D. Duff, A. Rosenfeld, S. Das, R. Grunwald, and J. Bonse, Journal of Applied Physics 105 (2009), https://doi.org/10.1063/1.3074106. [55] A. Mishchenko, N. Prokof'Ev, A. Sakamoto, and B. Svistunov, Physical Review B 62, 6317 (2000).[56] S. Fiedler, L. O. L. C. Lem, C. Ton- That, M. Schleuning, A. Hoffmann, and M. R. Phillips, Scientific reports 10, 2553 (2020).[57] A. Koch, H. Mei, J. Rensberg, M. Hafermann, J. Salman, C. Wan, R. Wambold, D. Blaschke, H. Schmidt, J. Salfeld, et al., Advanced Photonics Research 4, 2200181 (2023).[58] Y. E. Kesim, E. Battal, and A. K. Okyay, AIP Advances 4 (2014), https://doi.org/10.1063/1.4887520. [59] P. P. Hofer, M. Perarnau- Llobet, L. D. M. Miranda, G. Haack, R. Silva, J. B. Brask, and N. Brunner, New Journal of Physics 19, 123037 (2017).[60] L. V. Keldysh, SOVIET PHYSICS JETP 20 (1964).[61] A. S. Landsman, M. Weger, J. Maurer, R. Boge, A. Ludwig, S. Heuser, C. Cirelli, L. Gallmann, and U. Keller, Optica 1, 343 (2014).[62] M. Klaiber, K. Z. Hatsagortsyan, and C. H. Keitel, Physical Review Letters 114, 083001 (2015).[63] Y. Wei, Z. Liao, and X.- h. Wang, Physics Letters A 526, 129965 (2024).[64] D. Najer, I. Söllner, P. Sekatski, V. Dolique, M. C. Löbl, D. Riedel, R. Schott, S. Starosielec, S. R. Valentin, A. D. Wieck, et al., Nature 575, 622 (2019).[65] V. Di Giulio, E. Akerboom, A. Polman, and F. J. García de Abajo, ACS nano (2024), https://doi.org/10.1021/acsnano.3c12977. [66] E. Akerboom, V. Di Giulio, N. J. Schilder, F. J. García de Abajo, and A. Polman, ACS nano (2024), https://doi.org/10.1021/acsnano.3c12972. [67] Y. Michishita and R. Peters, Physical Review Letters 124, 196401 (2020).