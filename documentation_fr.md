## Table des matières

1. [Le problème de séparation des sources musicales](#1-le-probleme-de-separation-des-sources-musicales)
2. [Approches classiques : ICA et NMF](#2-approches-classiques--ica-et-nmf)
3. [La transformée de Fourier à court terme pour l'analyse audio](#3-la-transformee-de-fourier-a-court-terme-pour-lanalyse-audio)
4. [La FFT réelle et le spectre de magnitude](#4-la-fft-reelle-et-le-spectre-de-magnitude)
5. [Apprentissage profond pour la séparation de sources : vue d'ensemble](#5-apprentissage-profond-pour-la-separation-de-sources--vue-densemble)
6. [HTDemucs : une architecture hybride temps-fréquence](#6-htdemucs--une-architecture-hybride-temps-frequence)
7. [MDX-Net : un U-Net dans le domaine fréquentiel](#7-mdx-net--un-u-net-dans-le-domaine-frequentiel)
8. [Objectifs d'entraînement pour la séparation de sources](#8-objectifs-dentrainement-pour-la-separation-de-sources)
9. [Contrôle du gain et remixage des stems](#9-controle-du-gain-et-remixage-des-stems)
10. [Normalisation de crête après le mixage](#10-normalisation-de-crete-apres-le-mixage)

---

## 1. Le problème de séparation des sources musicales

La séparation de sources musicales est un cas particulier du problème de **Blind Source Separation (BSS)**. Un mélange enregistré $x(t)$ est la superposition de $J$ signaux sources latents $s_j(t)$ :

$$
x(t) = \sum_{j=1}^{J} s_j(t)
$$

où, dans le contexte musical, les quatre sources canoniques sont **vocals**, **drums**, **bass** et **other** (instruments mélodiques). Le terme "blind" reflète le fait que ni le processus de mélange ni les sources individuelles ne sont observés : seul $x(t)$ est disponible à l'inférence.

Ce problème est fondamentalement **sous-déterminé** : avec un seul canal de mélange et quatre sources, le système possède bien plus d'inconnues que d'équations à chaque instant. La reconstruction n'est possible qu'en exploitant une structure supplémentaire : les propriétés statistiques des sources, leurs caractéristiques spectrales, ou des a priori appris et encodés par un réseau de neurones.

Dans le cas stéréo, le mélange possède deux canaux $(x_L(t), x_R(t))$, ce qui fournit certains indices spatiaux (différences inter-canaux de niveau et de phase), mais le problème reste sous-déterminé puisqu'il y a toujours plus de sources que de canaux. Les méthodes modernes d'apprentissage profond contournent entièrement cette formulation sous-déterminée en traitant la séparation comme un **problème de régression supervisée** : à partir d'un grand jeu de données de stems isolés et de leurs mélanges, un réseau est entraîné à associer directement un mélange à ses sources constitutives.

---

## 2. Approches classiques : ICA et NMF

Comprendre les méthodes classiques permet de situer le progrès représenté par l'apprentissage profond.

### 2.1 Analyse en composantes indépendantes

L'**Independent Component Analysis (ICA)** suppose que les sources $s_j$ sont mutuellement statistiquement indépendantes et non gaussiennes. Étant donné un mélange multicanal $\mathbf{x}(t) = A\,\mathbf{s}(t)$ où $A$ est une matrice de mélange inconnue, l'ICA estime la **matrice de démélange** $W = A^{-1}$ en maximisant une mesure d'indépendance statistique entre les signaux reconstruits $\hat{\mathbf{s}}(t) = W\mathbf{x}(t)$.

L'objectif canonique consiste à maximiser la **non-gaussianité** des signaux séparés : d'après le théorème central limite, une somme de variables indépendantes est plus gaussienne que chacune de ses composantes prises individuellement, donc inverser le mélange revient à maximiser la non-gaussianité. Celle-ci est mesurée par la négentropie ou par le kurtosis. Appliqué bin de fréquence par bin de fréquence dans le domaine STFT, cela donne la **Frequency-Domain ICA (FDICA)**.

L'ICA souffre de deux indéterminations fondamentales : une **ambiguïté de permutation** (l'ordre des sources estimées à travers les bins fréquentiels est arbitraire) et une **ambiguïté d'échelle** (chaque source n'est reconstructible qu'à un facteur scalaire près). Plus critique encore, l'ICA exige **au moins autant de microphones que de sources**, et suppose un modèle de mélange linéaire statique, ce qui n'est vérifié ni pour une seule piste stéréo enregistrée, ni pour un mélange musical réaliste.

### 2.2 Factorisation en matrices non négatives

La **Non-negative Matrix Factorisation (NMF)** opère sur le **spectrogramme de magnitude** $V \in \mathbb{R}_{\geq 0}^{F \times T}$, où $F$ est le nombre de bins fréquentiels et $T$ le nombre de trames temporelles. La NMF cherche :

$$
V \approx W H, \quad W \in \mathbb{R}_{\geq 0}^{F \times K},\; H \in \mathbb{R}_{\geq 0}^{K \times T}
$$

où $K$ est le nombre de composantes latentes. Les colonnes de $W$ sont des **gabarits spectraux** (profils fréquentiels de sources élémentaires) et les lignes de $H$ sont leurs **activations temporelles**. La non-négativité est imposée comme a priori physiquement motivé : les spectres de magnitude sont intrinsèquement non négatifs.

La factorisation est généralement obtenue en minimisant la **divergence de Kullback-Leibler** :

$$
D_\text{KL}(V \,\|\, WH) = \sum_{f,t} \left[ V_{ft} \log \frac{V_{ft}}{(WH)_{ft}} - V_{ft} + (WH)_{ft} \right]
$$

par des règles de mise à jour multiplicatives qui préservent la non-négativité. La NMF peut séparer les composantes **harmoniques** (tonales) des composantes **percussives** (larges bandes, impulsionnelles) en exploitant les différences de structure de leurs gabarits spectraux. Cependant, la NMF est un modèle peu profond, sans mémoire et sans contexte temporel, et ses performances sur de la musique polyphonique réaliste restent nettement inférieures à celles des modèles d'apprentissage profond.

---

## 3. La transformée de Fourier à court terme pour l'analyse audio

La **Short-Time Fourier Transform (STFT)** est l'outil d'analyse central à la fois pour l'affichage du spectrogramme et pour la branche de traitement fréquentiel du réseau de neurones. Pour un signal en temps discret $x[n]$, la STFT s'écrit :

$$
X[m, k] = \sum_{n=-\infty}^{+\infty} x[n]\, w[n - m H]\, e^{-j 2\pi k n / N}
$$

où $w[\cdot]$ est la fenêtre d'analyse de longueur $N$ (ici $N = 2048$), $H$ est le pas entre trames en échantillons (ici $H = 512$), $m$ est l'indice de trame, et $k \in \{0, 1, \ldots, N/2\}$ est l'indice du bin fréquentiel.

### 3.1 La fenêtre de Hann

La **fenêtre de Hann** utilisée ici est :

$$
w[n] = \frac{1}{2}\!\left(1 - \cos\frac{2\pi n}{N-1}\right), \quad n = 0, 1, \ldots, N-1
$$

Elle réalise un bon compromis entre la **largeur du lobe principal** (résolution fréquentielle) et l'**atténuation des lobes secondaires** (suppression des fuites spectrales), avec des lobes secondaires décroissant à $-18\,\text{dB/octave}$. Cela est important en musique, où une composante spectrale forte (par exemple une note de basse à 80 Hz) ne doit pas masquer des composantes voisines plus faibles (par exemple une note mélodique à 100 Hz) à cause des fuites spectrales.

Le **recouvrement** entre trames successives vaut $(N - H)/N = 75\%$. Ce recouvrement élevé satisfait la condition **Constant Overlap-Add (COLA)** requise pour une reconstruction parfaite dans le cadre de la synthèse par Overlap-Add (OLA). En pratique, la fenêtre de Hann est compatible COLA à 50 % comme à 75 % de recouvrement, ce qui garantit qu'en sommant les trames de sortie fenêtrées on obtient une enveloppe de reconstruction plate.

### 3.2 L'affichage du spectrogramme

Le spectrogramme affiché utilise la **magnitude compressée logarithmiquement** :

$$
\tilde{X}[m, k] = \log\!\left(1 + |X[m, k]|\right)
$$

La transformation $\log(1 + \cdot)$ remplit deux fonctions. D'abord, elle évite la singularité numérique de $\log(0)$ dans les bins fréquentiels silencieux sans nécessiter de plancher de bruit explicite. Ensuite, elle compresse la très grande dynamique de la musique : le rapport entre les composantes les plus fortes et les plus faibles peut dépasser $10^6$ en amplitude ($120\,\text{dB}$), ce qui rendrait un spectrogramme en échelle linéaire visuellement inexploitable.

L'**axe fréquentiel** couvre $[0, f_s/2]$ avec $N/2 + 1 = 1025$ bins uniformément espacés, ce qui donne une résolution fréquentielle de $\Delta f = f_s / N$. Pour $f_s = 44100\,\text{Hz}$, on obtient $\Delta f \approx 21.5\,\text{Hz/bin}$. Le pas de l'**axe temporel** vaut $\Delta t = H / f_s \approx 11.6\,\text{ms}$, soit environ 86 trames par seconde.

Une subtilité mérite d'être signalée : `torch.stft` est appelé avec `center=True`, ce qui complète symétriquement le signal de $N/2$ échantillons avant le découpage en trames. Ainsi, la trame $m = 0$ est centrée sur l'échantillon $n = 0$, ce qui améliore l'alignement aux bords. L'axe temporel affiché utilise l'approximation $t_m = m \cdot H / f_s$ ; le temps centré exact est $t_m + N/(2f_s)$, soit un décalage fixe d'environ $23\,\text{ms}$ qui n'affecte pas l'interprétation perceptive.

---

## 4. La FFT réelle et le spectre de magnitude

La **FFT réelle** (`numpy.fft.rfft`) exploite la symétrie hermitienne de la TFD d'un signal réel : $X[N - k] = X^*[k]$. Pour un signal de longueur $N$, seuls les $\lfloor N/2 \rfloor + 1$ premiers coefficients sont indépendants, donc `rfft` renvoie un vecteur de cette longueur, divisant par deux le coût de calcul et de stockage par rapport à la TFD complète.

Le **spectre de magnitude normalisé** est :

$$
|S[k]| = \frac{|\text{rfft}(x)[k]|}{N}, \quad k = 0, 1, \ldots, \left\lfloor \frac{N}{2} \right\rfloor
$$

La division par $N$ convertit les coefficients bruts de la TFD en **spectre d'amplitude** exprimé dans les mêmes unités physiques que le signal d'entrée. Pour une sinusoïde pure $x[n] = A \cos(2\pi f_0 n / f_s)$, le pic au bin $k_0 = \text{round}(f_0 N / f_s)$ a une hauteur $A/2$ dans le spectre unilatéral (le facteur 2 provient de la combinaison des deux bins symétriques de la TFD). L'**axe fréquentiel** est :

$$
f_k = \frac{k \cdot f_s}{N}, \quad k = 0, 1, \ldots, \left\lfloor \frac{N}{2} \right\rfloor
$$

ce qui s'obtient de manière équivalente par `numpy.fft.rfftfreq(N, d=1/f_s)`.

### 4.1 Sous-échantillonnage pour l'affichage

Pour un fichier audio long, le spectre rfft contient des centaines de milliers de bins, ce qui rend la figure interactive peu réactive. La courbe est **décimée** à au plus 5000 points en ne conservant qu'un échantillon sur $\lceil N_\text{bins} / 5000 \rceil$. Il s'agit d'une **décimation par pas sans filtrage anti-repliement préalable**, ce qui introduit du **repliement spectral dans l'affichage** : des pics spectraux situés entre deux échantillons conservés peuvent être manqués ou déformés dans la courbe. Il s'agit seulement d'une approximation d'affichage ; la séparation et le remixage sous-jacents continuent d'opérer sur le signal en pleine résolution.

---

## 5. Apprentissage profond pour la séparation de sources : vue d'ensemble

La séparation de sources neuronale moderne est formulée comme une **estimation supervisée de masques**. Un réseau de neurones $f_\theta$ prend en entrée le spectrogramme du mélange $X[m,k]$ (ou la forme d'onde brute) et prédit un **masque souple** $\mathcal{M}_j[m,k] \in [0,1]$ pour chaque source $j$ :

$$
\hat{S}_j[m, k] = \mathcal{M}_j[m, k] \cdot X[m, k]
$$

Collectivement, les masques répartissent l'énergie du mélange entre les sources, avec $\sum_j \mathcal{M}_j[m,k] \leq 1$ pour chaque bin temps-fréquence. La source estimée dans le domaine temporel est ensuite reconstruite par STFT inverse.

Les premières méthodes profondes opéraient exclusivement dans le **domaine des magnitudes STFT**, en réutilisant la phase du mélange pour la reconstruction. Cette **approximation de phase** introduit du **musical noise**, c'est-à-dire des artefacts de phase aléatoire perceptibles comme un miroitement métallique. L'évolution vers les **modèles dans le domaine temporel** (Demucs v1-v3) a permis d'y remédier en traitant directement la forme d'onde brute, et donc en estimant implicitement la phase. La génération la plus récente, les **modèles hybrides** (HTDemucs), traite simultanément les deux représentations afin de combiner leurs forces complémentaires.

---

## 6. HTDemucs : une architecture hybride temps-fréquence

**HTDemucs** (Rouard et al., 2023) repose sur un **encodeur-décodeur à double branche** qui traite le mélange simultanément dans le **domaine temporel** et dans le **domaine fréquentiel STFT**, en fusionnant l'information au goulot d'étranglement au moyen d'un **transformer inter-domaines**.

### 6.1 La colonne vertébrale U-Net encodeur-décodeur

Chaque branche suit une topologie de **U-Net** (Ronneberger et al., 2015) : un **encodeur** contractant suivi d'un **décodeur** expansif, reliés par des **skip connections** qui transmettent les cartes de caractéristiques de chaque niveau de l'encodeur vers le niveau correspondant du décodeur.

**Encodeur.** Chaque couche $\ell$ applique une convolution à stride $S_\ell$, qui sous-échantillonne l'entrée et élargit le champ réceptif :

$$
\mathbf{h}_\ell = \sigma\!\left(W_\ell^{(\text{enc})} * \mathbf{h}_{\ell-1}\right)
$$

où $\sigma$ est une activation non linéaire (GELU). Le facteur total de sous-échantillonnage après $L$ couches d'encodeur est $\prod_\ell S_\ell$, ce qui détermine la résolution au goulot.

**Décodeur.** Chaque couche applique une convolution transposée (suréchantillonnage appris par $S_\ell$) et concatène la skip connection correspondante :

$$
\mathbf{h}'_\ell = \sigma\!\left(W_\ell^{(\text{dec})} *^\top [\mathbf{h}'_{\ell+1};\, \mathbf{h}_\ell]\right)
$$

Les skip connections préservent les détails fins qui seraient autrement détruits par la compression au goulot, et permettent une reconstruction fidèle des transitoires et de la structure spectrale fine.

La **branche temporelle** traite la forme d'onde brute. La **branche fréquentielle** traite la STFT complexe du mélange, en considérant les parties réelle et imaginaire comme une carte de caractéristiques à deux canaux. Dans cette branche fréquentielle, la **normalisation de couche est appliquée indépendamment pour chaque bin fréquentiel** : les statistiques des basses fréquences, dominées par les fondamentales de basse, diffèrent fortement de celles des hautes fréquences, davantage marquées par les harmoniques et le bruit de fond. Une normalisation globale serait donc nuisible.

### 6.2 Goulot transformer inter-domaines

Au goulot, après sous-échantillonnage complet dans les deux branches, HTDemucs insère un **transformer inter-domaines** qui permet aux deux branches d'échanger de l'information via de la **cross-attention** :

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

où les **queries** $Q$ proviennent des caractéristiques du goulot d'un domaine, et les **keys** $K$ ainsi que les **values** $V$ proviennent des caractéristiques du goulot de l'autre domaine. Cet échange bidirectionnel permet par exemple au modèle d'utiliser la structure harmonique précise visible dans le domaine fréquentiel pour affiner l'estimation des enveloppes temporelles dans le domaine temporel, et réciproquement.

En plus de cette attention croisée, chaque branche applique aussi une **self-attention** au goulot, ce qui lui donne un contexte temporel global avant l'échange inter-domaines. Ensemble, ces mécanismes d'attention permettent au réseau de raisonner sur des dépendances de longue portée, par exemple le suivi d'une ligne mélodique qui se prolonge sur plusieurs secondes. Le bloc d'attention complet suit la structure standard du **Transformer** (Vaswani et al., 2017) : attention multi-têtes, connexion résiduelle, normalisation de couche et réseau feed-forward positionnel.

### 6.3 HTDemucs fine-tuné (`htdemucs_ft`)

La variante `htdemucs_ft` est obtenue par **fine-tuning** du modèle HTDemucs de base après pré-entraînement sur MUSDB18-HQ et des données supplémentaires. Le fine-tuning utilise un taux d'apprentissage plus faible ainsi qu'une pondération de perte spécifique aux sources, allouant davantage de signal d'entraînement aux stems acoustiquement les plus difficiles, notamment **vocals** et **bass**. Cette étape améliore de manière régulière le Signal-to-Distortion Ratio (SDR) sur ces stems pour des genres musicaux variés.

---

## 7. MDX-Net : un U-Net dans le domaine fréquentiel

**MDX-Net** (Kim et al., 2021), vainqueur du Music Demixing Challenge 2021, est un séparateur opérant uniquement dans le domaine fréquentiel ; il ne possède pas de branche temporelle.

Sa brique fondamentale est l'unité **TFC-TDF** (Time-Frequency Convolution - Time-Distributed Fully Connected), qui combine deux opérations complémentaires dans chaque étage encodeur/décodeur :

- les couches **TFC** appliquent des convolutions 2D sur le plan temps-fréquence afin d'extraire des motifs locaux spectro-temporels ;
- les couches **TDF** appliquent des projections linéaires denses indépendamment à chaque trame temporelle sur l'**axe fréquentiel complet**. Cela revient à une banque de filtres sélectifs fréquentiels apprise et appliquée uniformément dans le temps, autrement dit à une généralisation pilotée par les données des approches classiques par banques de filtres.

Le U-Net TFC-TDF prédit un **masque complexe**, en opérant à la fois sur les parties réelle et imaginaire de la STFT, ce qui lui permet d'estimer conjointement magnitude et phase des sources au lieu de réutiliser la phase du mélange.

La variante `mdx_extra` est un **ensemble de modèles** entraîné avec des données supplémentaires et diverses stratégies d'augmentation. L'agrégation des sorties de plusieurs modèles entraînés indépendamment réduit la variance et améliore la robustesse sur des genres musicaux variés. L'architecture de chaque membre de l'ensemble reste le même U-Net TFC-TDF ; le gain provient de la diversité d'entraînement, non d'une différence architecturale.

---

## 8. Objectifs d'entraînement pour la séparation de sources

La fonction de perte détermine directement la qualité perceptive.

La **perte de forme d'onde $\ell_1$** minimise l'erreur absolue moyenne entre la forme d'onde estimée et la forme d'onde de référence :

$$
\mathcal{L}_{\ell_1} = \frac{1}{T} \sum_{t=0}^{T-1} \left|\hat{s}_j[t] - s_j[t]\right|
$$

La norme $\ell_1$ est plus robuste aux erreurs ponctuellement grandes que $\ell_2$ et évite les sorties excessivement lissées, au rendu "boueux", typiques des modèles entraînés par MSE.

La **perte STFT multi-échelle** évalue la qualité de reconstruction simultanément à plusieurs résolutions temps-fréquence :

$$
\mathcal{L}_\text{STFT} = \sum_r \left\| \log |X_r[\hat{s}]| - \log |X_r[s]| \right\|_F
$$

où $r$ indexe différents réglages de STFT, c'est-à-dire différentes valeurs de $N$ et $H$. Les multiples échelles permettent de capturer à la fois la structure temporelle fine (petit $N$, résolution fréquentielle grossière) et le détail spectral (grand $N$, résolution fréquentielle fine), ce qu'aucune STFT unique ne peut fournir seule.

Le **Signal-to-Distortion Ratio (SDR)**, métrique standard d'évaluation, mesure le rapport entre la puissance du signal de référence et la puissance de l'erreur résiduelle :

$$
\text{SDR} = 10 \log_{10}\!\left(\frac{\|s_j\|^2}{\|\hat{s}_j - s_j\|^2}\right) \; [\text{dB}]
$$

Les modèles de l'état de l'art atteignent des SDR de l'ordre de 8 à 10 dB sur le stem vocals du benchmark MUSDB18-HQ. Un SDR de $0\,\text{dB}$ signifie que l'erreur a la même puissance que le signal ; chaque augmentation supplémentaire de $3\,\text{dB}$ divise approximativement par deux l'énergie relative de l'erreur.

---

## 9. Contrôle du gain et remixage des stems

Après séparation, les quatre stems sont recombinés avec des **ajustements de gain** indépendants afin de produire un remix personnalisé. Le gain est exprimé en **décibels (dB)**, l'unité standard de mise à l'échelle perceptive de l'amplitude en ingénierie audio.

L'**échelle décibel** pour l'amplitude est :

$$
G_\text{dB} = 20 \log_{10}(g)
$$

où $g$ est le multiplicateur linéaire d'amplitude. La conversion inverse, d'un réglage en dB vers le multiplicateur linéaire appliqué aux échantillons audio, est :

$$
g = 10^{G_\text{dB}\, /\, 20}
$$

Le facteur $20$ plutôt que $10$ apparaît parce que le décibel a été défini à l'origine pour la **puissance**, et que la puissance est proportionnelle au carré de l'amplitude : $P \propto g^2$, donc $G_\text{dB} = 10 \log_{10}(g^2) = 20 \log_{10}(g)$.

La plage de gain des curseurs est $[-60, +12]\,\text{dB}$. À $-60\,\text{dB}$, $g = 10^{-3} = 0.001$, soit une réduction à $0.1\%$ de l'amplitude d'origine, perceptivement proche du silence. À $+12\,\text{dB}$, $g \approx 3.98$, soit presque un quadruplement de l'amplitude. Le signal remixé au temps $t$ et au canal $c$ est :

$$
y_c[t] = \sum_{j \in \mathcal{J}} g_j \cdot s_{j,c}[t]
$$

où $\mathcal{J} = \{\text{vocals, drums, bass, other}\}$. Lorsque tous les gains sont à $0\,\text{dB}$ ($g_j = 1$ pour tout $j$), on reconstruit exactement le mélange d'origine, puisque le réseau est entraîné de telle sorte que $\sum_j s_j \approx x$. Toute déviation par rapport à $0\,\text{dB}$ sur l'un des stems produit un remix créatif qui ne peut pas être obtenu à partir du mélange original seul.

---

## 10. Normalisation de crête après le mixage

Après sommation des stems pondérés par leurs gains, le signal remixé peut dépasser l'intervalle d'amplitude représentable $[-1, 1]$ pour l'audio en virgule flottante, en particulier si plusieurs stems sont amplifiés simultanément. Le **clipping dur** consistant à tronquer les échantillons à $\pm 1$ introduit une forte distorsion harmonique non linéaire, perceptible comme un bourdonnement désagréable.

Pour éviter cela, une **normalisation de crête** n'est appliquée que lorsque c'est nécessaire :

$$
y_\text{norm}[t] = \frac{y[t]}{\max_t |y[t]|} \quad \text{si } \max_t |y[t]| > 1
$$

Cela met à l'échelle tout le signal de manière uniforme afin que l'échantillon le plus fort atteigne exactement $0\,\text{dBFS}$ (décibels relatifs à la pleine échelle). Les **dynamiques relatives** entre tous les stems et tous les instants sont strictement préservées ; seul le niveau global change.

Il s'agit d'une forme de **limitation de crête** avec un rapport de compression infini au-dessus d'un seuil fixé à 1.0, appliquée globalement plutôt qu'échantillon par échantillon. Une approche plus sophistiquée reposerait sur un **limiteur look-ahead** avec des temps d'attaque et de relâchement finis, ou sur un **compresseur de dynamique**, afin d'éviter le saut de niveau abrupt qu'une normalisation globale peut provoquer lorsqu'un bref transitoire impose $\max_t |y[t]| \gg 1$. Pour cette application de remixage créatif interactif, la normalisation globale plus simple est appropriée.

---

## Références

- Rouard, S., Massa, F., & Défossez, A. (2023). Hybrid Transformers for Music Source Separation. *ICASSP 2023*.
- Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2021). Music Source Separation in the Waveform Domain. *arXiv:1911.13254*.
- Kim, Y., Choi, K., Choi, M., Kim, B., & Won, M. (2021). Kuielab-MDX-Net: A Two-Stream Neural Network for Music Demixing. *ISMIR Workshop on Music Source Separation*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
- Vaswani, A., Shazeer, N., Parmar, N., Jiang, Z., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.
- Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430.
- Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S. I., & Bitteur, R. (2017). MUSDB18 — a corpus for music separation. *Zenodo*.
