# How it works

This is the long-form companion to the [README](../README.md): the method of Gatys et al., the
exact losses this code optimises, how they relate to the papers, and every place where this
implementation deviates from them. Numbers come from runs described in
[`docs/gallery/README.md`](gallery/README.md), each with the command that reproduces it.

## 1. Style transfer as image optimisation

Given a content photo $c$ and a painting $s$, Gatys, Ecker & Bethge (2016) synthesise an image
$x$ by minimising

$$
\mathcal{L}(x) = \alpha\,\mathcal{L}_\text{content}(x, c) + \beta\,\mathcal{L}_\text{style}(x, s)
               + \gamma\,\mathrm{TV}(x)
$$

over the **pixels** of $x$. Both loss terms are defined on the activations of a fixed image
classification network, VGG-19; the network's weights never change. Each evaluation is one
forward and one backward pass through the network, and every step of the optimiser needs at least
one evaluation, which is why the method is slow: a step at 256 × 384 px took 0.95 s on the shared
4-vCPU VM used here (two threads, load average about 2.4; `docs/gallery/speed.json`).
Feed-forward networks trained to approximate the optimisation (Johnson et al., 2016) are much
faster but give up the flexibility this repository is about.

## 2. The feature space: why VGG-19, and which VGG-19

VGG-19 (Simonyan & Zisserman, 2015) is a plain stack of sixteen 3 × 3 convolutions with ReLUs,
separated by five 2 × 2 pooling layers, with no normalisation layers or skip connections. Its
activations form a hierarchy: early layers respond to colours and edges within a few pixels,
deeper layers to textures and object parts over large regions (the receptive field of `relu4_2`
is 84 × 84 pixels, that of `relu5_1` 156 × 156). Gatys et al. built both of their representations
on it, and it has remained the default network for optimisation-based style transfer; Wang, Li &
Vasconcelos (2021) study why other architectures such as ResNet stylise markedly worse unless their
activations are smoothed.

Layer names follow the VGG convention: `conv4_2` is the second convolution of block 4, `relu4_2`
its rectified output, `pool4` the pooling layer that ends the block
([`neural_style/layers.py`](../neural_style/layers.py)).

**Two sets of weights.** `--weights torchvision` uses torchvision's ImageNet weights, which expect
RGB input normalised by the ImageNet mean and standard deviation. `--weights caffe` uses the
original Caffe release of Simonyan & Zisserman, the one Gatys et al. used, which expects Caffe's
input convention: channels in BGR order, values on a 0–255 scale, the mean pixel
$(103.939, 116.779, 123.68)$ subtracted, and no division. The two are different networks, and the
same loss weights produce different results with them, so each weight source has its own defaults
(section 5). Only the Caffe weights were available on the machine that produced the gallery
(`download.pytorch.org` was unreachable), so every result shown here uses them.

**Getting the Caffe weights without Caffe.** François Chollet converted the Caffe weights for
Keras and published them as a GitHub release asset; the file's MD5 equals the `file_hash` that
`keras.applications.vgg19` pins for it, and this repository pins its SHA-256
([`neural_style/weights.py`](../neural_style/weights.py)). Keras stores a convolution kernel as
(height, width, in, out) and PyTorch as (out, in, height, width), so the converter transposes
`(3, 2, 0, 1)`. TensorFlow's `conv2d`, like PyTorch's, computes a cross-correlation, so no spatial
flip is needed. The conversion is checked three ways:

1. *Offline, in CI*: on a tiny synthetic HDF5 file in the Keras layout, the converted layers
   reproduce a direct NumPy cross-correlation of the raw kernels to within $10^{-10}$, while
   spatially flipped kernels do not ([`tests/test_weights.py`](../tests/test_weights.py)).
2. *On the real file*: the converted `conv1_1` on a Caffe-preprocessed photo matches the NumPy
   cross-correlation to $1.1 \times 10^{-13}$ (float64).
3. *Semantically*: [`scripts/check_caffe_weights.py --full`](../scripts/check_caffe_weights.py)
   rebuilds the complete network, classifier included, from the 575 MB Keras file and classifies
   224 × 224 centre crops of the example photos. It is run once with the converted kernels and
   once with every kernel flipped, which is what a wrong convention would produce:

   | Photo | Converted kernels (top 3) | Flipped kernels (top 3) |
   |---|---|---|
   | `chelsea.png` | Egyptian cat 0.38, tiger cat 0.33, tabby 0.28 | tiger cat 0.38, tabby 0.30, Egyptian cat 0.28 |
   | `coffee.png` | espresso 0.97, cup 0.02, coffee mug 0.00 | espresso 0.28, soup bowl 0.18, consomme 0.17 |
   | `rocket.jpg` | missile 0.40, drilling platform 0.23, projectile 0.19 | crane 0.55, chime 0.09, pole 0.04 |

   The converted network is right on all three photos. The flipped one still finds the cat,
   which is roughly symmetric, but loses the espresso and calls the launch pad a crane (the
   machine, ImageNet class 517).

## 3. Content: feature maps

Let $F_l(x) \in \mathbb{R}^{N_l \times M_l}$ be the activations of layer $l$, with $N_l$ channels
and $M_l = H_l W_l$ positions. The content loss compares feature maps position by position:

$$
\mathcal{L}_\text{content} = \sum_{l \in \text{content}} \frac{1}{N_l M_l}
    \sum_{i,j} \big(F_l(x) - F_l(c)\big)_{ij}^2 .
$$

A deep layer (`relu4_2` by default) keeps the arrangement of objects but not the exact pixel
values, which is what leaves room for the style.

## 4. Style: second-order feature statistics

The style representation discards position altogether. The Gram matrix

$$
\hat G_l(x) = \frac{1}{N_l M_l}\, F_l(x)\, F_l(x)^\top \in \mathbb{R}^{N_l \times N_l}
$$

holds, for every pair of channels, their correlation across all positions: an uncentred
second-order statistic of the distribution of feature vectors. It is unchanged by any permutation
of positions, so it describes texture (which features occur together) and not layout (where
they occur). Gatys, Ecker & Bethge (2015) introduced it for texture synthesis; Li et al. (2017)
showed that matching Gram matrices is equivalent to minimising the maximum mean discrepancy
between the two sets of feature vectors under a second-order polynomial kernel, so style transfer
can be read as matching feature distributions. Using several layers, from `relu1_1` (receptive
field 3 pixels) to `relu5_1` (156 pixels), matches the statistics at every scale in between (the
parameter grid in the gallery shows what stopping earlier does). The style loss is

$$
\mathcal{L}_\text{style} = \sum_{l \in \text{style}} \frac{1}{N_l^2}
    \sum_{i,j} \big(\hat G_l(x) - \hat A_l\big)_{ij}^2,
\qquad \hat A_l = \sum_k w_k\, \hat G_l(s_k),
$$

where the weights $w_k$ (normalised to sum to one) blend several style images. Because
$\hat G_l$ is an average over positions, the style image may have any size and aspect ratio; it is
resized to `style_scale`² times the content image's pixel count, never stretched.

## 5. Normalisation, and how it relates to the papers

Writing $G_l = F_l F_l^\top$ for the unnormalised Gram matrix, the paper defines the style term of
layer $l$ as $E_l = \frac{1}{4 N_l^2 M_l^2} \sum_{ij} (G_l - A_l)_{ij}^2$, with equal weights
$w_l = 1/5$ on `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1` and `conv5_1`. The term above is

$$
\frac{1}{N_l^2} \sum_{ij} \big(\hat G_l - \hat A_l\big)_{ij}^2
    = \frac{1}{N_l^4 M_l^2} \sum_{ij} (G_l - A_l)_{ij}^2 = \frac{4}{N_l^2}\, E_l ,
$$

so compared with the paper's equal weights it weights layer $l$ by $1/N_l^2$: the 512-channel
layers count 64 times less than `relu1_1`. This is the convention of the widely used
implementations: the PyTorch tutorial and jcjohnson/neural-style normalise the same way, and
Gatys's own PyTorch reference
([leongatys/PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer),
commit `be6c8e2`) divides the Gram matrix by $M_l$, averages over its $N_l^2$ entries and uses the
layer weights $10^3 / N_l^2$, which is exactly this loss times $10^3$.

The paper also used a *normalised* VGG-19, rescaled so that every filter's mean activation over
ImageNet images is one (a rescaling that ReLU networks allow without changing their output). This
code uses the published weights as they are; that changes the relative scale of the channels in
the losses, and it is why the paper's α/β ratios, of the order of $10^{-3}$, do not carry over.
With the Caffe weights the defaults are α = 1 and β = 100, chosen from real runs. In the gallery's
style-weight × layer grid, β = 10 stylises the photo only mildly, β = 100 stylises it strongly
while keeping it recognisable, and β = 1000 dissolves most of it.

**Presets.** The defaults depend on the weight source ([`transfer.py`](../neural_style/transfer.py)):

| Weights | Content layer | Style layers | Pooling | α | β |
|---|---|---|---|---|---|
| `caffe` | `relu4_2` | `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1` | average | 1 | 100 |
| `torchvision`, `random` | `conv2_2` | `conv1_1`, `conv1_2`, `conv2_1`, `conv2_2`, `conv3_1` | max | 1 | $10^6$ |

The Caffe preset takes the paper's layers after their ReLU, as Gatys's own PyTorch port does
(`r11` … `r51` and `r42`); that is also what a Caffe blob named `conv4_2` holds, because Caffe's
VGG definitions apply their ReLUs in place. The paper reports that replacing max pooling by
average pooling "yields slightly more appealing results" for synthesis. The torchvision preset keeps the defaults of version 0.1 (those
of the PyTorch tutorial) unchanged, because the torchvision weights could not be downloaded to
calibrate new ones; any layer can be chosen with `--content-layers` and `--style-layers`.

## 6. Optimisation: why L-BFGS

The objective is smooth, deterministic and has exactly computable gradients, and the unknowns are
a few hundred thousand pixel values. That is the setting quasi-Newton methods are made for:
L-BFGS (Liu & Nocedal, 1989) builds an approximation of the inverse Hessian from the last 100
gradient differences and takes Newton-like steps with it, while first-order methods such as Adam
(Kingma & Ba, 2015) only rescale the gradient. Gatys et al. used L-BFGS.

Measured on the Caffe VGG-19 (chelsea × The Starry Night at 192 px, from the photo, same loss;
[`docs/gallery/convergence.json`](gallery/convergence.json)):

| Optimiser | Loss after 25 | 50 | 100 | 200 evaluations | End (evaluations, seconds) |
|---|---|---|---|---|---|
| L-BFGS (default) | 3.9 × 10⁷ | 2.9 × 10⁵ | 1.21 × 10⁵ | 8.4 × 10⁴ | **7.6 × 10⁴** (300, 165 s) |
| L-BFGS + line search | 9.1 × 10⁵ | 2.6 × 10⁵ | 1.15 × 10⁵ | 8.5 × 10⁴ | 7.7 × 10⁴ (331 in 150 steps, 181 s) |
| Adam, lr 0.01 | 1.1 × 10⁶ | 4.7 × 10⁵ | 1.55 × 10⁵ | 1.06 × 10⁵ | 9.4 × 10⁴ (300, 153 s) |
| Adam, lr 0.02 | 6.0 × 10⁵ | 1.8 × 10⁵ | 1.10 × 10⁵ | 9.0 × 10⁴ | 8.4 × 10⁴ (300, 155 s) |
| Adam, lr 0.05 | 2.0 × 10⁵ | **1.2 × 10⁵** | **1.09 × 10⁵** | **8.1 × 10⁴** | 8.3 × 10⁴ (300, 154 s) |

The loss at the start is 4.9 × 10⁶. The result is not "L-BFGS is always faster": Adam with a
large learning rate descends fastest for the first hundred or so evaluations, and it plateaus
with small bumps (its loss at 300 evaluations is higher than at 200). L-BFGS reaches the lowest
loss of the five, 9 % below the best Adam run after 300 evaluations, which is why it stays the
default for full-length runs; for short previews Adam, or L-BFGS with a line search, is the
better choice. Each evaluation costs the same for all of them, so wall-clock time follows the
evaluation count (about 0.5 s per evaluation here, two threads, 1-minute load average about 4).
Plot: [`convergence-light.png`](gallery/convergence-light.png).

The image is kept in $[0, 1]$ by projecting it after every step. PyTorch's L-BFGS without a line
search (the configuration of all the reference implementations) scales its first step by the
inverse L1 norm of the gradient, which here is a tiny step, and then takes a full quasi-Newton step
from a curvature estimate measured over that tiny step. That step can overshoot badly: in the run
above the loss jumped from 4.9 × 10⁶ to 6.8 × 10⁸ and oscillated for about 40 evaluations before
the curvature estimate settled. With `--line-search` every step
satisfies the strong Wolfe conditions (Nocedal & Wright, 2006) and the loss decreases
monotonically, at the price of about two evaluations per step (331 evaluations for 150 steps
above). The default stays without it, because the plain method ends slightly lower for the same
budget; the line search is the safer choice for short runs.

## 7. Total variation

$\mathrm{TV}(x)$ is the mean absolute difference between neighbouring pixels (anisotropic, L1).
Mahendran & Vedaldi (2015) introduced such priors for inverting CNN representations, and
Johnson et al. (2016) used one for style transfer; Gatys et al. did not. It is off by default
(γ = 0). Its scale is very different from the other terms': with the Caffe weights the weighted
content and style terms end the gallery runs between about $10^3$ and $10^5$ (the JSON files next
to the images record them), while the TV of a natural image is well below one, so a useful γ is
correspondingly large.

## 8. Colour control

Style transfer copies the painting's palette along with its brushwork. Gatys, Bethge, Hertzmann &
Shechtman (2016) give two ways to keep the photo's colours instead
([`neural_style/color.py`](../neural_style/color.py)); the gallery shows both.

**Luminance-only transfer** (`--color luminance`). Content and style are converted to YIQ and only
their luminance $Y$ is kept. The style luminance is first matched to the content's mean and
standard deviation, $L_s' = \frac{\sigma_c}{\sigma_s}(L_s - \mu_s) + \mu_c$, which the paper
recommends when the two luminance histograms differ strongly; this code always applies it.
A **single-channel** image is then optimised (fed to VGG as three equal channels), and the photo's
I and Q channels are put back at the end. Version 0.1 instead stylised in RGB and swapped the
chrominance afterwards; that keeps the photo's hue, but the painting's colours have already shaped
the image (a test now checks that two style images with the same luminance but different colours
give the same result).

**Colour histogram matching** (`--color match`). Before the transfer, the style image's pixels
$p$ are mapped affinely, $p' = A(p - \mu_s) + \mu_c$, so that their mean and covariance become
the photo's: $A \Sigma_s A^\top = \Sigma_c$. Any $A = \Sigma_c^{1/2} Q\, \Sigma_s^{-1/2}$ with
$Q$ orthogonal solves this; the paper uses two:

- `--color-match eigen` (default): the Image Analogies formulation (Hertzmann et al., 2001),
  $A = \Sigma_c^{1/2}\Sigma_s^{-1/2}$ with symmetric square roots from eigendecompositions. It
  does not depend on the order of the colour channels.
- `--color-match cholesky`: $A = L_c L_s^{-1}$ with $\Sigma = L L^\top$.

Both are computed in float64 with $10^{-5}$ added to the diagonal, so a grey style image
(singular covariance) gives a finite result, and the recoloured style is clipped to $[0, 1]$
(which is the only reason its statistics can differ from the photo's). Tests check the covariance
equation for random colour distributions.

## 9. Spatial control

Gatys et al. (2017) make style *spatially* controllable with guidance channels: masks
$T_r \in [0, 1]^{H \times W}$, one per region, each paired with its own style
([`neural_style/losses.py`](../neural_style/losses.py), [`transfer.py`](../neural_style/transfer.py)).

**Guided Gram matrices.** The features are weighted by the region's mask at that layer,
$F_{l,r} = F_l \circ T_{l,r}$, and the Gram matrix is taken over the weighted features. This code
normalises it by the mask's energy instead of the number of positions,

$$
\hat G_{l,r} = \frac{F_{l,r} F_{l,r}^\top}{N_l \sum_i T_{l,r,i}^2},
$$

so that for a binary mask it is exactly the ordinary Gram matrix of the pixels inside the region
and does not depend on how large the region is; without this, a region's target would scale with
its area and a whole painting could not be matched to a part of the photo. An all-ones mask gives
the ordinary Gram matrix.

**Masks at every layer.** The masks are brought to each layer's resolution by following the
network's own structure: 3 × 3 convolutions with padding keep the size, and each pooling layer is
mirrored by an average pooling with the same kernel, stride and rounding. The downsampled masks
therefore have exactly the feature maps' shapes (odd sizes included), and masks that sum to one at
every pixel still do after downsampling. Along boundaries that do not align with the pooling grid
the downsampled masks take fractional values.

**Weighting.** Region $r$'s style term is multiplied by its share of the image,
$\frac{1}{M_l}\sum_i T_{l,r,i}^2$. The gradient that a guided Gram matrix sends to each pixel is
inversely proportional to the region's size (for a given mismatch of statistics), so without this
factor small regions would be stylised much more strongly than large ones. With it, the per-pixel
balance between content and style does not depend on the region's size, and the same β means
roughly the same strength with or without masks. Pixels outside every mask are only pulled towards
the photo and stay photographic. `--blend` weights then scale each region's strength.

Optional *style* masks (library only) restrict which part of each painting provides the
statistics, which the paper uses to transfer, for example, the sky of a painting to the sky of a
photo.

## 10. Scale control

**Coarse-to-fine synthesis** (`--size 256 512 --steps 300 100`). VGG-19 was trained on 224 × 224
images. At high resolution even `relu5_1` sees only a small part of the image, so large-scale
structures of the style (the swirls of *The Starry Night*) do not form, and every step is
expensive. Gatys et al. (2017) therefore synthesise at a low resolution first, upsample the
result, and use it to start a shorter optimisation at the high resolution, which only has to add
detail. Each scale recomputes its targets from the inputs resized to that scale.

Measured with [`scripts/gallery.py multiscale`](../scripts/gallery.py) on the 4-vCPU VM, two
torch threads, 1-minute load average 2.5–3.5 from other jobs (`docs/gallery/multiscale.json`); every
result is scored with the same full-resolution objective:

| Run (rocket × *The Starry Night*, final size 384 × 576) | Wall-clock (median of 2) | Final loss at 384 px |
|---|---|---|
| single-scale, 200 steps | 547 s | 7.61 × 10⁴ |
| coarse-to-fine, 200 steps at 192 px + 50 at 384 px | 238 s | 8.16 × 10⁴ |
| single-scale, 87 steps (the same time as coarse-to-fine) | 230 s | 1.11 × 10⁵ |

For the same time budget, coarse-to-fine ends with a 26 % lower loss than single-scale synthesis.
Given 2.3 times as long (200 steps), single-scale ends 7 % lower. A step at 192 px cost about 0.5 s here and
one at 384 px about 2.7 s, so the 200 coarse steps took less time than the 50 fine ones. The result also looks different:
the swirls laid down at 192 px span a larger part of the image than any single-scale run produces
([`docs/gallery/multiscale.jpg`](gallery/multiscale.jpg)).

**Style scale** (`--style-scale`). The size of the painting relative to the photo decides how many
pixels a brush stroke covers, while the network's receptive fields have fixed sizes in pixels. In
the gallery's example, at `--style-scale 0.5` whole motifs of *The Starry Night* (stars, small
swirls) fit into the receptive fields and reappear as small repeated blobs; at 2 the fields see
only parts of single strokes, and the result shows long, broad brushwork with more of the photo
left intact.

## 11. Deviations from the papers, in one place

- The VGG-19 weights are used as published, not in the normalised form of the 2016 paper (§5);
  the loss weights are calibrated for them.
- Layer weights are $1/N_l^2$ instead of equal (§5), as in the common implementations.
- The torchvision preset keeps version 0.1's layers (§5).
- The image is projected onto $[0, 1]$ after each step; the reference implementations let pixels
  leave the range and clip at the end (§6).
- Guided Gram matrices are normalised by mask energy and weighted by region share (§9); masks
  are downsampled by average pooling, not propagated through the convolutions.
- Luminance-only transfer always matches the style luminance's mean and standard deviation to the
  photo's, which the paper suggests when they differ strongly (§8).
- Colour-matched style images are clipped to $[0, 1]$ (§8).
- An optional total-variation term, off by default (§7).

## References

- L. A. Gatys, A. S. Ecker, M. Bethge. *Texture Synthesis Using Convolutional Neural Networks.*
  NeurIPS 2015. [arXiv:1505.07376](https://arxiv.org/abs/1505.07376)
- L. A. Gatys, A. S. Ecker, M. Bethge. *Image Style Transfer Using Convolutional Neural Networks.*
  CVPR 2016.
  [PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
  (first as *A Neural Algorithm of Artistic Style*, [arXiv:1508.06576](https://arxiv.org/abs/1508.06576))
- L. A. Gatys, M. Bethge, A. Hertzmann, E. Shechtman. *Preserving Color in Neural Artistic Style
  Transfer.* 2016. [arXiv:1606.05897](https://arxiv.org/abs/1606.05897)
- L. A. Gatys, A. S. Ecker, M. Bethge, A. Hertzmann, E. Shechtman. *Controlling Perceptual Factors
  in Neural Style Transfer.* CVPR 2017. [arXiv:1611.07865](https://arxiv.org/abs/1611.07865)
- K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image
  Recognition.* ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Y. Li, N. Wang, J. Liu, X. Hou. *Demystifying Neural Style Transfer.* IJCAI 2017.
  [arXiv:1701.01036](https://arxiv.org/abs/1701.01036)
- P. Wang, Y. Li, N. Vasconcelos. *Rethinking and Improving the Robustness of Image Style
  Transfer.* CVPR 2021. [arXiv:2104.05623](https://arxiv.org/abs/2104.05623)
- J. Johnson, A. Alahi, L. Fei-Fei. *Perceptual Losses for Real-Time Style Transfer and
  Super-Resolution.* ECCV 2016. [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
- A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them.*
  CVPR 2015. [arXiv:1412.0035](https://arxiv.org/abs/1412.0035)
- A. Hertzmann, C. E. Jacobs, N. Oliver, B. Curless, D. H. Salesin. *Image Analogies.*
  SIGGRAPH 2001.
- D. C. Liu, J. Nocedal. *On the limited memory BFGS method for large scale optimization.*
  Mathematical Programming 45, 1989.
- J. Nocedal, S. J. Wright. *Numerical Optimization*, 2nd ed. Springer, 2006.
- D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015.
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
