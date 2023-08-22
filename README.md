# MTCP

#### Short disclaimer

This project is a part of my master thesis. At the end of it, we had little time and had to rush with everything, so the
overall structure might not be very good. Anyway, I trust that you will figure it out. Feel free to fork this repository
and modify it to better suit your needs.

Just in case you wonder, MTCP stands for the **M**aster **T**hesis **C**ERN **P**roject.

## Structure

There are several parts to this project:

1. [`trex-fitter`](./trex-fitter) - contains the stuff to work with the
   [TRExFitter](https://gitlab.cern.ch/TRExStats/TRExFitter).
2. [`data_processing`](./data_processing) - here is the code that processes input data - from the n-tuples in the
   `.root` files we create the `.npy` files used for training. Different pre-processing (e.g. normalization) happens here.
3. [`ml`](./ml) - Here you can find the code for training and evaluating your neural networks.
4. [`friend_ntuples`](./friend_ntuples) - here, we have a good model, we can use it for prediction on the systematic
   n-tuples. This produces the friend n-tuples, which you would then feed into the TRExFitter.
5. [`jobs`](./jobs) - Here I tried to organize the whole process into jobs that could be run with the
   [pipeliner](./pipeliner/), but I was not able to do it in time, unfortunately. You are encouraged, however, to follow
   the examples provided there and improve the system.
6. [`pipeliner`](./pipeliner) - This is my attempt to make a system to manage this whole mess with different jobs on
   different clusters/machines. I wasn't able to really finish it, but you can try to do it in case you are interested and
   have some free time. It's briefly described in the corresponding README.
7. [`thesis`](./thesis) - Stuff for the thesis.

## Installation

My approach was the following (maybe you can find a better one):

1. Clone this repository to the lxplus.
2. Clones this repository to your local machine as well (this is useful for some early training/testing if you have a
   decent GPU.)
3. Optionally clone this to some other cluster where a really good GPU is provided (you can do it later).
4. In case you have enough space, I recommend also copying the input n-tuples to your local machine. For me, on v0801 it
   was about 40 GB, so it was manageable. You would probably need to locate the input n-tuples. Word of advice: check
   what other colleagues are doing and keep the same path. For me, the path was
   `/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801/systematics-full/nominal`, specified in the
   [corresponding TRExFitter config file](./trex-fitter/replacements/replacement.txt).

5. Now in all the places you have cloned the repository, create the Python environment and install the required
   packages. On my local machine, I used [micromamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html)
   (like [miniconda](https://docs.conda.io/en/latest/miniconda.html), but much much faster). On lxplus, you will have to use
   `venv`: `python3 -m venv venv --system-site-packages`. The `--system-site-packages` option allows the `ROOT` to be
   available (I think it was needed in the later stages).
   The required packages are specified in the [`requirements.txt`](./requirements.txt) (for `pip`) and in the
   [`environment.yml`](./environment.yml) (for `conda`/`mamba`). I assume you know how to work with these tools.

6. TRExFitter, oh TRExFitter. This one was difficult for me. You can use the [docker
   image](https://gitlab.cern.ch/TRExStats/TRExFitter#using-a-docker-image) to run locally, but I just gave up on that idea
   and was using it always on lxplus. So, you would need to clone it (on lxplus7 only), then build from source, as
   described [here](https://gitlab.cern.ch/TRExStats/TRExFitter#setup) (remember to clone with git submodules)). **Make
   sure to then also copy the
   [`jobSchema.config`](https://gitlab.cern.ch/atlasHTop/ttHMultiGFW2/-/blob/CTU-output/CombinedFit_ttHML/v0801/2lSS_FriendTree/jobSchema.config)
   to the TRExFitter folder. This is needed for the friend trees production.** Remember to `source setup.sh` in each shell
   session before you try to do anything there. For me, it was one of the strange pitfalls.

## Usage

### Check the path to the input n-tuples, produce pre-fit/yields plots.

(as you will be using the TRExFitter, remember to `source setup.sh` in each shell session before you try to do anything)

-   `cd trex-fitter`
-   `trex-fitter n configs/pre-fit.config` - read the n-tuples
-   `trex-fitter w configs/pre-fit.config` - create the workspace
-   `trex-fitter d configs/pre-fit.config` - draw the yields plots
-   Now check the [outputs folder](./trex-fitter/outputs/pre-fit/Plots/) for the plots. You should see something like:

![pre-fit yields](./trex-fitter/outputs/pre-fit/Plots/lep-pt-0.pdf)
