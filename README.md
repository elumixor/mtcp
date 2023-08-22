# MTCP

#### Short disclaimer

-   This project is a part of my master thesis. At the end of it, we had little time and had to rush with everything, so the
    overall structure might not be very good. Anyway, I trust that you will figure it out. Feel free to fork this repository
    and modify it to better suit your needs.

-   Always check for the paths yourself! Look at what files are you running. Study the files and adjust to your needs!

-   Just in case you wonder, MTCP stands for the **M**aster **T**hesis **C**ERN **P**roject.

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

-   Edit the config file [`trex-fitter/configs/pre-fit.config`](./trex-fitter/configs/pre-fit.config) - uncomment the
    `# PlotOptions: "YIELDS"` line. This will then show the number of events for each sample. I had it commented out for
    the thesis, where I didn't want the plot to be cluttered with the numbers.

-   `cd trex-fitter`
-   `trex-fitter n configs/pre-fit.config` - read the n-tuples
-   `trex-fitter w configs/pre-fit.config` - create the workspace
-   `trex-fitter d configs/pre-fit.config` - draw the yields plots
-   Now check the [outputs folder](./trex-fitter/outputs/pre-fit/Plots/) for the plots. Check if they make any sense or
    if there is a mistake.

### If all is okay, you can proceed to process the root n-tuples into the numpy arrays.

First, check which features are you using. I had a [bunch of different feature sets](./data_processing/features), where
I just basically merged them together into [this final set of features](./data_processing/features/merged.yaml).

Once this features file is ready, you can run the data processing with `python data_processing/main.py`. This will
create the `data_processing/output` folder with the processed dataset in the numpy format.

Now I recommend you transfer these outputs files from lxplus to somewhere where you plan to do the NN training.

### Training the NN

There's a [bunch of different configurations I tried](./ml/configs). You can follow the examples there and create your
own. To train a simple resnet, you should:

```bash
python ml/train.py resnets/resnet-6
```

This uses [wandb](https://wandb.ai/) for logging. So check how you are doing there.

Once the run is finished, some evaluations will be run automatically. However, if you want to make sure all is fine and
run manually, you can do:

```bash
python ml/evaluate.py resnets/resnet-6
```

Then you will have some outputs in the [`ml/outputs`](./ml/outputs) folder.

### Producing the friend trees

To evaluate the systematic uncertainties, we need to get the model prediction for all the events in the systematic
n-tuples (not just the `nominal`, but also `Sys1`, `Sys2`, ...). All the files weigh about 2.7 TB, so the approach we
took was:

1. Apply selection to only get events in the SR. This will produce the so-called small n-tuples.
2. Produce friend trees for just those small n-tuples.

For the first step, you would need to run the
[HTCondor](https://htcondor.readthedocs.io/en/latest/users-manual/submitting-a-job.html) job.

On the lxplus:

1. Edit the [`jobs/produce-small/run.sh`](./jobs/produce-small/run.sh) and adjust the path to the systematic n-tuples.
   Also adjust some paths (there is my user path, change to your user).
2. Submit the condor job: `bash ./jobs/produce-small/run.sh`.
3. [Monitor the job until it's done](https://htcondor.readthedocs.io/en/latest/users-manual/managing-a-job.html).

Once it's done, you should have a folder with the small n-tuples under the `friend_ntuples/output/small`

Now, you can copy it to the cluster with your main GPU.

On the cluster, you can now run the friend trees production:

(but first check the [config file](./friend_ntuples/config-friend.yaml) and adjust some values/paths there to suit your case)

```bash
python friend_ntuples/produce_friend.py
```

If that succeeded, you should have the friend trees in the `friend_ntuples/output/friend` folder.

Now, you can copy those back to lxplus to evaluate the systematic uncertainties.

### Evaluating the systematic uncertainties

Running on all the systematics and all backgrounds takes a lot of time, so it is a good idea to first comment some of
them out and try running on just a few:

```bash
cd trex-fitter
trex-fitter nwdfr configs/probs-partial-sys-partial-bg.config
```

If that went well, we would need to run on the full set of systematics and backgrounds. That takes a very long time and
on lxplus the process is just killed. So, we need to launch a condor job for that:

(remember to change the paths in the [`jobs/trex-sys/submit.sh`](./jobs/trex-sys/submit.sh) and [`jobs/trex-sys/job.sh`](./jobs/trex-sys/job.sh) files)

```bash
bash ./jobs/trex-sys/submit.sh
```

And that should be it.

# Good luck.
