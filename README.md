# Pipeliner

Manages my ttH research. Where I want to automate stuff, visualize different steps, jobs, artifacts, etc.

It communicates with the lxplus cluster via ssh and scp. Some operations should be run there, because:

- The whole size of the simulated dataset is 2.7 terabytes. It is not feasible to download it to my laptop,
  nor to Lambda cluster where we have GPUs.
- Thus, when we want to work with it (for example, with systematics), we need to run the code on lxplus.
- Sometimes we want to utilize the batch system (HTCondor) to run multiple jobs in parallel. We can do that
  on lxplus only as well.
- Maybe some other reasons...

Sometimes we want to run on Lambda cluster, because:

- It has powerful GPUs.

Also, I wanted to manage different stages, artifacts between them. I want to be able to restart the job, reproduce the
job, etc. I want to visualize the artifacts and have a simple web page front-end where I can easily interact with all of
it.

To synchronize it all, I built this pipeliner tool.
