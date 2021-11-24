This document is a step by step guide on how to access the GPUs on the PACE CoC-ICE (College of Computing Instructional Cluster Environment).

**STEP 1:**
- Make sure you are connected to either eduroam or Georgia Tech VPN (link for VPN : https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0026837).<br/>

**STEP 2:**
- ssh into the head node of the cluster using the command : ssh \<username\>@coc-ice.pace.gatech.edu
- Enter the password to your Georgia Tech account when prompted
- Each user is allocated a storage quota of 15GB (You can request for extra storage based on need). Check your storage quota using pace-quota.<br/>

**STEP 3:**
- Once you're in your head node, you can copy data to/from coc-ice using scp or rsync, check out the various modules that are pre-installed in the server, etc.
- Below are the commands to control the pre-installed modules in the server:
  - module spider: Lists all software and its available versions on cluster
  - module avail: Lists all available modules that can be loaded with current environment
  - module list: Displays all the modules that are currently loaded
  - module load <module name>: Loads a module to the environment
  - module rm <module name>: Removes a module from the environment
  - module purge: Removes all loaded modules
- Once you load the anaconda module, you can create a conda environment. For our project the following libraries need to be installed in the environment:
  - Pypianoroll
  - Progress
  - Music21
  - Ipdb

**STEP 4:**
- COC-ICE uses the moab scheduler
- For the purpose of this project, you will be using the BATCH mode of operation
- The following are the available job queues. You need to choose a job queue based on your requirement when scheduling the job.

**COC-ICE     Max CPU per Job    Max walltime    Note**<br/>	
coc-ice.       28                 2:00:00        Higher priority<br/>	
coc-ice-gpu    28                 2:00:00        For GPU jobs, higher priority<br/>	
coc-ice-multi  128                0:30:00        For MPI jobs, lower priority<br/>	
coc-ice-long   28                 8:00:00        Lower priority<br/>	
coc-ice-devel  128                8:00:00        Limited access, lowest priority<br/>	
coc-ice-grade  128                12:00:00       Instructors/TAs only, highest priority<br/>

- For our project, you will be using coc-ice-gpu<br/>
- Creating a new job :<br/>

  - Everything needs to be scripted. No user interaction once the job starts (things like press 'y' to continue are not allowed)
  - You need to write a .pbs script that contains resource requirements, environmental settings, and tasks
  - Once the .pbs file is ready submit it using **qsub <your_pbs_script.pbs>**
  - Error and output logs are printed to the files mentioned in your pbs script
  - Below is the pbs file that can be used to run our project on the GPUs :

[trainingrun.txt](https://github.com/alimirzazadeh/wolfGANg/files/7591964/trainingrun.txt)

**STEP 5:**
- Monitoring your jobs:
  - To monitor your jobs use : **qstat –u <username> -n**
  - To cancel a submitted job use qdel <job_id>
  - To summarize the utilization of each queue use pace-queue-check <queue_name>
  
Now all that's left is to enjoy the super fast training!

