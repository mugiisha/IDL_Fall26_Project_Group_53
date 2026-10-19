# Setup guide (Group 53)

Follow these steps once on your laptop. Everyone uses the same setup, so results are comparable.

## 1. Install Miniconda

Download from https://docs.conda.io/en/latest/miniconda.html and install with the default options.

- **Windows:** afterwards, open **Anaconda Prompt** from the Start menu.
- **Mac:** pick **Apple Silicon** (M1/M2/M3/M4) or **Intel**, then open **Terminal**.

Check it worked: `conda --version`

## 2. Create the project environment

```
conda create -n optswitch python=3.10 -y
conda activate optswitch
```

You should see `(optswitch)` at the start of the line.
**Run `conda activate optswitch` every time you open a new terminal.**

## 3. Get the code

```
git clone https://github.com/mugiisha/IDL_Fall26_Project_Group_53.git
cd IDL_Fall26_Project_Group_53
```

If you already cloned it, get the latest version instead: `git pull`

## 4. Install the packages

PyTorch first.

- **Mac:** `pip install torch`
- **Windows or Linux** (CPU version, much smaller):
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`

Then everything else:

```
pip install -r requirements.txt
```

## 5. Check everything works

```
python -c "import gymnasium as gym; e = gym.make('HalfCheetah-v4'); print('HalfCheetah OK', e.reset(seed=0)[0].shape)"
python -c "import gymnasium as gym; e = gym.make('Hopper-v4'); print('Hopper OK', e.reset(seed=0)[0].shape)"
python -c "import torch; print('torch', torch.__version__, 'Muon available:', hasattr(torch.optim, 'Muon'))"
```

Expected: `HalfCheetah OK (17,)`, `Hopper OK (11,)` and `Muon available: True`.

## 6. Smoke test (2 to 3 minutes)

- **Windows:** `python src\optswitch\ppo.py --env-id HalfCheetah-v4 --total-timesteps 50000 --seed 1`
- **Mac:** `python src/optswitch/ppo.py --env-id HalfCheetah-v4 --total-timesteps 50000 --seed 1`

It should print `global_step=...` lines and finish without errors.
The warnings about `wandb-entity`, `target-kl` and `settotalmass` are harmless.

## Running experiments

- One run of 1 million steps takes about 40 to 60 minutes on a laptop CPU.
- Open up to 3 terminal windows and run one experiment in each.
- Keep the laptop **plugged in and awake**. Closing a window, sleep or updates stop a run.
  On a Mac, put `caffeinate -i` in front of the command to stop it sleeping.
- Results are saved in the `runs/` folder. It is **not** uploaded to GitHub (too big);
  zip it and upload it to the shared Google Drive folder instead.
- Watch training live: `tensorboard --logdir runs`, then open http://localhost:6006

## Uploading code changes

Only add the files you changed. Never use `git add .` (it would try to upload `runs/`).

```
git add path/to/your/file.py
git commit -m "Short description of the change"
git push
```

The first push opens a browser window to sign in to GitHub.
You must be a collaborator on the repo (ask the repo owner to add you).
