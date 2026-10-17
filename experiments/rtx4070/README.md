# RTX 4070 experiment: step-by-step guide

You do not need an AI assistant, and you do not need to understand the code. Follow the steps in order, copy the commands exactly, and send back **one file** at the end.

**Status of this package: prepared, NOT yet validated on real hardware.** It was tested only with mocks on a different machine. If something does not behave as described, that is useful information, not your mistake: run `package` anyway and send the result (see `RESULTS.md`).

Commands are labelled by where you type them:

- **PowerShell**: the Windows PowerShell window (Start menu, type `PowerShell`).
- **Ubuntu**: the Ubuntu window (Start menu, type `Ubuntu`). Its prompt looks like `name@PC:~$`.

## What this does and what you need

It runs five bounded experiments (A to E) with two small language models on your RTX 4070 (12 GB), one served by **Ollama** and one loaded through **Hugging Face Transformers** with 4-bit (NF4) quantization, and records what happened.

**Stage E (a model running shell commands) is not part of this first round.** It is refused automatically on a normal WSL2 setup, and the run then ends with `E` marked FAIL: refused. That is intended and needs no action. Stages A to D need nothing special and give most of the answers. Ignore the last section about stage E unless the maintainer asks you to do it.

| Need | Amount |
|---|---|
| GPU | NVIDIA RTX 4070, 12 GB |
| Windows | Windows 11, up to date, with the **NVIDIA driver version 580 or newer** (the locked PyTorch needs CUDA 13.0) |
| Free disk (in WSL2) | at least 25 GB, 35 GB recommended |
| RAM | 16 GB recommended (6 GB is the hard minimum) |
| Internet | needed for steps 1 to 8 only (about 15 GB of downloads). The experiments themselves (step 10) work offline |
| Time | 1 to 2 hours, mostly downloading |

Close games, browsers with video, and other GPU-heavy programs before the experiments. They share the 12 GB of video memory. (The experiment unloads its own Ollama model between stages by itself; you never need to do that.)

## Step 1. Check the Windows NVIDIA driver (PowerShell)

```powershell
nvidia-smi
```

Look at the top right: `CUDA Version` must be **13.0 or higher**. If the command is missing or the version is lower, install the latest "Game Ready" or "Studio" driver for the RTX 4070 from nvidia.com, restart Windows, and try again. **Do not install any NVIDIA driver inside Ubuntu**; WSL2 uses the Windows driver. You also do not need the CUDA Toolkit.

## Step 2. Install WSL2 with Ubuntu (PowerShell, as Administrator)

Right-click PowerShell, choose "Run as administrator":

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows when asked. After the restart an Ubuntu window opens and asks you to create a Linux username and password (anything; remember the password, it is needed for `sudo`). If WSL is already installed, check that it is version 2:

```powershell
wsl --list --verbose
```

The `VERSION` column must say `2`. If it says `1`: `wsl --set-version Ubuntu-24.04 2`.

Optional but recommended (gives Ubuntu enough memory): create the file `C:\Users\<you>\.wslconfig` with

```
[wsl2]
memory=12GB
swap=8GB
```

then in PowerShell run `wsl --shutdown` and reopen Ubuntu. See `TROUBLESHOOTING.md`, ".wslconfig".

## Step 3. Check that Ubuntu sees the GPU (Ubuntu)

Open **Ubuntu** from the Start menu:

```bash
nvidia-smi
```

You should see the RTX 4070 table. If you get "command not found" or "no devices", go to `TROUBLESHOOTING.md`, "CUDA errors". Do not continue until this works.

## Step 4. Install the basic tools (Ubuntu)

```bash
sudo apt update
sudo apt install -y git curl zstd python3 python3-venv ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

`zstd` is required by the Ollama installer. If `uv --version` fails after reopening Ubuntu, run `source "$HOME/.local/bin/env"` again.

## Step 5. Get the project onto the Linux filesystem (Ubuntu)

Keep it under your Linux home (`~`), **not** under `/mnt/c`. Files on `/mnt/c` are very slow and break file permissions.

You will be given one of the two forms. Use the one you received.

**(a) A git address and a commit id**

```bash
cd ~
git clone <ADDRESS-YOU-WERE-GIVEN> smol-harness
cd smol-harness
git checkout <COMMIT-ID-YOU-WERE-GIVEN>
```

**(b) A file named `smol-harness.bundle` (works with no internet access to the repository)**

Copy the file into your Windows Downloads folder, then:

```bash
cd ~
git clone /mnt/c/Users/<your-windows-name>/Downloads/smol-harness.bundle smol-harness
cd smol-harness
git checkout <COMMIT-ID-YOU-WERE-GIVEN>
```

Check the revision matches what you were told:

```bash
git rev-parse HEAD
```

## Step 6. Install the Python environment (Ubuntu)

From now on all commands are run from `~/smol-harness`.

```bash
cd ~/smol-harness
python3 experiments/rtx4070/rtx4070.py preflight
```

This first **preflight** only reads things; it changes and downloads nothing. It prints `PASS`, `WARN`, `FAIL` or `UNVERIFIED` per check. Fix every `FAIL` (the line tells you what to do) and run it again, **with one exception**: a FAIL named "Windows host is unreachable from the shell sandbox" belongs to stage E only. It is expected on a normal WSL2 setup. Leave it alone, do not change any WSL setting because of it, and carry on (the script prints "Stages A to D can run"). A check shown as `UNVERIFIED ... failed first` was not checked because an earlier check failed; fix that earlier one first. A passing preflight is never printed while something it needs is broken.

Then:

```bash
python3 experiments/rtx4070/rtx4070.py install-deps
```

It asks for confirmation and downloads about 6 GB (PyTorch and NVIDIA libraries). It also installs one extra package, `bitsandbytes==0.50.2`, which is not in the project lock file.

**Never run `uv sync` yourself afterwards**: it removes `bitsandbytes`. If you do by accident, run `install-deps` again.

## Step 7. Install and start Ollama inside Ubuntu (Ubuntu)

Install Ollama inside Ubuntu, even if you also have Ollama for Windows:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start it. Try the service first:

```bash
sudo systemctl start ollama
```

If that says systemd is not running, start it in the background yourself:

```bash
nohup ollama serve > ~/ollama.log 2>&1 &
```

Check it is up:

```bash
curl -s http://127.0.0.1:11434/api/version
```

You should see a short JSON answer with a version. If Ollama for Windows is also running and uses the same port, quit it from the Windows system tray (or accept that the Windows one is used; the preflight shows which one answers). See `TROUBLESHOOTING.md`, "Ollama".

## Step 8. Download the models (Ubuntu)

```bash
python3 experiments/rtx4070/rtx4070.py prepare
```

It tells you the size first (about 7.8 GB: an Ollama 7B model of 4.7 GB and a Hugging Face 1.5B model of 3.1 GB) and asks for confirmation. It pulls the Ollama model, creates a tiny derived Ollama model with a fixed 8192-token context window, and downloads one pinned revision of the Hugging Face model. Downloads can be re-run if interrupted (they resume).

If you run short of memory or disk, use the smaller model set; it needs no `bitsandbytes`:

```bash
python3 experiments/rtx4070/rtx4070.py --profile fallback prepare
```

(and add `--profile fallback` in front of every later command; `install-deps` is still the same).

Then the second preflight, which checks the downloaded models, that Ollama really has the fixed context window, and the free video memory:

```bash
python3 experiments/rtx4070/rtx4070.py preflight --phase post
```

All `FAIL` items must be fixed before continuing, except the stage-E-only one described in step 6.

## Step 9. Before the experiments

- Close GPU-heavy programs. Check with `nvidia-smi`; 10 GB or more of free memory is wanted.
- Leave the machine alone while it runs (no gaming, no sleep). Plug in the power cable.
- You may be asked nothing during the run.

## Step 10. Run the experiments (Ubuntu)

All five stages in order, in one new results folder:

```bash
python3 experiments/rtx4070/rtx4070.py run
```

Each stage has a hard time limit and is killed (never retried forever) if it exceeds it; the attempts it had finished are kept. A stage whose prerequisite failed is **skipped**, not forced. Between stages the script unloads its own Ollama model from the GPU and checks with Ollama that it is really gone; if it cannot confirm that, the next stage is blocked and says why. Expect 20 to 60 minutes in total.

On a normal WSL2 setup the run ends with `E` marked **FAIL: refused**, with a reason that names the Windows drive or interop route it found. That is the safety gate working, not an error to fix.

**Do not close the window or press Ctrl-C casually while a stage runs.** If you must stop, press Ctrl-C **once** and wait (up to about a minute): the script stops the running stage and everything it started, unloads the Ollama model, keeps what was recorded and exits with code 130. Then run `package` to send what exists. If the window was closed or the computer went to sleep, see `TROUBLESHOOTING.md`, "Other", before starting again.

You can also run stages one at a time, which is the right thing to do after fixing a problem:

```bash
python3 experiments/rtx4070/rtx4070.py stage A
python3 experiments/rtx4070/rtx4070.py stage B
python3 experiments/rtx4070/rtx4070.py stage C
python3 experiments/rtx4070/rtx4070.py stage D
python3 experiments/rtx4070/rtx4070.py stage E
python3 experiments/rtx4070/rtx4070.py status
```

`stage X` reuses the newest results folder; add `--new-run` to start a fresh one. Models are loaded on the GPU only; if the GPU cannot be used a stage **fails** and says so. There is no silent switch to the CPU, and no unrestricted shell.

### What each stage proves

| Stage | Question | PASS means |
|---|---|---|
| A | Is the machine ready? | GPU, driver, CUDA, PyTorch, bitsandbytes, sandbox and workspace checks all passed, and the environment was recorded |
| B | Does the Ollama model run on the GPU and call tools? | Generation worked with the model fully in VRAM, the context window is as set, and tool-call mechanics worked through effGen |
| C | Does the Hugging Face model load on CUDA with NF4? | Weights are on `cuda`, the quantization really is 4-bit, generation works, and unloading frees the memory |
| D | Does the mixed-provider **mechanism** work? | The Ollama orchestrator really generated and really executed `delegate`; the Hugging Face worker really generated, on `cuda`; the worker's report came back to the orchestrator through effGen; and the GPU was **measured** with both models loaded and Ollama's model still on the GPU. Whether the models then **solved** the little task is recorded separately (`task_success`) and does not change PASS or FAIL. A final answer that merely says it delegated counts for nothing |
| E | Can the pair finish a tiny coding task in a throwaway repository? | The task's own tests pass afterwards, judged from the repository itself, not from what the model said. Refused unless the Windows host is unreachable |

PASS never means the models are good at coding; only that the mechanism worked. A FAIL in E can be a model-quality outcome and is still valuable data. If the two models do not fit in video memory together, stage D fails with a "resource limit" reason and suggests the smaller model set (`--profile fallback`, which needs `prepare --profile fallback` first).

## Step 11. Package and send back the results (Ubuntu)

```bash
python3 experiments/rtx4070/rtx4070.py package
```

Run this **even if stages failed**; failures are what we most need to see. It prints the path of ONE `.tar.gz` file, its size and checksum. To open its folder in Windows Explorer (with Windows interop on):

```bash
explorer.exe experiments/rtx4070/results
```

or, if interop is off, type `\\wsl$\Ubuntu-24.04\home\<your-linux-name>\smol-harness\experiments\rtx4070\results` into the Explorer address bar.

Send that single `.tar.gz` file back (email, cloud drive, chat). It contains logs, versions, GPU readings and a manifest with checksums. It never contains model files. Known credential formats (Hugging Face and GitHub tokens, `sk-` keys, bearer headers, passwords inside URLs, `key=value` secrets, private-key blocks) and your home path are removed automatically, and the finished archive is scanned once more. **That is pattern matching, not a guarantee**: text a model printed from a file (the shell can read anything your Linux user can) could contain a secret in a form no pattern knows. Before sending, look inside with `tar -tzf <file>` and open anything unexpected, or extract it and search it. What is inside is explained in `RESULTS.md`.

## Stage E (NOT part of the first round): isolation requirements

**Stage E is disabled on the default WSL2 setup and is not part of the initial handoff. Skip this whole section unless the maintainer asks you to do it.** Do the steps above first and send the results (step 11).

Stage E runs shell commands written by a language model. They run inside a Linux sandbox, but that sandbox was never checked against two things WSL2 adds, and a probe that fails to escape one route proves nothing about the others:

1. **Windows drives** (`/mnt/c` and friends): a writable view of your Windows disks.
2. **Windows interop**: from Ubuntu a command can start Windows programs (`cmd.exe`, `powershell.exe`, any `.exe`). Those run as normal Windows processes, **outside** every Linux protection.

So stage E is only enabled where **no route to the Windows host exists**, whatever the sandbox does. All of these must hold (the preflight checks the first three and fails otherwise):

- No Windows drive or host file system is mounted (no `drvfs` or `9p` mount except the GPU libraries).
- Windows interop is off, and no `.exe` is on the Linux `PATH`.
- The probes that try to write outside the workspace and to start `cmd.exe` / `powershell.exe` from inside the sandbox cannot.
- The Ubuntu distribution is **dedicated and disposable**: it holds none of your files, passwords, SSH keys, cloud credentials or browser profiles. The sandbox does **not** stop a command from *reading* files of the Linux user (a known, accepted limit), and what a command prints can end up in the results archive.
- The repository for the task is the throwaway one the script creates.

If you cannot guarantee that, skip stage E. Do not look for a way to switch the check off.

How to meet this on WSL2 (only if asked, and only after you have packaged and sent the results of stages A to D):

1. Look at the file first. It may not exist; on Ubuntu 24.04 it usually contains a `[boot]` section with `systemd=true`, which the Ollama service needs:

   ```bash
   cat /etc/wsl.conf
   ```

2. **Edit the file, never replace it.** Keep every line that is already there, above all `[boot]` / `systemd=true`. Open it with `sudo nano /etc/wsl.conf` and add the lines below at the end. If a section with the same name already exists, change or add the keys inside that section instead of creating a second one with the same name. Save with Ctrl-O and Enter, leave with Ctrl-X.

   ```
   [automount]
   enabled = false

   [interop]
   enabled = false
   appendWindowsPath = false
   ```

3. In **PowerShell** run `wsl --shutdown`, then reopen Ubuntu.
4. Check, in Ubuntu: `cat /etc/wsl.conf` still shows `systemd=true` (if you had it), `nvidia-smi` still works (these settings are expected to leave the GPU alone, but that is **unverified**; if the GPU disappears, undo the change and tell the maintainer), and Ollama is up (`curl -s http://127.0.0.1:11434/api/version`; if not, `sudo systemctl start ollama` or the `nohup ollama serve` line from step 7).
5. Run `python3 experiments/rtx4070/rtx4070.py preflight`. The check "Windows host is unreachable from the shell sandbox" should now PASS. If it still FAILs, read what it names and stop; do not run stage E.
6. Run `python3 experiments/rtx4070/rtx4070.py stage E` (add `--profile fallback` before `stage` if you used the fallback profile earlier). It reuses the newest results folder, in which stages A to D must have passed, then run `package` again.

Effects: `/mnt/c` and `explorer.exe` no longer work in Ubuntu, so open results through `\\wsl$\Ubuntu-24.04\home\<your-linux-name>\smol-harness\experiments\rtx4070\results` in the Windows Explorer address bar.

To undo: edit `/etc/wsl.conf` again and remove **only the lines you added** (or set the values back to `true`). Never delete the file if it contains `[boot]`. Then `wsl --shutdown` in PowerShell.

## What is unverified until this runs on your machine

Everything about real hardware. In particular:

- the WSL2 GPU path and the driver/CUDA 13.0 requirement of the locked PyTorch;
- that Ollama in WSL2 really unloads its model on `keep_alive: 0`, and what `/api/ps` reports there;
- whether both models are on the GPU together (stage D measures it; the figures in `models.json` are only estimates);
- the effect of the `/etc/wsl.conf` isolation settings (including on the GPU), and how effGen's sandbox treats mounted Windows drives (never observed; stage E is refused while any exist);
- whether `bitsandbytes` 0.50.2 has working CUDA 13 kernels for the RTX 4070 (if not, stage C fails and `--profile fallback` is the answer);
- how Ollama behaves in WSL2 (GPU use, what `/api/ps` reports, the effective context window);
- whether both models fit in 12 GB at the same time (the figures in `models.json` are estimates);
- how well the models perform at tool calling and coding.

If a stage fails, do not retry it endlessly: read `TROUBLESHOOTING.md`, fix at most the thing it names, run that stage again once, and then package whatever you have.

## For the maintainer (before sending this to a colleague)

1. Commit the package and the code state you want tested, then note the commit id (`git rev-parse HEAD`). The experiment records the commit and whether the tree was dirty.
2. Give your colleague either the repository address and the commit id, or a bundle: `git bundle create smol-harness.bundle --all`.
3. Make sure `uv.lock` is committed (the install is `uv sync --locked`).
4. Stage E is not part of the initial handoff: it is refused by design on a default WSL2 setup. Enable it only for a dedicated, disposable distribution prepared as described in its section (editing, never replacing, `/etc/wsl.conf`); do not weaken the `host_isolation` check.
5. When the archive returns, compare `manifest.json` (`git.commit`, `stages`, `excluded_from_archive`, `re_redacted_files`) with what you expected. Only then update `docs/implementation-plan.md` to mark anything as hardware-validated.
