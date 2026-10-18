# Troubleshooting

Find the heading that matches the message. Commands are for **Ubuntu** unless marked **PowerShell**. After fixing something, run the failed command or stage once more, and when it still fails run `python3 experiments/rtx4070/rtx4070.py package` and send the archive with a short note of what you tried.

## Installing the dependencies

- `uv: command not found`: `source "$HOME/.local/bin/env"` or reopen Ubuntu; if missing, repeat the `curl ... astral.sh/uv/install.sh` line of README step 4.
- Download errors or timeouts during `install-deps`: run it again; `uv` resumes and caches. Behind a company proxy set `HTTPS_PROXY` first.
- `No space left on device`: free disk space (see "Disk" below), then re-run.
- `uv sync --locked` says the lock file is out of date: the project checkout was modified. Run `git status`; use a clean checkout of the commit you were given.
- After a manual `uv sync`, `bitsandbytes` is missing: run `python3 experiments/rtx4070/rtx4070.py install-deps` again.

## Downloads

- Ollama `pull` fails or hangs: check that `curl -s http://127.0.0.1:11434/api/version` answers, then re-run `prepare`; it resumes. Needs internet access to `registry.ollama.ai`.
- Hugging Face download fails: re-run `prepare`; it resumes. Needs access to `huggingface.co`. No token is needed for this model; do not paste one anywhere.
- Slow downloads on `/mnt/c`: the repository must live under `~`.
- Offline machine: do the downloads on a machine with internet, in an identical WSL2 Ubuntu. Copy `~/.ollama/models` and `~/.cache/huggingface` to the same locations, then run `preflight --phase post`.

## Ollama

- Preflight says "Ollama server reachable" failed: start it (`sudo systemctl start ollama`, or `nohup ollama serve > ~/ollama.log 2>&1 &`). Look at `~/ollama.log` or `journalctl -u ollama -n 50`.
- Port 11434 already used by Ollama for Windows: either quit the Windows one, or point the experiment at the one you want with `export SMOL_OLLAMA_URL=http://127.0.0.1:11434`. Models pulled into one are not visible to the other.
- "model not found": re-run `prepare` for the same `--profile`.
- Context window mismatch: the derived model `smol-orch:8k` (or `smol-orch-small:8k` in the fallback profile) must exist; re-run `prepare`. Do not edit it by hand.

## "could not verify that our Ollama model is unloaded" / a model stays on the GPU

The script unloads its own Ollama model between stages (an explicit `keep_alive: 0` request) and then asks Ollama (`/api/ps`) whether it is really gone. If that fails, the next GPU stage is blocked, and the message names the model. This is our own model, not another Windows program:

```bash
ollama ps                      # what is loaded
ollama stop smol-orch:8k       # use the name shown (smol-orch-small:8k in the fallback profile)
sudo systemctl restart ollama  # if it refuses to go (or: stop `ollama serve`, start it again)
```

Then run the blocked stage again. If `GET /api/ps` itself fails ("HTTP 404"), the installed Ollama is too old to report what is loaded: update it (`curl -fsSL https://ollama.com/install.sh | sh`).

## Both models do not fit together (stage D, "resource limit")

Stage D keeps the Ollama model and the Hugging Face model on the GPU at the same time and measures it. If Ollama's model spills to the CPU while the Hugging Face model is loaded, or CUDA runs out of memory, the stage fails with a resource-limit reason. The sizes in `models.json` are estimates; the measurement decides. Use the smaller set: `python3 experiments/rtx4070/rtx4070.py --profile fallback prepare`, then add `--profile fallback --new-run` to the commands. If the fallback also fails, send the archive. Nothing is moved to the CPU silently, and a result is never described as "both on the GPU" when that was not measured.

## Ollama uses the CPU

Stage B fails when less than 95 % of the model sits in video memory. Check:

```bash
ollama ps
```

The `PROCESSOR` column must say `100% GPU`. If it shows CPU:

1. Close GPU-heavy Windows programs (browser video, games) and check `nvidia-smi` shows 10 GB or more free.
2. `ollama stop smol-orch:8k`, then run stage B again.
3. Make sure `nvidia-smi` works inside Ubuntu (README step 3). Ollama in Ubuntu uses the same GPU path.
4. If it still falls back, use `--profile fallback` (smaller model) and say so when you send the results.

Do not accept a CPU run as a pass.

## CUDA errors

- `nvidia-smi: command not found` or "no devices were found" in Ubuntu: update the Windows NVIDIA driver (580 or newer), then in **PowerShell** run `wsl --shutdown` and reopen Ubuntu. Do not install a driver in Ubuntu.
- `/dev/dxg` missing: WSL2 is not in use (`wsl --list --verbose` in PowerShell must say VERSION 2) or the WSL kernel is old: `wsl --update` in PowerShell, then `wsl --shutdown`.
- "driver supports CUDA 12.x, needs 13.0" in preflight: update the Windows driver.
- `CUDA out of memory`: close other GPU programs; then re-run. If it repeats, use `--profile fallback` or send the archive.
- `bitsandbytes` import or kernel errors in stage C: this combination is untested. Send the archive, and run the fallback profile (`--profile fallback`), which does not use it.
- Stage C passes but D fails with out-of-memory: both models do not fit together (see "Both models do not fit together" above); send the archive.
- `install-deps` says the environment is inconsistent after installing `bitsandbytes`: run `uv sync --locked` to restore it, then use `--profile fallback`. `bitsandbytes` is installed with `--no-deps` so that it can never change the locked packages.

## Stage E is refused: "Windows host is unreachable from the shell sandbox"

Intended. Stage E lets a model run shell commands, and on a default WSL2 setup those commands could reach your Windows disks (`/mnt/c`) or start Windows programs (`cmd.exe`, `powershell.exe`) through WSL interop; the Linux sandbox has never been shown to stop either. The message names the route it found. Stage E is not part of the first round, so the normal answer is: do nothing. The only fix is to remove the route (README, "Stage E (NOT part of the first round)": edit `/etc/wsl.conf`, then `wsl --shutdown`) in a dedicated, disposable Ubuntu. **Edit that file, never replace it**: keep `[boot]` / `systemd=true`, or the Ollama service stops starting (next entry). Do not try to edit the check or the script to get past it. Stages A to D are not affected. If you do not want stage E, ignore this.

If the check fails with "the sandbox let a probe through", stop and send the archive to the maintainer at once; do not run stage E.

## Ollama does not start after changing WSL settings

If `sudo systemctl start ollama` fails with "System has not been booted with systemd", `/etc/wsl.conf` lost its `[boot]` section. Put `systemd=true` back under `[boot]` (`sudo nano /etc/wsl.conf`), run `wsl --shutdown` in PowerShell, reopen Ubuntu. Without systemd you can still start Ollama by hand: `nohup ollama serve > ~/ollama.log 2>&1 &`.

## Sandbox

## Sandbox

Stage E needs a shell backend it can hold a real process tree over. Under WSL2 that is the Linux-namespace sandbox: if the preflight check "Shell sandbox starts" fails:

- Make sure you are in WSL2 and Ubuntu is recent (`lsb_release -a`).
- Namespaces must be allowed. On Ubuntu 24.04 try `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` and re-run the preflight. This loosens Ubuntu's restriction on user namespaces for the whole distribution until the next restart, so do it only in a dedicated, disposable distribution; tell us if you needed it.
- If it still fails, stage E is simply not run. Send the archive.

## `.wslconfig`

WSL2 takes only part of your memory by default. Create or edit `C:\Users\<you>\.wslconfig`:

```
[wsl2]
memory=12GB
swap=8GB
```

Run **PowerShell**: `wsl --shutdown`, then reopen Ubuntu. Lower `memory` if Windows becomes sluggish. 16 GB in total is the comfortable minimum.

## Disk

Preflight needs 25 GB free (35 GB recommended). Check with `df -h ~`. Free space: `uv cache clean`, `rm -rf ~/.cache/huggingface` (then re-run `prepare`), or old `experiments/rtx4070/results/` folders you have already sent. To move WSL2 to another drive see Microsoft's `wsl --export` / `--import` documentation.

## Other

- A stage stays running forever: it cannot; each has a hard limit and is killed. Wait for it. The attempts it had finished are kept in `stage_X/attempts.jsonl` and go into the archive.
- Ctrl-C: press it **once** and wait up to a minute. The script stops the stage and everything it started, unloads the Ollama model, keeps what was recorded and exits with code 130. Run `package` afterwards.
- Computer sleeps or the Ubuntu window was closed mid-run: look for leftovers first, `pgrep -af stage_runner.py` (the stage; kill it with `pkill -f stage_runner.py`) and `ollama ps` (then `ollama stop <name>`), `nvidia-smi` should show the memory back. Then re-run the stage that was interrupted (`stage X`). Earlier results are kept.
- Permission errors in the repository: it is probably on `/mnt/c`. Clone under `~`.
- Anything else: run `package`, send the archive and the exact text of the error.
