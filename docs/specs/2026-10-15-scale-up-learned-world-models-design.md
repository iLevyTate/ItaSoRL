# Scale-up design: a published agent raised inside a learned world model

Date: 2026-10-15. Status: design only. Nothing here has been run, and no rule below is frozen yet.

## 1. The question

In ITASORL a linear probe reads the learned-dynamics fingerprint from the agent's state (0.733), and a nudge along the decoded direction moves behavior no more than a sham. The agent's own prediction error first appeared to separate the worlds (0.85 to 0.91), but on identical input a randomly initialized decoder separates them better (0.764 against 0.626), and agents raised in the authentic world are also more surprised by authentic episodes (artifacts/surprise/corrected_l3_h8_wm.json). That error tracks how predictable each stream is, not the law the agent was raised under. Question 1 below is built to avoid the same trap. Perceptual quantization (L1) was oracle-detectable and still not kept. This project asks the same three questions of a published agent trained inside a learned world model that can also act in the real environment:

1. **Detected.** Does the world model flag world-model transitions against real ones?
2. **Retained.** Does the policy's recurrent state keep the difference, under the ITASORL control battery?
3. **Used.** Does moving the state along the decoded direction change behavior more than a sham move?

Agents are now trained and tested inside learned simulators, so a policy-side simulator fingerprint bears on sim-to-real transfer and on evaluation awareness. Pixel world models also blur and smear texture, a perceptual tell separate from their dynamics error. Separating the two is the link to the L1-versus-L3 result.

## 2. Survey of candidate systems

Checked by web search on 2026-10-15. "Steppable" means a trained policy can be run closed loop inside the world model at evaluation time.

| System | License | Public world model + agent weights | Steppable in WM at eval | Training compute (as reported) |
|---|---|---|---|---|
| DIAMOND Atari (Alonso et al., 2024) | MIT | Yes: 26 Atari 100k games, one checkpoint each (WM, actor-critic, reward model), 1.41 GB | Yes, `src/play.py --pretrained` | 1 RTX 4090, about 2.9 days and 12 GB VRAM per game-seed |
| DIAMOND CS:GO branch | MIT | WM only (381M params) | Human only | 12 days on an RTX 4090 |
| IRIS (Micheli et al., 2023) | GPL-3.0 | Yes, Atari, on HF | Yes, `play.sh -a` | Multi-day per game (not re-verified) |
| DreamerV3 (Hafner et al., 2025) | MIT | No | In code; policy reads the RSSM latent | JAX, Linux/Mac, Python 3.11+ |
| STORM (Zhang et al., 2023) | None shown | No | In code | RTX 3090; repo unmaintained |
| TWISTER (Burchi and Timofte, 2025) | Apache-2.0 | No | In code | Not stated |
| GameNGen (Valevski et al., 2024) | No code | No | No | SD 1.4 on VizDoom |
| Genie 1/2/3 (DeepMind) | Closed | No; Genie 3 is a hosted 60 s product | No | Not public |
| Jafar, Jasmine (open Genie) | Open | No agent | No | Jafar: CoinRun, one L40S, under a week |
| Open-Oasis 500M | MIT tag | WM only, Minecraft | Human only | Inference code only |
| Matrix-Game 2.0 to 4.0 | 4.0 Apache-2.0 | WM only (2.0 is 1.8B) | Human only | Not stated |
| Dreamer 4 (Hafner et al., 2025) | No release | No | No | Minecraft, offline data |

Only DIAMOND and IRIS ship a per-game world model plus a policy trained inside it, with the real environment (ALE) freely available.

## 3. Feasibility on the workstation

Hardware: RTX 4050 Laptop, 6 GB VRAM, 16 GB RAM (often 2 to 4 GB free), Windows 11, Python 3.13 plus uv, Docker usually off.

| Candidate | Local readout | Local training | Cloud |
|---|---|---|---|
| DIAMOND Atari | Fits (4M-param WM, 13M total, 64x64 frames); separate Python 3.10 uv venv | No, needs about 12 GB | About 70 GPU-hours per extra game-seed |
| IRIS | Likely fits (pilot to confirm) | No | Same order or more |
| DreamerV3, TWISTER, STORM | No checkpoints; DreamerV3 also needs WSL2 or Docker | No | Full training per seed |
| Oasis, Matrix-Game | No agent | No | Out of scope |

4090-class cards on community clouds ran roughly $0.30 to $0.80 per hour (check at booking).

## 4. Recommendation

**DIAMOND on Atari, primary game Breakout, with the 26-game pretrained panel as the replication set.**

- It is the only system with public per-game world model and agent weights under a permissive license.
- Its actor-critic learns only from imagined rollouts (horizon 15, 3 Euler steps), so it is raised in the surrogate and scored in both worlds, the ITASORL headline design. It did act in the real game while collecting data, without learning from it there.
- The policy carries a 512-unit LSTM state, the analogue of the 96-unit GRU.
- Stages 0 to 2 run locally. Breakout has lawful ball physics, and OCAtari's vision mode extracts object positions from pixels for a dynamics oracle.

## 5. Experimental design

### 5.1 Episodes and arms

A start state is a real ALE state reached by the trained agent after a random number of steps in [20, 200], snapshotted with `cloneState`. From it, the last 4 real frames and actions condition every continuation. Each continuation runs T = 15 steps (the training horizon) and, as a secondary readout, T = 48.

| Arm | Dynamics | Pixels | How it is made |
|---|---|---|---|
| A, authentic | real | real | ALE continues from the snapshot, agent acting |
| S, raw surrogate | world model | world model | DIAMOND rolls out closed loop, agent acting |
| P, perceptual-only | real | world-model texture | Real next frame noised to level sigma_p, then denoised by the WM given the real context. Content stays real, texture becomes the model's |
| D, dynamics-only | world model | cleaned | S frames snapped pixel by pixel to the game's real palette (colors seen in real 64x64 frames) |

The pilot sets sigma_p so the single-frame detector reads P at least as high as S. P then carries the surrogate's perceptual tell with real dynamics, and D its dynamics without the perceptual tell. If palette snapping cannot bring the single-frame detector on D to 0.60 or lower, D is recorded as not constructible and the separation rests on S against matched P.

The primary readout uses independent start states (110 per class), as in ITASORL; the balanced readout (both classes from one start state, one CV group) is secondary.

### 5.2 Ceilings (detectability)

| Detector | Input | Role |
|---|---|---|
| Frame-statistics battery | per-frame channel means, 16-bin color histogram, Laplacian variance (blur), high-frequency spectral energy, off-palette pixel fraction | Perceptual detectability. The analogue of the L1 oracle |
| Single-frame CNN | shuffled single frames | Perceptual ceiling |
| Dynamics oracle | OCAtari vision-mode ball and paddle tracks; residual against a velocity and bounce law fit on real transitions | Privileged dynamics ceiling. The analogue of the L3 residual oracle |
| Frame-stack CNN | 4-frame stacks | Combined ceiling |

### 5.3 Question 1: world-model surprise

Per transition, the world model's EDM denoising loss on the next frame given the 4 previous frames and actions, averaged over 5 fixed noise levels with noise draws shared by all arms, summed per episode, AUROC against A. On S the model scores its own samples, which is self-recognition, so the informative numbers are on P and D. An independently trained world model (Stage 3) removes the self-scoring problem. Binding floor, learned from ITASORL: the same loss from a randomly initialized world model of the same architecture, on the same frames with the same noise draws. Raw surprise AUROC can come from how predictable each stream is rather than from detection, so only the margin over that floor counts, and the sign must flip for a model trained on real frames only.

### 5.4 Question 2: probe and control battery

- **Probe.** Features [mean, last] of the LSTM hidden and cell states (512 each), logistic regression at scikit-learn defaults, standardized in fold, grouped 5-fold CV by start state, AUROC; shuffled-label baseline.
- **L0 gate.** Two real pools from disjoint ALE seed bases, equivalent to 0.5 within plus or minus 0.05 by TOST. A second L0, two WM pools differing only in sampler seed, checks that sampling noise alone does not separate.
- **Positive control.** A within-world variable on real episodes (Breakout: ball x above its median, from OCAtari RAM) must read at least 0.75.
- **Leakage.** Reward sum and termination must sit within 0.1 of chance for P and D; length is fixed. The frame-statistics battery is reported for every arm as perceptual detectability, gated only in the D construction check.
- **Untrained floor.** The same CNN-LSTM at random init (5 seeds), on its own random-policy episodes and on replay of the trained agent's episodes. Replay is binding, because a random CNN will read blur.
- **Sensory echo.** Regress the agent's CNN features of the current and previous frame out of the state in fold, then probe the residual.

### 5.5 Question 3: causal test

In each episode at step 5, move (h, c) one standard deviation of the probe score along the decoded direction, toward the other class. Sham: a random direction of matched norm orthogonal to the probe weights. Outcome: total-variation distance between the nudged and un-nudged action distributions over the next 5 steps, plus window return. Report real minus sham with a t-based 90% interval.

### 5.6 Replication unit and decision rule (to freeze before Stage 2)

Games are the replication unit in Stage 2: one pretrained agent per game, and only games whose agent passes an engagement gate (real-env score above a random policy by the ITASORL 0.15 margin on normalized return). Breakout seeds are the unit in Stage 3. t-based 90% intervals throughout.

| Claim | Rule |
|---|---|
| Arm X is decodable (X in S, P, D) | mean probe AUROC at least 0.65 with lower bound at least 0.65; margin over untrained replay at least 0.05; L0, positive control and leakage pass |
| Dynamics fingerprint retained | D decodable |
| Perceptual tell detectable but not retained | P frame-detector AUROC at least 0.85 and P not decodable |
| Dissociation (headline) | both rows above, with P at least as perceptually detectable as S |
| Detected by the world model | WM-surprise AUROC on D at least 0.65 and at least 0.05 above the random-init world model on the same frames |
| Not used | real-minus-sham interval inside plus or minus 0.05 (TOST) |
| Used | real-minus-sham lower bound above 0.05 |

Any other pattern is recorded under its plain label, and a failed gate makes that arm uninformative, never a pass.

## 6. Staged plan

| Stage | Where | Work | Estimate |
|---|---|---|---|
| 0, setup | local | Python 3.10 uv venv outside the ITASORL tree, CUDA PyTorch, ALE ROMs, `play.py --pretrained` smoke test; weights off git | 1 to 2 h |
| 1, pilot | local | Breakout and Pong, 110 episodes per class per arm, disjoint seed base. Measure throughput and VRAM/RAM peaks, calibrate sigma_p, test D, check L0, positive control and T = 48 drift | 4 to 8 h |
| 2, registered panel | local | Freeze the rule, then all engaged games, arms A, S, P, D, floors, causal test | About 1.5M frames, 1 to 2 days (pilot rate replaces this) |
| 3, seeds, second WM | cloud | 5 DIAMOND seeds on Breakout plus one independent WM | About 420 GPU-hours, roughly $130 to $350 |
| 4, second family | local | Repeat Stage 2 on IRIS (discrete tokens) | Set by Stage 1 |

Pilot pass criteria: Stage 2 fits in under 3 days; peak RAM under 3 GB; D constructible; L0 inside 0.45 to 0.55; positive control at least 0.75. No pilot number enters a verdict. The OCAtari dynamics oracle covers Breakout and any panel game OCAtari supports; other games report the pixel ceilings only.

## 7. Risks and what would make the result uninteresting

| Risk | Mitigation |
|---|---|
| Blur makes S decodable by any network, untrained included | P and D arms; replay floor is binding |
| Palette snap leaves residual texture (D not constructible) | Fall back to S against matched P |
| Rollouts past 15 steps drift to noise | Primary horizon is 15 |
| One pretrained agent per game confounds game with agent | Per-game reporting; Stage 3 seeds |
| Self-scoring inflates Question 1 on S | Independent WM in Stage 3 |
| 2 to 4 GB free RAM | Small batches, stream to disk, RAM preflight |
| Python 3.10 pin, Atari ROM terms | Separate venv; user accepts ROM license |

Uninteresting outcomes: every arm decodable including the untrained replay floor, or no arm decodable including S.

Most novel outcome: D decodable while P, at equal or higher perceptual detectability, is not, with the world model detecting both and a null causal test. That reproduces the L3-kept, L1-dropped split in a published agent raised in a learned simulator. The reverse (P kept, D dropped) is also reportable: standard pixel agents would carry the simulator's texture rather than its physics, the opposite of the ITASORL agents.

## References

- Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A., Pearce, T., and Fleuret, F. (2024). Diffusion for world modeling: Visual details matter in Atari. NeurIPS. https://arxiv.org/abs/2405.12399, code https://github.com/eloialonso/diamond, weights https://huggingface.co/eloialonso/diamond
- Micheli, V., Alonso, E., and Fleuret, F. (2023). Transformers are sample-efficient world models. ICLR. https://arxiv.org/abs/2209.00588, code https://github.com/eloialonso/iris
- Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. (2025). Mastering diverse control tasks through world models. Nature. Code https://github.com/danijar/dreamerv3
- Zhang, W., Wang, G., Sun, J., Yuan, Y., and Huang, G. (2023). STORM: Efficient stochastic transformer based world models for reinforcement learning. NeurIPS. https://arxiv.org/abs/2310.09615, code https://github.com/weipu-zhang/STORM
- Burchi, M., and Timofte, R. (2025). Learning transformer-based world models with contrastive predictive coding. ICLR. https://arxiv.org/abs/2503.04416, code https://github.com/burchim/TWISTER
- Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. (2024). Diffusion models are real-time game engines. https://arxiv.org/abs/2408.14837
- Bruce, J., et al. (2024). Genie: Generative interactive environments. ICML. https://arxiv.org/abs/2402.15391
- Willi, T., Jackson, M., and Foerster, J. (2024). Jafar: An open-source Genie reimplementation in JAX. ICML workshop. https://github.com/FLAIROx/jafar
- Etched and Decart (2024). Open-Oasis. https://github.com/etched-ai/open-oasis
- Skywork AI (2025, 2026). Matrix-Game. https://github.com/SkyworkAI/Matrix-Game
- Hafner, D., Yan, W., and Lillicrap, T. (2025). Training agents inside of scalable world models. https://arxiv.org/abs/2509.24527
- Delfosse, Q., Blüml, J., Gregori, B., Sztwiertnia, S., and Kersting, K. (2023). OCAtari: Object-centric Atari 2600 reinforcement learning environments. https://arxiv.org/abs/2306.08649, code https://github.com/k4ntz/OC_Atari
