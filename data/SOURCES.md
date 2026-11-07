# Data sources

Every file here is read by the library or an experiment script; nothing else is shipped.
Files are arranged by task.

## Indirect Object Identification (`ioi/`)

| File | Content |
|---|---|
| `eng_ioi_abba.json`, `eng_ioi_baba.json` | 10,000 prompt records each for the ABBA and BABA templates. Each record has `clean_prefix`, `patch_prefix` (the counterfactual prompt with all three names replaced), `clean_answer` (the indirect object), `patch_answer` (the subject name of the clean prompt) and `case`. |
| `wang_circuit.json` | The 26 attention heads of the circuit of Wang et al., each with the position at which the head was designated (`S2`, `S1+1`, `END`). |

Origin: Wang, Variengien, Conmy, Shlegeris and Steinhardt, *Interpretability in the Wild: a
Circuit for Indirect Object Identification in GPT-2 small*, ICLR 2023
(<https://arxiv.org/abs/2211.00593>). The prompts were generated with the IOI dataset code of
that paper (`ioi_dataset.py` of <https://github.com/redwoodresearch/Easy-Transformer>, MIT
License), in the copy distributed with Hanna et al.'s `eap-ig-faithfulness` repository. The
circuit follows Table 1 and Appendix of Wang et al. The records are shipped as generated; the
generation script is not part of this release.

## Subject-verb agreement (`sva/`)

| File | Content |
|---|---|
| `nounpp_train.json`, `nounpp_test.json` | JSON-lines files, one record per line: `clean_prefix`, `patch_prefix` (number of the subject noun flipped), `clean_answer`, `patch_answer` (the correct and incorrect verb), `case`. |

Origin: Marks, Rager, Michaud, Belinkov, Bau and Mueller, *Sparse Feature Circuits:
Discovering and Editing Interpretable Causal Graphs in Language Models*, ICLR 2025
(<https://arxiv.org/abs/2403.19647>), file `data/nounpp_{train,test}.json` of
<https://github.com/saprmarks/feature-circuits> (MIT License). Copied unchanged.

## Gender Bias (`gender_bias/`)

| File | Content |
|---|---|
| `gpt2.csv`, `pythia.csv` | Prompt pairs for the GPT-2 and Pythia tokenizers: `clean` (a profession), `corrupted` (the profession replaced by "man" or "woman"), the answer token ids for the clean and corrupted prompt, and `label`. |
| `qwen25.csv`, `llama31.csv` | The same construction for the Qwen2.5 and Llama-3.1 tokenizers, built with `mediation.tasks.gender_bias.build_gender_bias_dataframe` (seed 0). |
| `professions_male_stereo_filtered.json`, `professions_female_stereo_filtered.json` | Profession lists with the stereotype direction of Vig et al., used by the construction. |

Origin: the counterfactual is that of Vig, Gehrmann, Belinkov, Qian, Nevo, Singer and
Shieber, *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*,
NeurIPS 2020, in the form released by Hanna, Pezzelle and Belinkov, *Have Faith in
Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*, COLM 2024
(<https://arxiv.org/abs/2403.17806>), directory `data/gender-bias` of
<https://github.com/hannamw/eap-ig-faithfulness> at commit
`d42fea6aa2c7acca39007c9c94923770669cb9ae`. `gpt2.csv`, `pythia.csv` and the two profession
lists are byte-identical to that commit. The repository has no license file; the files are
re-hosted here with attribution to the authors, who released them openly with the paper.

## Greater-Than (`greater_than/`)

| File | Content |
|---|---|
| `gpt2.csv`, `pythia.csv` | 1,000 prompt pairs each for the GPT-2 and Pythia tokenizers: `clean` ("The {noun} lasted from the year XXYY to the year XX"), `corrupted` (YY replaced by 01), and `correct_idx` (YY). |

Origin: Hanna, Liu and Variengien, *How does GPT-2 compute greater-than?: Interpreting
mathematical abilities in a pre-trained language model*, NeurIPS 2023
(<https://arxiv.org/abs/2305.00586>), in the form released with Hanna et al. (2024):
directory `data/greater-than` of <https://github.com/hannamw/eap-ig-faithfulness> at commit
`d42fea6aa2c7acca39007c9c94923770669cb9ae`. Both files are byte-identical to that commit and
are re-hosted under the same terms as the Gender Bias files above.

## Checksums (SHA-256)

```
05f0479c0f5198838465a6cb540dc8e839dc02232f4e4f423a28c78a858fc6fe  greater_than/gpt2.csv
e11a4bba1a996597edc51ae24ea9be8b7c39207b35e4432340c10da90ef8bb52  greater_than/pythia.csv
09dfcbf32a464e314f7c42b571ff46fc937513cfcf338a89aab8f0dbf8b5c5a9  gender_bias/gpt2.csv
91bf186d9e9f7f1ee65f2e3648e575345460f23d01586410bdc199a18d035892  gender_bias/pythia.csv
78f61f712a10cc728784c4c7583db6b2c7935734291d6aa6e85def941e270982  gender_bias/professions_female_stereo_filtered.json
62067345d6ad50e88919cf13a9f3f83a331dbfdc38745c924b604fd6903626a5  gender_bias/professions_male_stereo_filtered.json
```
