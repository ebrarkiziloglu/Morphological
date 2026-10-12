# Results: L1 pretraining and German case learning

Sources: `Models_Evaluations.xlsx` (5-seed means, dat→acc only) and `tr_stripped_variants_analysis.html` (per-seed statistics recomputed from `eval_results/*/…_per_seed_by_category.csv`, both test directions). Status as of 12 Oct 2026.

**Setup.** Character-level LMs are pretrained for 5 epochs on an L1 (phase 1) and then trained on German NLLB (phase 2, evaluated at epochs 1/3/5). They are scored on minimal pairs:

- **dat→acc** (9,389 pairs): does the model prefer the correct dative over an accusative?
- **acc→dat** (9,189 pairs): does the model avoid over-using the dative?

Phase-2 configurations: **regular** (L1 + German), **reset** (L1 with embeddings reset before German) and **shuffled L1** (shuffled L1 corpus + German). Every cell is a mean over 5 seeds. Tests are Welch's t over the 5 per-seed accuracies, with Benjamini–Hochberg FDR inside each comparison family. "pp" means percentage points.

Model keys: DE = German only; EN, TR, ZH = English, Turkish, Chinese L1; Stripped Case / Postp / Nominal Infl / Nominal = Turkish with case suffixes / case suffixes + postpositions / all nominal inflection / all nominal morphology removed.

---

## 1. Any L1 pretraining helps German case learning, and the help is largest early

1. **Every L1 beats German-only in every configuration and epoch** (all gains q<0.03 at epoch 5).
   - At **epoch 1**, the gain over German-only is +3.1 to +5.0pp on dat→acc and +5.7 to +10.8pp on acc→dat. German-only starts at 77.5 / 70.2%.
   - At **epoch 5**, the gain shrinks to +1.3 to +1.9pp on dat→acc and +2.7 to +4.4pp on acc→dat. German-only reaches 83.8 / 80.9%.
   - The advantage is partly a head start that German-only training closes over time. Whether it disappears with more German data is not yet tested.
2. **The gain on acc→dat is about twice the gain on dat→acc.** L1 pretraining mostly helps the model *not over-use the dative*. acc→dat is also the test that separates L1s and variants best (see §4).
3. **The benefit is concentrated where German-only is weakest.** In the dat→acc category breakdown (regular, epoch 5):
   - The core (non-prepositional) dative-argument aggregate rises from **36.3% (DE)** to 44.6% (EN), 43.1% (TR) and 41.5% (ZH), a gain of +5 to +8pp.
   - The prepositional aggregate rises only from 86.8% to about 88%, a gain of about +1.4pp.
   - Within prepositional items, the largest gain is in NaN_prep (72.4 → 77%).

## 2. Core dative arguments are not learned by any model

4. **Every model is below chance on core dative arguments.** Even the best model (EN, regular, epoch 5) scores only 44.6% on the core aggregate. It scores 40.9% on `dative-case`, 56.3% on `no-case` and 41.2% on `other-case`. In the same models, prepositional datives sit at 83–98%. The models have learned preposition-governed dative, not dative as a verb-argument case.
5. **The accusative preference is already present before any German.** After phase 1, every L1 model scores about 19–36% on the core dative categories (TR regular: dative-case 30.9%, no-case 25.5%).
   - On the full dat→acc test, every phase-1 model sits at about 43–48%, slightly below chance.
   - On acc→dat it sits at about 46–54%.
   - Phase-2 training fixes the prepositional cases but leaves this core-argument bias largely in place.
6. **Difficulty ranking among prepositional categories (stable across models):** Instrumental (about 98%) > Mixed (about 95%) > Ablative (about 93%) > Locative (about 83%) > NaN_prep (about 72–77%). Locative is the hardest large category (n=4,660). Locative contexts plausibly map onto German two-way prepositions (in/an/auf + Dat/Acc), where case depends on meaning rather than on the preposition.

## 3. The L1's typology barely matters: English ≈ Turkish

7. **English and Turkish transfer equally well.** At epoch 5, EN − TR is:
   - dat→acc: +0.16 / −0.39 / −0.07pp (regular / reset / shuffled)
   - acc→dat: +0.31 / +0.23 / −0.32pp

   Three cells reach raw p<0.05, but in opposite directions, and none survives FDR (smallest q=0.078). A case-marking agglutinative L1 gives no measurable advantage over an analytic L1 for learning German case.
8. **English is numerically the best L1** in the regular configuration at every epoch on both tests: 85.66 / 85.21% at epoch 5 versus TR 85.50 / 84.90%. The lead is small and not significant against TR.
9. **Chinese transfers least.** It keeps 74% (dat→acc) and 66% (acc→dat) of Turkish's gain in the regular configuration, and only 30% / 47% with shuffled L1. It is the lowest L1 in every regular-configuration ranking.

## 4. Stripping Turkish nominal morphology costs little

10. **Every stripped variant keeps most of Turkish's benefit.** In the regular configuration at epoch 5 they keep 86–100% of TR's dat→acc gain and 86–90% of its acc→dat gain. Removing 22–38% of word forms and up to 18% of the characters costs only tenths of a point.
11. **The cost is small but consistent in direction.** TR is ahead of the variant in 33/36 TR-vs-variant cells on dat→acc and 34/36 on acc→dat. The cost is clearest on **acc→dat, regular, epoch 5**:

    | Variant | TR − variant | q |
    |---|---|---|
    | Postp | +0.58pp | 0.031 |
    | Nominal Infl | +0.57pp | 0.031 |
    | Nominal | +0.55pp | 0.044 |
    | Stripped Case | +0.42pp | 0.307 (n.s.) |

    - With shuffled L1 the gaps are similar (+0.43 to +0.51pp); only Nominal Infl survives FDR (q=0.044).
    - On dat→acc every TR − variant gap is ≤0.34pp and none survives FDR.
12. **Stripped Case shows an asymmetry.** On dat→acc it is identical to TR (85.50 vs 85.50, regular, epoch 5). On acc→dat it trails by 0.42pp. Removing case suffixes leaves the model's ability to pick the correct dative intact and slightly weakens its ability to avoid an incorrect one.
13. **There is no dose–response.** Removing more morphology does not remove more transfer:
    - Stripped Case (the lightest edit) costs nearly as much as Nominal (the heaviest). Stripped Case − Nominal is +0.13pp (p=0.56).
    - Dropping derivation on top of inflection changes nothing. Nominal − Nominal Infl is +0.03pp (p=0.85).
    - No variant-vs-variant contrast survives FDR in any configuration (smallest q at epoch 5 = 0.878).

    The data do not support "more Turkish case morphology → more German case transfer". At most about 0.3–0.6pp of the transfer on acc→dat can be tied to Turkish nominal morphology. The remaining 85–100% of Turkish's benefit does not depend on it.
14. **Without its nominal morphology, Turkish falls behind English on acc→dat.** In the regular configuration at epoch 5, EN beats:

    | Variant | EN − variant | q |
    |---|---|---|
    | Stripped Case | +0.73pp | 0.047 |
    | Nominal | +0.86pp | 0.003 |
    | Nominal Infl | +0.89pp | 0.002 |
    | Postp | +0.90pp | 0.002 |

    Over intact TR, EN leads by only +0.31pp (q=0.078). Turkish morphology is therefore not useless: it is what keeps Turkish level with English on this test. Under reset and shuffled L1, no EN-vs-variant gap on acc→dat survives FDR.

## 5. Where the transfer lives: embeddings vs. the rest of the network

15. **Most of the transfer survives an embedding reset.** With reset, TR still gains +1.32pp (dat→acc) and +2.68pp (acc→dat) over German-only at epoch 5. That is 76% and 66% of its regular-configuration gain. A large part of the L1 benefit is stored in the transformer layers, not in the character embeddings.
16. **Resetting embeddings costs Latin-script L1s more.** Regular − reset at epoch 5:

    | L1 | dat→acc | acc→dat |
    |---|---|---|
    | EN | +0.96pp (p<0.001) | +1.45pp (p<0.001) |
    | TR | +0.42pp (p=0.017) | +1.36pp (p=0.001) |
    | ZH | +0.11pp (n.s.) | +0.76pp (p=0.016) |

    Chinese characters barely overlap with German characters, so their trained embeddings are of little use to German anyway. For EN, and to a lesser degree TR, part of the benefit sits in shared Latin-character embeddings. English, whose orthography is closest to German, loses the most. This fits the reset cost being largest early (EN at epoch 1: +1.85 / +4.76pp).
17. **Under reset, the stripping penalty mostly disappears.**
    - On dat→acc at epoch 5, Stripped Case, Nominal Infl and Nominal are within 0.1pp of TR, and all four variants are TOST-equivalent to TR within ±0.5pp.
    - On acc→dat, Nominal is within 0.04pp; the others trail by 0.13–0.35pp, none significantly.
    - Variants also pay a smaller reset cost than TR on dat→acc (Nominal 0.26, Nominal Infl 0.19pp vs TR 0.42pp).

    Whatever stripping costs is carried mostly by the phase-1 character embeddings, plausibly the altered character and suffix distribution, rather than by deeper representations.
18. **Postp is the most consistently penalised variant.**
    - Under reset it trails TR by 0.17pp (dat→acc) and 0.35pp (acc→dat).
    - Its retained gain under reset is 87% on both tests, where the other variants keep 93–100%.
    - It is the only variant that also deletes whole words (255k postpositions). That removes word-level structure, not just suffix characters, so this deficit is less likely to be an embedding-only effect.

## 6. L1 word order matters little for Turkish but more for English and Chinese

19. **Shuffling the Turkish L1 corpus costs almost nothing.** Regular − shuffled at epoch 5:
    - TR and all four variants: −0.14 to +0.08pp on dat→acc (all n.s.) and +0.14 to +0.31pp on acc→dat (only TR reaches p<0.05: +0.26pp, p=0.035).
    - EN: +0.21pp on dat→acc (n.s.) but **+0.89pp on acc→dat (p<0.001)**.
    - ZH: **+0.75pp (p=0.003)** and **+0.91pp (p=0.009)**.

    Turkish's benefit does not depend on intact L1 sentence structure. English's acc→dat benefit does, partly. This fits English word order (SVO, prepositions) being closer to German than Turkish word order (SOV, postpositions). Chinese, without shared characters, needs intact structure most.
20. **Taken together (§3–§6), the L1 benefit is largely generic.** It comes from pretraining on any natural-language text: it survives shuffling (for Turkish), survives embedding reset, and is unaffected by L1 typology or by removing case morphology. Language-specific factors (shared script, word-order similarity, Turkish case morphology) each add only a few tenths of a point on top.

## 7. Phase-1 (zero-shot) scores do not predict transfer

21. **After phase 1, all models are near chance on German pairs** (dat→acc 43–48%, acc→dat 46–54%). No Turkish variant differs from intact Turkish (all q>0.8). The only notable gap is EN vs TR on acc→dat (47.0 vs 50.9%, p=0.012, q=0.06), and phase 2 erases it. Zero-shot scores cannot be used to rank L1s for transfer.

## 8. Reliability and caveats

22. **The effects are at the edge of statistical power.** Five seeds detect about 0.4–0.5pp per configuration, while most L1 and variant differences are 0.1–0.6pp. A non-significant cell means "not detected". TOST is the right tool for claims of equivalence, and it confirms several (for example all four variants ≈ TR on dat→acc under reset).
23. **Training-run variance is larger than seed variance suggests.** Retraining the Stripped Case phase-2 models moved its results by 0.2–0.4pp. In the 5 Oct analysis, Stripped Case looked better than Postp and Nominal on acc→dat (+0.39pp, q=0.011). After retraining that ordering vanished. Any claim that rests on a sub-0.5pp gap should be replicated with more seeds or a second training run.
24. **Data volume is confounded with morphology removal.** Stripped corpora have the same number of lines and optimizer steps but fewer characters (Nominal −18%, Nominal Infl −13.5%, Postp −9%, Stripped Case −7%). Some of each variant's small deficit could be less L1 signal rather than missing morphology. That would make the morphology-specific effect even smaller than measured.
25. **Results tend to converge with more German training.** Most L1 and variant gaps are largest at epoch 1 and shrink by epoch 5. They are also noisiest at epoch 1, so isolated epoch-1 effects should be read with caution.
26. **Spreadsheet notes.**
    - `Models_Evaluations.xlsx` contains only the dat→acc test. All acc→dat conclusions come from the per-seed CSVs.
    - The variant rows labelled "+ DE NLLB Shuffled" are the *shuffled-L1* configuration (phase-1 corpus shuffled, regular German). They are not a shuffled German corpus.
    - The spreadsheet means match the per-seed CSVs to two decimals (63/63 cells checked).

## 9. Suggested next steps

- Add seeds, or a second training run per condition, for the regular and shuffled-L1 configurations, to firm up the 0.3–0.6pp acc→dat effects.
- Add a volume-matched control: intact Turkish with 18% of characters randomly removed, or lines truncated, to separate morphology from data volume.
- Use per-item paired tests (McNemar) on the core dative categories, which are too small (134–254 items) for seed-level tests.
- Train longer in phase 2 to check whether the L1 advantage vanishes or plateaus.
- Investigate the below-chance core-dative behaviour (finding 4), which is the largest and least-solved effect in the data.
