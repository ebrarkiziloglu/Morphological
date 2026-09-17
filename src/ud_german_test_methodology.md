### Methodology 

* Load the UD German-GSD dataset
* Find every token with `Case=Dat` in all 3 splits.
* For **each Dative token** in each sentence, create a record with:
  * `gold_text`: original sentence
  * `corrupt_1`, `corrupt_2`: two corrupted variants where that specific Dative token is changed toward an Accusative-like form.
  * Metadata: `split`, `sent_id`, `dative_token_index`, `dative_form`, `dative_lemma`, `dative_upos`.

### How the corruption works:

* The script uses a small mapping `DAT_TO_ACC_FORM` for common German Dative → Accusative forms (articles, pronouns, possessives), e.g.:
   * `dem → den`, `der → die`, `einem → einen`, `mir → mich`, `ihm → ihn`, etc.
* For each Dative token:
   * `DET / PRON` (upos == "DET" or "PRON"): use the `DAT_TO_ACC_FORM` mapping (dem→den, einem→einen, mir→mich, etc.).
   * `corrupt_1`: if the token’s surface form (lowercased) is in the mapping `DAT_TO_ACC_FORM`, it is replaced with the mapped Accusative form (casing preserved); otherwise the token is left as-is.
   * `corrupt_2`: picks some other Accusative form from the mapping (different string), again preserving casing. If there is no alternative, it falls back to corrupt_1.
* Sentences are reconstructed from the UD `tokens` list with simple punctuation spacing rules (no space before .,!?;: etc.), so text is consistent.

`DAT_TO_ACC_FORM` can be extended for more coverage or later plug in a proper generator (e.g. SMOR) while keeping the rest of the pipeline intact.