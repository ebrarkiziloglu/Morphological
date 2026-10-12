## Learning curves:
1. In dat->acc:
    - in Regular config: **EN > TR > EN variants**. 
    - in Reset embeds. in epoch 3 and 5, all TR variants outperform English.
    - in Shuffled, in epoch 3 and 5, **TR > EN > TR variants**.
2. In acc->dat:
    - in Regular config: **EN > TR > EN variants**.
    - in Reset embeds. in epoch 3 and 5, EN > TR.
    - in Shuffled, **TR > EN > TR variants**.


3. There is a consistency in Regular and Shuffled versions in both directions.

4. Resetting embeddings hurt dative structure more than accusative structure for English.

## Comparison C:

1. Regular & Epoch 3: 
    - dat->acc shows clear ordering: **EN > TR > Stripped Case > Nominal > Nominal Infl > Postp**. Non significant.
    - acc->dat: **Nominal** is clearly the worst... **EN > TR > Stripped Case > Nominal Infl > Postp > Nominal**

    - **Conclusion**: Removing postpositions hurts the dative structure more. Removing nominal morphology hurts the accusative structure more, which makes sense ? :)

## Per-Category Performances:

The 571 core (non-prepositional) dative arguments benefit much more from Phase 1: every language beats the German-only model (36.32) by +3.2 to +8.3 points, 
with English gaining most (+8.3 regular, +7.9 shuffled) and intact Turkish +6.8.