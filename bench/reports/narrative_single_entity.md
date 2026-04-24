# Single-entity narrative: what each signal surfaces, where it ranks positives

Each dataset, one entity whose cooc-standalone NDCG is near the median.
For each signal acting as a complete recommender, the top-10 candidates
are shown with scores. Positives (items the entity actually interacts
with in the test window) are marked with `*`. For positives that
didn't make the top-10, we show their rank in the full retrieval
output (budget 500) — so you can see whether the signal FAILED to
surface the right item or just RANKED it poorly.

---

## ML-1M: entity 2896 (cooc-standalone NDCG on this entity: 0.214)

Owned (train) items: 186. The signals infer this user likes
action/thrillers with a side of prestige drama.

**Test positives (ideal top-5):**
1. Silence of the Lambs, The (593)
2. Independence Day (ID4) (780)
3. Skulls, The (3484)
4. Keeping the Faith (3536)
5. U-571 (3555)

### cooccurrence
```
    1. item=2858   55705.00  American Beauty (1999)
  * 2. item=593    53947.00  Silence of the Lambs, The (1991)
    3. item=1197   50285.00  Princess Bride, The (1987)
    4. item=527    47585.00  Schindler's List (1993)
    5. item=858    44730.00  Godfather, The (1972)
    6. item=1214   43512.00  Alien (1979)
    7. item=1265   43387.00  Groundhog Day (1993)
    8. item=1617   43322.00  L.A. Confidential (1997)
    9. item=1200   41870.00  Aliens (1986)
   10. item=2396   41868.00  Shakespeare in Love (1998)
   (positive 780  ranks #21 — just missed top-10)
   (positive 3484 NOT RETRIEVED — cooc graph doesn't reach this year-2000 film)
   (positive 3536 NOT RETRIEVED — same)
   (positive 3555 ranks #289 — in the pool but buried)
```
**Hits top-10: 1 of 5.** The hit is at #2, which is why the NDCG is positive.
Cooc surfaces prestige/classic films that co-occurred with the user's
history; the three newer (year-2000) positives aren't reachable because
the graph at training time had few edges to them.

### item_item_cosine
```
  * 1. item=780    0.9350   Independence Day (ID4) (1996)
    2. item=2353   0.8576   Enemy of the State (1998)
    3. item=349    0.8281   Clear and Present Danger (1994)
    4. item=2115   0.8162   Indiana Jones and the Temple of Doom
    5. item=1197   0.7901   Princess Bride, The (1987)
    6. item=2006   0.7748   Mask of Zorro, The (1998)
    7. item=1527   0.7670   Fifth Element, The (1997)
  * 8. item=593    0.7657   Silence of the Lambs, The (1991)
    9. item=2406   0.7562   Romancing the Stone (1984)
   10. item=2763   0.7456   Thomas Crown Affair, The (1999)
   (positive 3484 NOT RETRIEVED)
   (positive 3536 ranks #445 — deep in the pool)
   (positive 3555 ranks #138)
```
**Hits top-10: 2 of 5.** Surfaces Independence Day #1 (cooc had it at
#21), Silence of the Lambs #8. Cosine ranks action/thriller more
tightly than cooc does. But same coverage gap on the year-2000 films.

### als_factor
```
  * 1. item=780    0.8108   Independence Day (ID4) (1996)
  * 2. item=593    0.6460   Silence of the Lambs, The (1991)
    3. item=2706   0.6305   American Pie (1999)
    4. item=2683   0.5924   Austin Powers: The Spy Who Shagged Me
    5. item=588    0.5641   Aladdin (1992)
    6. item=2353   0.5589   Enemy of the State (1998)
    7. item=595    0.5179   Beauty and the Beast (1991)
    8. item=1552   0.5064   Con Air (1997)
    9. item=919    0.5059   Wizard of Oz, The (1939)
   10. item=2115   0.4734   Indiana Jones and the Temple of Doom
   (positive 3484 NOT RETRIEVED)
   (positive 3536 ranks #292)
   (positive 3555 ranks #21 — best-available rank)
```
**Hits top-10: 2 of 5, both in top-2.** ALS's top-10 is noisier than
cosine's (American Pie at #3 for this action-thriller fan is weird),
but it lands two positives at #1 and #2, and surfaces U-571 at #21
where cosine had it at #138.

### persona
```
    1. item=2706   0.2799   American Pie (1999)
    2. item=3793   0.2495   X-Men (2000)
    3. item=3408   0.2422   Erin Brockovich (2000)
    4. item=3114   0.2369   Toy Story 2 (1999)
  * 5. item=780    0.2345   Independence Day (ID4) (1996)
    6. item=316    0.2291   Stargate (1994)
  * 7. item=3555   0.2240   U-571 (2000)
    8. item=3175   0.2226   Galaxy Quest (1999)
    9. item=1356   0.2219   Star Trek: First Contact (1996)
   10. item=2683   0.2211   Austin Powers: The Spy Who Shagged Me
   (positive 593  ranks #285 — buried)
   (positive 3484 NOT RETRIEVED)
   (positive 3536 ranks #353 — deep)
```
**Hits top-10: 2 of 5.** The interesting case. Persona catches U-571
(the year-2000 action film NONE of the neighborhood signals put in
the top-10) at #7. Persona puts Silence of the Lambs at #285 — which
cooc/cosine had at #2/#8 respectively. Persona sees the user as
"generic 2000 mainstream" and ranks year-2000 popular films high;
it loses the 90s thriller thread that cooc/cosine capture precisely.

### path_basket — **0 hits in top-10**
Top picks: Eraser, Con Air, US Marshalls, Under Siege, Broken Arrow,
Peacemaker, Executive Decision, Jackal, Outbreak, Mercury Rising.
These are all 90s action thrillers — structurally similar to the user's
profile but none match the specific test positives. Path-basket is
correctly identifying the genre but not the specific films.
Positives ranked: 780 at #36, everything else not retrieved.

### path_tail — **0 hits in top-10**
Top picks: Galaxy Quest, General's Daughter, Go, Frequency, Gods and
Monsters. Essentially random among the user's recent trajectory.
Path-tail can't find the right film from just "what came after the
user's single most recent movie" — too little context.
All positives not retrieved.

### What the ML-1M case tells us

- **Retrieval ceiling**: 3484 (Skulls, The) isn't surfaced by ANY
  signal. That's the hard ceiling — no amount of blending helps. The
  year-2000 films have few training-time edges into the user's 1990s-
  dominated history. This is where HNSW-over-ALS or a learned
  embedding retriever would fill in the gap.
- **Retrieval differences**: each signal surfaces a DIFFERENT subset
  of positives in its top-10. No signal gets all 5; cosine+ALS+persona
  each find 2 of 5, but they're overlapping 2s (780 and 593) for
  cosine/ALS while persona picks 780 and 3555. **Union of cosine
  and persona top-10 would have 3 positives** — exactly the
  recall-via-different-retrievers story.
- **Ranking quality shows**: ALS lands positives at #1 and #2. Cosine
  at #1 and #8. Persona at #5 and #7. Same 2 out of 5 positives,
  very different ranks → very different NDCG.

---

## grocery-deep: entity 274 (cooc-standalone NDCG: 0.262)

Owned (train) items: 72. User has shopped multiple categories but
leans heavily on category 4 (bread-like) and category 1
(dairy/eggs-like) based on cooc's top picks.

**Test positives (ideal top-4):**
1. 38 (cat1_item13)
2. 39 (cat1_item14)
3. 153 (cat6_item3)
4. 154 (cat6_item4)

### cooccurrence
```
    1. item=109    25157  cat4_item9
    2. item=113    25125  cat4_item13
    3. item=120    24677  cat4_item20
    4. item=118    24625  cat4_item18
    5. item=104    24143  cat4_item4
  * 6. item=38     22673  cat1_item13
    7. item=33     21513  cat1_item8
  * 8. item=39     21446  cat1_item14
    9. item=37     21357  cat1_item12
   10. item=42     21171  cat1_item17
   (positive 153 ranks #52 — cat6 items buried)
   (positive 154 ranks #66 — same)
```
**Hits top-10: 2 of 4** (38 and 39 — both cat1 items). Cat4 items
dominate top-5 because this user has strong cat4 co-occurrences.
Cat6 positives (153, 154) are both ranked deep — the cat6 items in
this user's history are weak and don't pull cat6 neighbors high.

### path_tail — the surprise winner on this entity
```
    1. item=46     0.0296   cat1_item21
    2. item=33     0.0274   cat1_item8
    3. item=25     0.0252   cat1_item0
    4. item=42     0.0231   cat1_item17
  * 5. item=39     0.0229   cat1_item14
    6. item=27     0.0227   cat1_item2
    7. item=37     0.0210   cat1_item12
  * 8. item=38     0.0148   cat1_item13
    9. item=34     0.0107   cat1_item9
   10. item=180    0.0083   cat7_item5
   (positive 153 ranks #109)
   (positive 154 ranks #95)
```
**Hits top-10: 2 of 4.** The user's LAST item happens to be cat1, so
tail surfaces cat1 neighbors — both positives in cat1 land in top-10.
Standalone path_tail averages weaker (0.181 global) but on this
specific entity it hits the same top-2 positives as cooc, at better
ranks (#5, #8 vs #6, #8).

### path_basket
```
    1. item=104    0.0119   cat4_item4
    2. item=120    0.0114   cat4_item20
    3. item=109    0.0112   cat4_item9
    4. item=113    0.0110   cat4_item13
    5. item=118    0.0104   cat4_item18
    6. item=37     0.0096   cat1_item12
  * 7. item=39     0.0096   cat1_item14
    8. item=33     0.0094   cat1_item8
    9. item=27     0.0090   cat1_item2
   10. item=46     0.0090   cat1_item21
   (positive 38  ranks #11 — barely missed)
   (positive 153 ranks #55)
   (positive 154 ranks #50)
```
**Hits top-10: 1 of 4** (39). Same cat4-dominated head as cooc,
slightly different cat1 ordering. Would have been 2 of 4 with a
K=11 cutoff (positive 38 at rank 11).

### item_item_cosine
```
    1. item=109    1.0000   cat4_item9
  * 2. item=38     0.9997   cat1_item13
    3. item=113    0.9985   cat4_item13
    ...
   (positive 39  ranks #12)
   (positive 153 ranks #32)
   (positive 154 ranks #38)
```
**Hits top-10: 1 of 4.** Weird — cosine tightens 38 to #2 but loses
39 (which cooc had at #8) to #12. Cosine's L2 normalization sharpens
one winner but scatters the next.

### als_factor
```
    1. item=52     0.6361   cat2_item2
    2. item=118    0.5697   cat4_item18
    3. item=42     0.5665   cat1_item17
    ...
   (positive 38  ranks #16)
   (positive 39  ranks #39)
   (positive 154 ranks #11)
   (positive 153 ranks #41)
```
**Hits top-10: 0 of 4.** ALS's latent factors go to cat2 first (which
no one else prioritized), then cat4. 154 at #11 is the best near-miss
— factor space has 154 close but not quite in top-10.

### persona
```
    1. item=25     1.2697   cat1_item0
    2. item=46     1.2634   cat1_item21
    ...
  *10. item=39     1.2414   cat1_item14
   (positive 38  ranks #12)
   (positive 153 ranks #15)
   (positive 154 ranks #28)
```
**Hits top-10: 1 of 4** (39 at #10). Persona sees this user as "cat1
shopper", ranks cat1 items high — but packs the top-9 with popular
cat1 items before hitting the test-specific ones.

### What the grocery case tells us

- **Every signal nails the category** (cat1 / cat4 dominate most
  tops) but differs on WHICH items within that category.
- **path_tail wins ranking** on this entity because the last item
  IS in the right category. Last-item context works well when the
  user's last recent action predicts the next. If this same entity
  had ended their session on a cat4 item, path_tail would've gotten
  zero positives — inherent variance.
- **Union would help materially.** cooc gets {38, 39}; path_tail gets
  {38, 39}; cosine gets {38}; path_basket gets {39}; persona gets
  {39}; ALS gets none. Union of cooc + path_basket = {38, 39} only.
  Adding ALS wouldn't help (rec@topK = 0 here). **None of the
  signals surface positives 153 or 154** in the top-10 for this
  entity — they're cat6, and this user's cat6 signal is too weak.

---

## The narrative, one paragraph

Both entities show the same structural phenomenon: **every signal sees a
DIFFERENT slice of the user's taste and ranks items from that slice
well**, but no single slice covers the full test window. The ML-1M case
is cleaner: cosine sees the 1990s-thriller thread (780, 593), persona
sees the year-2000 action thread (780, 3555), cooc sees the prestige
axis (593 at #2 next to Godfather and Schindler's). The grocery case
shows that path_tail can beat globally-stronger retrievers for a
specific user whose recent activity happens to align with the test
window. This is exactly why union-retrieval + precise ranking is the
architecture the data supports.

No signal is "best." The composition is.
