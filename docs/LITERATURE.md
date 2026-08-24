# The informational literature layer

`src/faciometry/report/literature.py` names, for a measurement that was reported,
what the clinical literature associates with it, and cites the source. It is
off unless a caller asks for it. This document is the argument for both halves
of that sentence, and the table of what is in it.

## The line

There is a sentence this layer is allowed to write and a sentence it is not,
and they are about the same measurement.

**Allowed.** "The rhinoplasty literature measures the nasofrontal angle at the
radix when planning the dorsal profile (Metgudmath et al. 2023,
doi:10.1007/s12070-022-03363-z)."

That is a claim about what a body of published work contains. It is checkable:
a reader can open the paper and see whether the nasofrontal angle appears in
it. It says nothing about the face in the photograph.

**Forbidden.** "You should consider rhinoplasty." "This would be improved by a
filler." "Recommended: blepharoplasty."

Those are claims about a person. Each one asserts a goal state the face does
not currently occupy, and none of them has any measurement behind it, because
the software has no way to know what anybody wants their face to look like.
Faciometry has no ground truth for a goal state and does not invent one.

Three rules follow, and each is a test rather than an intention:

1. **No scalar aggregate.** The project has no single number standing for a
   face, and this section does not smuggle one in as a count of topics.
   `test_no_aggregate_score_in_the_rendered_section` checks the rendered text.
2. **Nothing second-person.** No "you", no "your face". The sentences are about
   the measurement. `contains_second_person` scans every string in the table at
   import, and the test asserts it over the whole table rather than a sample.
3. **Every emitted string passes the shared gate.** Not a private word list:
   `derm.findings.contains_advice`, the same function that guards every
   dermatological finding, plus `report.prose.PRESCRIPTIVE_TERMS`, the same
   tuple the written report is checked against. A phrase failing either raises
   on import.

### One word the disclaimer cannot use

`contains_advice` matches the substring `diagnos` and cannot tell a claim of
one from a denial of one. That is deliberate on the derm side — the docstring
there says the cost of a false positive is rewording a note and the cost of a
false negative is a measurement tool telling somebody what to do about their
face — and this layer inherits it rather than carving out an exception.

So `DISCLAIMER` says *"not a medical assessment"*, *"makes no clinical
determination about the person in the photograph"*, and *"not a medical
device"*. It means what a not-a-diagnosis statement means, in words the shared
gate permits, and `test_disclaimer_passes_the_same_checks` asserts it stays
that way. The alternative was an exempt string that no test could check, which
is the arrangement that lets one exempt string become four.

## Why it is off by default

The default is `enabled=False`, a keyword-only parameter with no environment
variable and no module-level switch, so nothing can turn it on except a caller
that names it.

The reason is the reader. Body dysmorphic disorder runs at **18.6 percent** in
aesthetic plastic surgery pools ([PMC11241264](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11241264/)),
against roughly 2 percent in the general population. Joseph et al. (2017) found
that surgeons — clinicians, in the room, with the patient in front of them —
identified **2 of 43** screen-positive patients. The screening instrument
caught 43; the professionals caught 2.

Software renders a page. It has no room, no patient in front of it, and no
screening instrument. If trained clinicians recognise 5 percent of the people
for whom a list of surgical topics beside a measurement of their own face is
the worst possible document, a report generator recognises none of them. The
layer is therefore silent unless somebody asks, and when it does speak it says
why it was silent.

That is also why the gate is *the flag*, not the content. Softening the wording
would not help: the risk here is not that a sentence is rude, it is that a
named procedure attached to a number reads as a suggestion however the grammar
is arranged.

## The three emission rules

**Only for a measurement that was actually reported.** A withheld measurement
has no number — the gate decided the value described the photograph rather than
the face. Attaching a surgical topic to it would put the finding back in the
document in prose after the gate removed it as a number. `literature_for` reads
`Measured.shown` and `Finding.reportable`, and
`test_nothing_is_emitted_for_a_withheld_measurement` runs that over every entry
in the table, with a positive control that the same entry does speak when the
same value is shown.

**Only when the value falls outside the published range.** Where the cited
source publishes a range and the value is inside it, nothing is emitted at all.
The reference-range sentence in `prose.py` has already said where the sample
fell, and repeating it under a heading of surgical topics adds a topic and no
information. Where the value is outside, the sentence is:

> This value sits outside the 115.0 to 130.0 deg range reported by Powell and
> Humphreys 1984, as recorded by Metgudmath et al. 2023, for this measurement.
> That range describes where a cited sample fell, and a value outside it is a
> position relative to that sample and nothing more.

Never "abnormal", never "deviates" — those are refused by the import audit in
any case, since `deviates` is in `PRESCRIPTIVE_TERMS`.

Where no range exists — `facial_thirds_ratio` is the case in the table — the
note carries no range sentence, because there is nothing to place the value
against.

**Every note carries the disclaimer.** `LiteratureNote.disclaimer` defaults to
`DISCLAIMER`, `sentences()` always includes it, and the constant is exported so
a renderer that shows one association shows the statement too.

## Wiring

```python
from faciometry.report.literature import literature_for, literature_text

notes = literature_for(
    report.measurements,       # Sequence[Measured]
    findings.findings,         # Sequence[Finding], optional
    enabled=args.literature,   # default False
)
print(literature_text(notes))  # "" when nothing was emitted
```

The CLI flag is `--literature`, `action="store_true"`, default `False`. The
function returns `()` when the flag is absent, so a caller that always calls it
prints nothing until somebody opts in.

## The table

Seventeen associations over fourteen distinct measurements and three
dermatological finding kinds, drawn from nineteen sources. Every source below
was retrieved and read; none was written from memory.

### Nose

| Key | Topic | Source |
| --- | --- | --- |
| `nasofrontal_angle` | radix and dorsal profile in rhinoplasty planning | Metgudmath, Belaldavar, Singh, Ramanan and Das 2023, *Indian J Otolaryngol Head Neck Surg* 75(Suppl 1), [doi:10.1007/s12070-022-03363-z](https://doi.org/10.1007/s12070-022-03363-z), [PMC9758669](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758669/) |
| `nasolabial_angle` | nasal tip rotation in septorhinoplasty | Khabir, Sezavar, Bohluli, Mesgarzadeh and Tavakoli 2020, *Maxillofac Plast Reconstr Surg* 42(1), [doi:10.1186/s40902-020-00261-8](https://doi.org/10.1186/s40902-020-00261-8), [PMC7280400](https://pmc.ncbi.nlm.nih.gov/articles/PMC7280400/), CC BY |
| `nasal_tip_projection_ratio` | nasal tip projection in rhinoplasty | Wiegmann, O'Neill, Taiberg, Sinno and Rohrich 2024, *Plast Reconstr Surg Glob Open* 12(11), [doi:10.1097/GOX.0000000000006330](https://doi.org/10.1097/GOX.0000000000006330), [PMC11581758](https://pmc.ncbi.nlm.nih.gov/articles/PMC11581758/); Goode, in Powell and Humphreys (eds) 1984, pages 15–39, ISBN 978-0-86577-117-8 |

Metgudmath et al. measured 3D CT reconstructions of 30 patients: mean
nasofrontal angle 116.69°, 110.43° in men and 126.08° in women. They record
Powell and Humphreys' range as **115–130°**, where the Faciometry catalogue
carries **115–135°** from the same book. Two transcriptions of one 1984 volume
disagree by five degrees, which is recorded in the entry's caveat rather than
resolved, because neither transcription can be checked without the book.

Wiegmann et al. is the reason the tip-projection entry exists at all: no group
in their comparison — Caucasian, Middle Eastern or African American — fell
inside the 0.55–0.60 band Goode reports, and every group measured higher. The
catalogue's reference range is Goode's, so on this measurement the literature
note and the reference range disagree with each other in a way the reader
should see. Their profiles were generated rather than photographed, which the
caveat says.

### Lips and lower face

| Key | Topic | Source |
| --- | --- | --- |
| `e_line_upper_lip`, `e_line_lower_lip` | lip position in orthodontic profile assessment | Ahuja, Ahuja, Verma, Arunima and Thosar 2024, *Cureus* 16(2), [doi:10.7759/cureus.55015](https://doi.org/10.7759/cureus.55015), [PMC10973926](https://pmc.ncbi.nlm.nih.gov/articles/PMC10973926/), CC BY |
| `lip_vermilion_ratio` | upper-to-lower vermilion proportion in lip augmentation | Hong, Choi, Yoon, Wan and Yi 2025, *Life* 15(2):315, [doi:10.3390/life15020315](https://doi.org/10.3390/life15020315), [PMC11856795](https://pmc.ncbi.nlm.nih.gov/articles/PMC11856795/), CC BY |
| `mentocervical_angle` | lower-third profile analysis in chin augmentation | Arroyo, Olivetti, Lima and Jurado 2016, *Braz J Otorhinolaryngol* 82(5), [doi:10.1016/j.bjorl.2015.09.009](https://doi.org/10.1016/j.bjorl.2015.09.009), [PMC9444627](https://pmc.ncbi.nlm.nih.gov/articles/PMC9444627/), CC BY |
| `submental_cervical_angle` | cervical contour in neck surgery | Arroyo et al. 2016 (above); Ellenbogen and Karlin 1980, *Plast Reconstr Surg* 66(6):826, [doi:10.1097/00006534-198012000-00003](https://doi.org/10.1097/00006534-198012000-00003) |

Ricketts placed the upper lip about 4 mm and the lower lip about 2 mm behind
the aesthetic plane in his Caucasian sample. Ahuja et al. measured 407 children
in the Bankura district of West Bengal against the same plane and found the
commonest category to be lips *ahead* of it, and state explicitly that this
does not follow Ricketts' inference on a Caucasian population.

The vermilion entry is the clearest case of the pattern this whole layer keeps
running into. Hong et al. record the quoted upper-to-lower proportion of about
1:1.6 as a Caucasian figure, and set beside it about 1:1.25 in Chinese women
and 1:1.11 to 1:1.25 in Korean women. The number in the textbook and the number
in the population are different numbers.

Arroyo et al. is doing double duty because it is one of the few open sources
that writes down a convention conflict instead of picking a side. It records
**two definitions of the mentocervical angle**: Lehmann's, from the nasal tip
to pogonion across the submental line, normal 110–120°, and Powell and
Humphreys', from glabella to pogonion, which is what Faciometry computes and what
the catalogue's 80–95° belongs to. A value read under one definition cannot be
placed against the range published for the other. The same paper records the
cervicomental angle at 121° in men and 126° in women, which is why it, not
Ellenbogen and Karlin, supplies the numbers in the neck entry.

### Jaw and midface

| Key | Topic | Source |
| --- | --- | --- |
| `gonial_angle_l`, `gonial_angle_r` | mandibular divergence pattern in orthodontic and orthognathic assessment | Tashkandi, Alnaqa, Al-Saif and Allam 2024, *J Multidiscip Healthc* 17, [doi:10.2147/JMDH.S463688](https://doi.org/10.2147/JMDH.S463688), [PMC11070157](https://pmc.ncbi.nlm.nih.gov/articles/PMC11070157/) |
| `bizygomatic_width` | zygomatic contour in malar augmentation | Kauke-Navarro, Knoedler, Klimitz, Diatta, Kong, Bigus, Alperovich, Lellouch and Safi 2026, *Plast Reconstr Surg Glob Open* 14(4), [doi:10.1097/GOX.0000000000007554](https://doi.org/10.1097/GOX.0000000000007554), [PMC13061527](https://pmc.ncbi.nlm.nih.gov/articles/PMC13061527/) |

Tashkandi et al. measured 448 adults on both panoramic radiographs and lateral
cephalograms and found the two disagree. Both read bone. A photograph reads
soft tissue over gonion, which is a self-occluding landmark, so the photographic
quantity is not the skeletal one the orthodontic literature is about — and that
is in the entry's caveat, not in a footnote somewhere else.

The malar entry is attached to `bizygomatic_width` because it is the nearest
thing in the catalogue. It is not the same quantity: Kauke-Navarro et al. pooled
15 studies and 796 patients on implants placed over the **anterolateral malar
eminence**, which is an anteroposterior projection, and bizygomatic width is a
transverse breadth. Both entries say so. The review's own finding is that no
standardised planning or placement method exists across those 15 studies, with
several series relying on visual inspection or surface markings.

In practice this entry will almost never fire, because `bizygomatic_width` is
`REQUIRES_3D` and a 2D photograph withholds it: Lim et al. (2022, n=96) measured
bizygomatic breadth against calipers at a 3.3 mm mean difference with limits of
agreement from −7.5 to 14.2 mm. That is the intended behaviour of rule one, and
it is worth seeing an entry that exists and stays quiet.

### Eyes and proportions

| Key | Topic | Source |
| --- | --- | --- |
| `canthal_tilt_l`, `canthal_tilt_r` | canthal position in periorbital surgical planning | Lee, Choung and Choung 2020, *J Korean Assoc Oral Maxillofac Surg* 46(6):379, [doi:10.5125/jkaoms.2020.46.6.379](https://doi.org/10.5125/jkaoms.2020.46.6.379), [PMC7783177](https://pmc.ncbi.nlm.nih.gov/articles/PMC7783177/) |
| `facial_thirds_ratio` | the vertical facial-thirds canon | Jayaratne, Deutsch, McGrath and Zwahlen 2012, *PLoS ONE* 7(12):e52593, [doi:10.1371/journal.pone.0052593](https://doi.org/10.1371/journal.pone.0052593), [PMC3532441](https://pmc.ncbi.nlm.nih.gov/articles/PMC3532441/), CC BY; Farkas, Hreczko, Kolar and Munro 1985, *Plast Reconstr Surg* 75(3):328, [doi:10.1097/00006534-198503000-00005](https://doi.org/10.1097/00006534-198503000-00005) |

Lee et al. measured pupil level, medial and lateral canthal level and canthal
tilt in 76 Korean patients to quantify vertical orbital dystopia before surgical
correction, which is the setting in which canthal tilt is actually recorded.

The facial-thirds entry is the one whose literature says the measurement does
not work. Farkas et al. tested nine neoclassical formulas on 153 young adults
and found the vertical profile proportions the poorest performers; Jayaratne et
al. tested the three-section canon on 3D scans of 103 southern Chinese adults
and found it in **none** of them — their table reads `0 0 0 0`, and the text
says it "could not be found even in a single participant". The catalogue
publishes no reference range for this ratio, so the note emits at any value and
carries no range sentence.

### Dermatology

| Key | Topic | Source |
| --- | --- | --- |
| `acne_severity` | the Hayashi inflammatory-lesion grade | Hayashi, Akamatsu, Kawashima and the Acne Study Group 2008, *J Dermatol* 35(5), [doi:10.1111/j.1346-8138.2008.00462.x](https://doi.org/10.1111/j.1346-8138.2008.00462.x); Hayashi, Suh, Akamatsu, Kawashima and the Acne Study Group 2008, [doi:10.1111/j.1346-8138.2008.00463.x](https://doi.org/10.1111/j.1346-8138.2008.00463.x) |
| `erythema` | colorimetric quantification of cutaneous erythema | Aoki, Wong, Bartos and Mayrovitz 2026, *Cureus* 18(4), [doi:10.7759/cureus.107251](https://doi.org/10.7759/cureus.107251), [PMC13180473](https://pmc.ncbi.nlm.nih.gov/articles/PMC13180473/), CC BY |
| `periorbital_pigmentation` | the described types of periorbital hyperpigmentation | Sarkar and Das 2018, *Indian Dermatol Online J* 9(4), [doi:10.4103/idoj.idoj_303_17](https://doi.org/10.4103/idoj.idoj_303_17), [PMC6042190](https://pmc.ncbi.nlm.nih.gov/articles/PMC6042190/); Sarkar, Ranjan, Garg, Garg, Sonthalia and Bansal 2016, *J Clin Aesthet Dermatol* 9(1), [PMC4756872](https://pmc.ncbi.nlm.nih.gov/articles/PMC4756872/) |

Aoki et al. is the strongest fit in the table, because it validates the method
Faciometry already uses rather than merely naming a topic. They measured CIELAB at
an unaffected site and a rash site in each of 44 patients: **a\* 7.5 ± 3.9 at
baseline against 13.1 ± 4.6 at the rash site**, paired within subject. Baseline
a\* itself differed significantly across Fitzpatrick phototype groups
(F = 6.196, p < 0.001) and across ITA categories, which is the published reason
for reading erythema as a within-face paired contrast rather than as an absolute
colour — exactly what `derm/colorimetry.py` does. Their measurements came from
contact colorimeters under controlled probe pressure, not from an uncalibrated
photograph, and the paired sites were inner bicep and rash rather than two
regions of one face.

Sarkar and Das cite Huang's four types of periorbital hyperpigmentation —
pigmented, vascular, **structural**, and mixed — where the structural type is
shadow from tear-trough depression and skin laxity rather than pigment. That is
the published justification for `from_pigmentation` reporting the L\* and b\*
components separately instead of pooling them: an orbital-rim shadow is an L\*
deficit with no b\* shift. The types themselves are separated by dermoscopy,
Wood's lamp and histology, so a photograph does not distinguish them, and
Faciometry reports the contrast rather than a type.

Hayashi et al. established the grade boundaries by having dermatologists
classify global severity without a standard and then counting eruptions, fixing
the boundaries on the inflammatory lesion count of **one half face** — which is
why `from_acne_severity` carries a note when no facial midline was supplied. The
criteria were built on Japanese patients and tested among Japanese and Korean
dermatologists.

## What was left out, and why

Entries were dropped rather than written when the claim could not be checked.
An invented DOI in a health-adjacent tool is worse than a missing row.

- **Ellenbogen's numeric criterion.** The cervicomental angle is widely quoted
  at 105–120° on Ellenbogen and Karlin's authority. The 1980 paper is paywalled
  and its abstract states no numbers, so the range is not asserted here. The
  paper is cited for having set out the visual criteria, which a 2025 review in
  *Aesthetic Surgery Journal* ([doi:10.1093/asj/sjaf013](https://doi.org/10.1093/asj/sjaf013))
  confirms in its own abstract, and the numbers in the entry come from Arroyo
  et al. instead.
- **Malar prominence as its own measurement.** The catalogue has no
  anteroposterior malar projection, which is what the implant literature
  actually targets. The association is attached to `bizygomatic_width` with the
  mismatch stated, rather than a measurement being invented to hold it.
- **Per-third entries for `upper_face_height`, `middle_third_height` and
  `lower_third_height`.** The three-section canon is a claim about the ratio of
  the three, not a bound on any one of them. Repeating one source against three
  quantities it does not individually constrain would have inflated the count
  and said nothing.
- **`skin_tone` and `acne_lesion_count`.** No entry, deliberately. A Monk-scale
  swatch has no clinical association that could be stated without implying one,
  and the per-region lesion count is already covered by the severity entry.
- **A DOI for Powell and Humphreys 1984.** The book has none. It is cited by
  ISBN 978-0-86577-117-8, verified against Open Library and a university
  catalogue. A first draft of this table carried a guessed ISBN, which was wrong
  and was corrected before merge; it is recorded here because the guess is the
  exact failure mode this section exists to prevent.

## Tests

`tests/unit/test_literature.py`, iterating the whole table rather than a sample:

- no prescriptive term, no advice phrase and no second-person pronoun in any
  string, checked three ways — through `Association.strings()`, through the raw
  JSON walked node by node so a newly added field cannot arrive unchecked, and
  through every sentence the renderer generates;
- every citation carries a resolvable DOI, PMC id or ISBN, matched against a
  format regex; a citation without one raises at construction;
- nothing emitted for a withheld measurement or a withheld finding, over every
  entry, each with a positive control that the same entry speaks when shown;
- nothing emitted with the flag absent, with the flag `False`, and a `TypeError`
  if a caller tries to pass it positionally;
- silence inside every published range, and a neutral sentence outside it.
