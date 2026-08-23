# References

Every empirical claim in this codebase traces to something here. Where a source
is paywalled or non-redistributable, that is stated, because it determines
whether its numbers can be shipped or only cited.

## The measurement-validity case

These are the papers the architecture is built on. Read the first two if you
read nothing else.

**Kleinberg, K.F. and Vanezis, P. (2007).** "Variation in proportion indices and
angles between selected facial landmarks with rotation in the Frankfort plane."
*Medicine, Science and the Law* 47(2):107-116.
doi:[10.1258/rsmmsl.47.2.107](https://doi.org/10.1258/rsmmsl.47.2.107)
Full data tables in the open thesis:
<https://theses.gla.ac.uk/245/1/2008kleinbergphd.pdf>
Subjects photographed in ten-degree steps. At ten degrees of yaw the indices
shifted by 8 to 19 percent, against a between-subject relative standard
deviation at zero degrees of 1.2 percent for the tightest index. This is the
result that makes discriminability rather than accuracy the primary gate.

**FISWG (2026).** Facial Identification Scientific Working Group, V2.1,
section 6.4.1. Prohibits photo-anthropometry for identification, citing
Kleinberg 2007, Evison 2010 and Moreton and Morley 2011.

**Lim, A.Y., Abdul Shakor, A.S. and Shaharudin, M.R. (2022).** "Reliability and
Accuracy of 2D Photogrammetry: A Comparison With Direct Measurement."
*Frontiers in Public Health* 9:813058.
[PMC8826070](https://pmc.ncbi.nlm.nih.gov/articles/PMC8826070/)
n=96, ten dimensions, two raters, Bland-Altman against calipers. Seven of ten
dimensions agree to 0.3-1.0 mm. **Bigonial breadth: mean difference 9.3 mm,
limits of agreement -0.9 to 19.6 mm. Bizygomatic breadth: 3.3 mm, limits -7.5
to 14.2 mm.** The pattern is mechanistic: measurements whose endpoints lie on a
laterally curved, self-occluding surface fail; midline sagittal ones do not.

**Kramer, R.S.S. (2016).** "Within-person variability in men's facial
width-to-height ratio." *PeerJ* 4:e1801.
doi:[10.7717/peerj.1801](https://doi.org/10.7717/peerj.1801)
Variance decomposition of fWHR: posed expression eta-squared 0.58 against
identity 0.31, and the rank ordering of individuals changes with which
photograph is used. No projection model predicts this, which is why
`measured_within_person_rsd` overrides the derived sensitivity.

**Kramer, R.S.S., Jones, A.L. and Ward, R. (2012).** "A Lack of Sexual
Dimorphism in Width-to-Height Ratio in White European Faces Using 2D
Photographs, 3D Scans, and Anthropometry." *PLoS ONE* 7(8):e42705.
[PMC3413652](https://pmc.ncbi.nlm.nih.gov/articles/PMC3413652/) (CC BY)
The same 66 men measured at 2.01 from photographs and 1.83 from 3D scans, a gap
larger than any published sex difference in that measurement.

**Vaca, E.E. et al. (2022).** *Aesthetic Surgery Journal Open Forum*.
[PMC8830303](https://pmc.ncbi.nlm.nih.gov/articles/PMC8830303/)
Frankfort plane swept from -15 to +15 degrees. Intercanthal height falls from
4.39 mm to 0.128 mm, which is about 8.3 degrees of apparent canthal tilt to
0.2 -- roughly 0.27 degrees of tilt per degree of pitch.

**Gibelli, D. et al. (2021).** "Does Head Orientation Influence 3D Facial
Imaging?" *IJERPH* 18(8):4276.
doi:[10.3390/ijerph18084276](https://doi.org/10.3390/ijerph18084276)
3D precision is essentially unaffected by pose; the paper's explicit conclusion
about 2D is that "even minimal errors in yaw, roll and pitch of the head can be
cause of unreliability of 2D photography itself."

**Bland, J.M. and Altman, D.G. (1986).** "Statistical methods for assessing
agreement between two methods of clinical measurement." *Lancet*
327(8476):307-310. Correlation between two methods says nothing about bias or
scatter. Report mean difference with limits of agreement.

## Perspective and pose

**Ward, B., Ward, M., Fried, O. and Paskhover, B. (2018).** "Nasal Distortion in
Short-Distance Photographs: The Selfie Effect." *JAMA Facial Plastic Surgery*
20(4):333-335. At 30 cm against 150 cm, the nasal base appears about 30 percent
wider.

**Pressler, M.P. et al. (2022).** "Size and Perception of Facial Features with
Selfie Photographs." *Plastic and Reconstructive Surgery* 149(4):859-867.
The measured counterpart to Ward's model, n=30: nasal length +6.4 percent at
12 inches; alar-base-to-facial-width ratio -10.8 percent.

**ICAO.** *Portrait Quality (Reference Facial Images for MRTD)*, v1.0.
<https://www.icao.int/sites/default/files/TRIP/Publications/TR-Portrait-Quality-v1.0.pdf>
Camera-subject distance 0.7-4 m, best practice 1.0-2.5 m. Pitch and yaw within
5 degrees, roll within 8. Magnification distortion K = depth / distance with
depth 50 mm, which is the closed form in `core/scale.py`. Notes explicitly that
"selfie style portraits are likely not to maintain the minimal distance
requirement."

**Head pose estimator accuracy.** AFLW2000-3D mean absolute error: 6DRepNet
3.97, SynergyNet 3.35, img2pose 3.91, DAD-3DNet 3.66. Label noise gives an
effective floor near 2.5-3 degrees, so a five-degree gate is at the edge of
resolvability.

## Scale recovery

**Dodgson, N.A. (2004).** "Variation and extrema of human interpupillary
distance." *Proc. SPIE* 5291:36-46.
<https://www.neildodgson.com/pubs/EI5291A-05.pdf>
Analysis of ANSUR 1988, n=3,976: mean 63.36 mm, SD 3.832; male 64.67 (3.708),
female 62.31 (3.599). The SPIE paper is all-rights-reserved, but ANSUR itself
is a US Government work, so re-derive rather than copy the tables.

**Healy, C. and Stephan, C.N. (2026).** *International Journal of Legal
Medicine*. Pooled corneal diameter 11.84 mm, SD 0.79, over 296,887 eyes;
adult-equivalent from age four. The basis for iris-based scale.

**Evereklioglu, C. et al. (1999).** Near interpupillary distance runs about
3 mm below distance interpupillary distance, a 4.7 percent systematic bias at
the range people actually take selfies.

## Normative data

**NIOSH (2003).** Head-and-face anthropometric survey of US respirator users.
CDC dataset RD-10130-2020-0.
<https://www.cdc.gov/niosh/data/datasets/rd-10130-2020-0/default.html>
3,997 subjects, 20 facial dimensions measured with calipers, stratified by sex,
three age bands and four race groups, released as a **US Government work in the
public domain**. The primary normative source here, and the only openly
redistributable one that covers bigonial and bizygomatic breadth. Two data
quirks: missing values are `-9,999` with a thousands separator inside a quoted
CSV field, and `NECKCIRC` was added mid-collection so its missingness is not
random. Published as Zhuang and Bradtmiller (2005), *JOEH* 2(11):567-576, which
is itself copyrighted and unnecessary.

**ANSUR II (2012).** US Army anthropometric survey, cleared for unlimited public
release. 4,082 men and 1,986 women, 93 measures.
<https://www.openlab.psu.edu/datasets/ansur-ii/>
**Units trap:** `interpupillarybreadth` is recorded in tenths of a millimetre
while every other dimension is in millimetres. A naive read gives a 64 cm
interpupillary distance. Note also that ANSUR II dropped the nose, lip and
frontal-breadth measures that ANSUR I carried.

**Farkas, L.G. (1994).** *Anthropometry of the Head and Face*, 2nd ed. The
canonical reference, in copyright and out of print. Farkas et al. (2005),
*Journal of Craniofacial Surgery* 16(4):615-646, is the multi-ancestry table and
is paywalled. Neither can be shipped. Fragments survive in CC BY articles:
Al-Sebaei (2015) [PMC4369102](https://pmc.ncbi.nlm.nih.gov/articles/PMC4369102/)
Table 4, and Virdi et al. (2019)
[PMC6384287](https://pmc.ncbi.nlm.nih.gov/articles/PMC6384287/).

**Farkas, L.G., Hreczko, T.A., Kolar, J.C. and Munro, I.R. (1985).** *Plastic and
Reconstructive Surgery* 75(3):328-338. Farkas disproved the neoclassical canons
himself: in 153 adults the best-performing canon held in 40 percent of cases.
Wang/Le et al. (2012),
[PMC3532441](https://pmc.ncbi.nlm.nih.gov/articles/PMC3532441/), found the
facial-thirds and orbital canons holding in 0 percent of Southern Chinese
subjects.

**Weinberg, S.M. (2019).** *AJODO*,
[PMC6571015](https://pmc.ncbi.nlm.nih.gov/articles/PMC6571015/). Stereo\-
photogrammetric norms differ substantially from Farkas' caliper norms across 24
shared distances. Method and era are confounded: do not mix caliper norms with
photogrammetric output.

Open CC BY normative tables used in `norms/published.py`: Jayaratne et al. 2013
([PMC3730197](https://pmc.ncbi.nlm.nih.gov/articles/PMC3730197/)), Celebi et al.
2013 ([PMC3606791](https://pmc.ncbi.nlm.nih.gov/articles/PMC3606791/)), Saadeh
et al. 2025 ([PMC12228583](https://pmc.ncbi.nlm.nih.gov/articles/PMC12228583/)),
Bajracharya et al. 2021
([PMC8673450](https://pmc.ncbi.nlm.nih.gov/articles/PMC8673450/)), Liu et al.
2023 ([PMC10335162](https://pmc.ncbi.nlm.nih.gov/articles/PMC10335162/)).

**Three conventions that silently invert results.** Canthal tilt is reported
both as degrees from horizontal and as the supplementary angle Ex-En-En, where
larger means flatter. Gonial angle is measured as Ar-Go-Me, Co-Go-Me or
Me-Go-Co, which differ by 5 to 15 degrees in the same jaw. fWHR has no fixed
landmark definition, and published means run from 1.83 to 2.19 on nominally the
same quantity.

## Fairness

**Buolamwini, J. and Gebru, T. (2018).** "Gender Shades." *PMLR* 81:77-91.
Darker-skinned female error 20.8 to 34.7 percent against 0.0 to 0.8 percent for
lighter-skinned males.

**Grother, P., Ngan, M. and Hanaoka, K. (2019).** NISTIR 8280, *FRVT Part 3:
Demographic Effects*. 189 algorithms, 18.27M images. False-match rates 10 to 100
times higher for Asian and African-American faces in many US-developed
algorithms, and no such gap in algorithms developed in Asian countries.

**Parte, R. et al. (2026).** "Auditing Demographic Bias in Facial Landmark
Detection." [arXiv:2604.06961](https://arxiv.org/abs/2604.06961). Raw ethnicity
gaps in landmark NME largely dissolve once head pose and image resolution are
controlled; bounding-box height alone explains 29.3 percent of NME variance. Age
is the surviving effect (70+ at 2.40 against 2.05 for 4-19).

**Dooley, S., Goldstein, T. and Dickerson, J. (2021).** "Robustness Disparities
in Commercial Face Detection." [arXiv:2108.12508](https://arxiv.org/abs/2108.12508).
About 15 percent higher error for darker skin types. This matters upstream: a
missed detection never produces a landmark error at all, so detection gaps
silently censor the evaluation set.

**Heldreth, C., Monk, E.P. et al. (2024).** "Which Skin Tone Measures Are the
Most Inclusive?" *ACM Journal on Responsible Computing* 1(1):1-21. n=2,214.
Fitzpatrick rated least inclusive, especially by darker-skinned respondents. It
was designed in 1975 to predict UV burn response in light skin, and types V-VI
were added later, so it compresses all dark skin into two bins.
Monk Skin Tone scale: <https://skintone.google/>

## Why there is no score

**Holland, E. (2008).** "Marquardt's Phi Mask: Pitfalls of Relying on Fashion
Models and the Golden Ratio to Describe a Beautiful Face." *Aesthetic Plastic
Surgery* 32(2):200-208. The goodness-of-fit validation is faulty; the mask is
ill-suited to sub-Saharan African and East Asian faces; it best describes
masculinised white female fashion models, and the public prefers above-average
facial femininity, so fitting the mask better is anti-predictive.

**Pallett, P.M., Link, S. and Lee, K. (2010).** "New golden ratios for facial
beauty." *Vision Research* 50(2):149-154. The optima are 36 percent and 46
percent, neither of which is phi, and both of which match the population
average face.

**Hönekopp, J. (2006).** "Once More: Is Beauty in the Eye of the Beholder?"
*JEP:HPP* 32(2):199-209. The share of stable rating variance that is private
rather than shared runs 0.44 to 0.65, and rises to 0.84 on faces of similar
attractiveness -- exactly the range where a score would need to be informative.

**Liang, L. et al. (2018).** SCUT-FBP5500.
[arXiv:1801.06345](https://arxiv.org/abs/1801.06345). 5,500 faces, 73 percent
Asian, scored by 60 volunteers aged 18-27. A model trained here predicts that
rater pool's mean, and any correlation reported against its own held-out split
is circular.

**Jacobs, A.Z. and Wallach, H. (2021).** "Measurement and Fairness." *FAccT '21*.
Construct validity is the question of whether an operationalisation measures the
construct at all. "Attractiveness" operationalised as a rater panel's mean has
no ground truth to validate against.

**Stark, L. and Hutson, J. (2022).** "Physiognomic Artificial Intelligence."
*Fordham IP Media & Ent. L.J.* 32(4):922.

**Rajanala, S., Maymone, M.B.C. and Vashi, N.A. (2018).** "Selfies: Living in the
Era of Filtered Photographs." *JAMA Facial Plastic Surgery* 20(6):443-444.
Coins "Snapchat dysmorphia". Body dysmorphic disorder prevalence in aesthetic
plastic surgery pools at 18.6 percent
([PMC11241264](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11241264/)), and
surgeons identified only 2 of 43 screen-positive patients (Joseph et al. 2017).

## Licensing

**Ultralytics.** <https://www.ultralytics.com/license>. AGPL-3.0 compliance
requires releasing "the complete corresponding source code for the entire
derivative work... and, where applicable, model weights." The obligation is
asserted over models produced by the training code, so third-party face
checkpoints tagged MIT or Apache-2.0 are relabels that do not launder it.

**FLAME.** <https://flame.is.tue.mpg.de/modellicense.html>. Non-commercial, no
redistribution. FLAME 2023 Open under Creative Commons Attribution is the sole
exception.

**Basel Face Model.** <https://faces.dmi.unibas.ch/bfm/bfm2019.html>. Internal
non-commercial research only, no redistribution.

**InsightFace.** MIT code; pretrained models and annotation data are
non-commercial research only, and the package downloads them automatically.

## Software depended on

YuNet (MIT), OpenCV Zoo. Pytorch_Retinaface (MIT). SPIGA (BSD-3-Clause), 98-point
heatmaps plus head pose, WFLW NME 4.060 and 300W NME 2.994. MediaPipe Face
Landmarker (Apache-2.0 including bundled models), 478 points with ten iris
landmarks. 6DRepNet (MIT). STAR loss has the best published accuracy
(WFLW NME 4.02) and no license file at all, so it is not loaded.
