# Faciometry

Faciometry measures a face from a photograph and tells you how much each
measurement can be trusted. It runs entirely on your own computer. No photo is
uploaded anywhere.

![The capture screen of the local app](https://raw.githubusercontent.com/EricSpencer00/faciometry/main/docs/images/capture.png)

## The one thing to know first

**Most measurements get withheld, and that is the point.**

A photograph of a face is a bad instrument. Turn your head ten degrees and a
facial ratio moves more than that ratio differs between two different people
([Kleinberg and Vanezis 2007](https://theses.gla.ac.uk/245/1/2008kleinbergphd.pdf)).
So Faciometry checks every measurement against its own error, and where the
photograph contributes more than the person does, it prints the reason instead
of the number.

On an ordinary phone photo that means almost everything is withheld. The report
then tells you what to change: a ruler in the shot, better light, standing
further back, several photos instead of one.

Faciometry does not give you a score, a rating, or a number for your face. There
is no ground truth for such a thing, and it is the part of this idea that does
documented harm.

## Install

### Mac, no terminal needed

1. Download `Faciometry.dmg` from the [latest release](https://github.com/EricSpencer00/faciometry/releases/latest).
2. Open it and drag Faciometry to Applications.
3. **Right-click the app and choose Open** the first time. macOS will warn you
   because the app is signed but not yet notarised by Apple. After the first
   time, it opens normally.
4. It opens in your browser. The first launch downloads about 400 MB of models,
   once. Everything after that works with the internet off.

### Anything with Python

```
pip install 'faciometry[permissive]'
faciometry fetch-weights
faciometry analyze photo.jpg --out report/
```

### Docker

```
docker run -p 8731:8731 -v faciometry-weights:/weights ghcr.io/ericspencer00/faciometry
```

Then open <http://127.0.0.1:8731>.

## Taking a photo it can actually use

Most of the difference is here, not in the software.

- Stand about 1.5 m away and zoom in. Close-up distorts a face by about 17% at
  30 cm and about 3% at 1.5 m.
- Look straight at the lens, camera at eye height.
- Even light, no hat, hair off the face.
- **Hold a ruler flat against your cheek.** This is the single biggest
  improvement available: it replaces a population guess about your size with an
  actual measurement.
- Take several photos rather than one.

## What you get

`report.html` to read in a browser, `report.pdf` to keep, plus `report.json`
and `report.txt`. Every value carries a 95% interval, the landmarks it came
from, and a fingerprint of the formula that produced it.

![A page of the report](https://raw.githubusercontent.com/EricSpencer00/faciometry/main/docs/images/verdicts.png)

## Honest limits

- **Side-profile photos do not work yet.** The pose model saturates before a
  true profile, so the 20 profile measurements cannot be reached.
- **Skin measurements are comparable within one photo, not between photos**,
  unless you include a grey card.
- Not a medical device. Nothing here is a diagnosis or advice.

## Licence

Apache-2.0. The default face models are permissively licensed. An optional
YOLO-based tier exists for skin lesion detection; turning it on makes your
deployment AGPL-3.0, and `faciometry licenses` explains exactly what each option
commits you to.

More detail: [full README](docs/README-full.md) ·
[what it does and does not establish](docs/FINDINGS.md) ·
[privacy](docs/PRIVACY.md) · [references](docs/REFERENCES.md)
