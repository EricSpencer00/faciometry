/* Rendering a report, and the catalogue it was drawn from.
 *
 * One rule governs the whole file: a value never appears without its interval.
 * The interval is printed and also drawn, because a 95% span that is a third
 * of the value looks like a rounding artifact in text and looks like what it
 * is as a bar.
 *
 * The bar's full width is a relative interval of 0.35, which is where the
 * measurement layer stops reporting a value at all. So a bar that fills its
 * track is a measurement one step from being withheld, and the scale means
 * something rather than being fitted to whatever came back.
 */

const FULL_SCALE_RELATIVE_WIDTH = 0.35;

const EVIDENCE_SHORT = {
  validated_2d: 'validated',
  pose_invariant_ratio: 'ratio-inv',
  requires_3d: 'needs-3d',
  pose_critical: 'pose-crit',
  conventional: 'convention',
};

const el = (tag, attrs = {}, children = []) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    if (child) node.append(child);
  }
  return node;
};

const sig = (x, digits = 4) => {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const abs = Math.abs(x);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return x.toExponential(2);
  return Number(x.toPrecision(digits)).toString();
};

const amount = (value, unit) => {
  if (value === null || value === undefined) return 'unknown';
  return unit === 'deg' ? `${value.toFixed(2)} deg` : `${(value * 100).toFixed(2)}%`;
};

function head(title, count, note) {
  return el('div', { class: 'section-head' }, [
    el('h3', { text: title }),
    el('span', { class: 'count', text: note ? `${count} · ${note}` : String(count) }),
  ]);
}

function intervalBar(m) {
  const span = Math.abs(m.ci_high - m.ci_low);
  const rel = Math.abs(m.value) > 1e-12 ? span / Math.abs(m.value) : 1;
  const frac = Math.min(1, rel / FULL_SCALE_RELATIVE_WIDTH);
  const bar = el('div', { class: 'bar', title: `95% interval spans ${(rel * 100).toFixed(1)}% of the value` });
  const fill = el('span');
  fill.style.left = `${(0.5 - frac / 2) * 100}%`;
  fill.style.width = `${frac * 100}%`;
  const tick = el('i');
  tick.style.left = '50%';
  bar.append(fill, tick);
  return bar;
}

function measuredRow(m) {
  const caveat = m.reportability === 'caveat';
  const row = el('tr', { class: caveat ? 'caveat' : '' });
  row.append(
    el('td', {}, [
      el('b', { class: 'name', text: m.label }),
      el('span', { class: 'subid', text: `${m.id} · ${m.formula_fingerprint}` }),
    ]),
    el('td', { class: 'num' }, [
      el('span', { class: 'value', text: `${sig(m.value)}${m.unit === 'ratio' ? '' : ' ' + m.unit}` }),
      el('span', { class: 'interval', text: `${sig(m.ci_low)} to ${sig(m.ci_high)}` }),
    ]),
    el('td', { class: 'num' }, [intervalBar(m)]),
    el('td', { class: 'num' }, [
      el('span', {
        class: m.discriminability === null || m.discriminability === undefined
          ? 'tier'
          : 'tier ratio',
        text: m.discriminability === null || m.discriminability === undefined
          ? 'spread unknown'
          : `${m.discriminability.toFixed(2)}x`,
      }),
    ]),
  );
  if (caveat && m.reasons && m.reasons.length) {
    const note = el('tr', { class: 'caveat' });
    note.append(el('td', { colspan: '4' }, m.reasons.map((r) => el('p', { class: 'reason', text: r }))));
    return [row, note];
  }
  return [row];
}

export function renderReport(container, payload) {
  container.replaceChildren();

  const measured = payload.measurements || [];
  const withheld = payload.withheld || [];
  const unavailable = payload.unavailable || [];

  container.append(head('Reported', measured.length, 'value, 95% interval, spread over error'));
  if (measured.length) {
    const table = el('table', { class: 'data' });
    table.append(
      el('thead', {}, [
        el('tr', {}, [
          el('th', { text: 'measurement' }),
          el('th', { class: 'num', text: 'value and interval' }),
          el('th', { class: 'num', text: 'interval, to scale' }),
          el('th', { class: 'num', text: 'person / photograph' }),
        ]),
      ]),
    );
    const body = el('tbody');
    measured.forEach((m) => measuredRow(m).forEach((r) => body.append(r)));
    table.append(body);
    container.append(table);
  } else {
    container.append(el('p', { class: 'empty', text: 'No measurement cleared the gate on this photograph.' }));
  }

  container.append(head('Withheld', withheld.length, 'the reason, in place of the number'));
  if (withheld.length) {
    const table = el('table', { class: 'data' });
    const body = el('tbody');
    for (const w of withheld) {
      const row = el('tr', { class: 'withheld' });
      row.append(
        el('td', {}, [
          el('b', { class: 'name', text: w.label }),
          el('span', { class: 'subid', text: w.id }),
          ...(w.reasons || []).map((r) => el('p', { class: 'reason', text: r })),
        ]),
      );
      body.append(row);
    }
    table.append(body);
    container.append(table);
  } else {
    container.append(el('p', { class: 'empty', text: 'Nothing was withheld.' }));
  }

  container.append(head('Unavailable', unavailable.length, 'a landmark the model does not supply'));
  if (unavailable.length) {
    const table = el('table', { class: 'data' });
    const body = el('tbody');
    for (const u of unavailable) {
      const row = el('tr', { class: 'withheld' });
      row.append(
        el('td', {}, [
          el('b', { class: 'name', text: u.label }),
          el('span', { class: 'subid', text: u.id }),
          el('p', { class: 'reason', text: u.reason }),
        ]),
      );
      body.append(row);
    }
    table.append(body);
    container.append(table);
  } else {
    container.append(el('p', { class: 'empty', text: 'Every measurement in the catalogue had the landmarks it needs.' }));
  }

  if (payload.run) {
    const r = payload.run;
    const dropped = (r.frontal.exif_tags_dropped || []).length;
    container.append(
      el('p', { class: 'empty' }, [
        el('span', {
          text:
            `seed ${r.seed} · tier ${r.license_tier} · ` +
            `frontal ${r.frontal.width}×${r.frontal.height}, sha256 ${r.frontal.sha256.slice(0, 16)} · ` +
            `${dropped} metadata tag${dropped === 1 ? '' : 's'} dropped at ingest`,
        }),
      ]),
    );
  }
}

export function renderCatalogue(container, payload) {
  container.replaceChildren();
  const rows = payload.measurements || [];
  const byView = { frontal: [], profile: [] };
  for (const r of rows) (byView[r.view] || (byView[r.view] = [])).push(r);

  container.append(
    head('Catalogue', rows.length, `moves quoted at ${payload.quoted_pose_deg} degrees`),
  );
  container.append(
    el('p', { class: 'empty', text:
      'The last column is between-person spread divided by how far the measurement ' +
      'moves at ten degrees of head rotation. Below 1, the photograph contributes ' +
      'more variance than the person does, and the value is withheld.' }),
  );

  for (const [view, group] of Object.entries(byView)) {
    if (!group.length) continue;
    container.append(head(`${view} view`, group.length));
    const table = el('table', { class: 'data' });
    table.append(
      el('thead', {}, [
        el('tr', {}, [
          el('th', { text: 'measurement' }),
          el('th', { class: 'num', text: 'evidence' }),
          el('th', { class: 'num', text: 'tolerance' }),
          el('th', { class: 'num', text: 'moves @10°' }),
          el('th', { class: 'num', text: 'between people' }),
          el('th', { class: 'num', text: 'ratio' }),
        ]),
      ]),
    );
    const body = el('tbody');
    for (const r of group) {
      const ratio = r.discriminability_at_quoted_pose;
      const row = el('tr', { class: ratio !== null && ratio < 1 ? 'withheld' : '' });
      row.append(
        el('td', {}, [
          el('b', { class: 'name', text: r.label }),
          el('span', { class: 'subid', text: r.id }),
        ]),
        el('td', { class: 'num' }, [el('span', { class: 'tier', text: EVIDENCE_SHORT[r.evidence] || r.evidence })]),
        el('td', { class: 'num', text: `${r.pose_tolerance_deg.toFixed(0)}°` }),
        el('td', { class: 'num', text: amount(r.move_at_quoted_pose, r.unit) }),
        el('td', { class: 'num', text: amount(r.between_subject_spread, r.unit) }),
        el('td', { class: 'num' }, [
          el('span', {
            class: ratio === null || ratio === undefined ? 'tier' : 'tier ratio',
            text: ratio === null || ratio === undefined ? 'unknown' : `${ratio.toFixed(2)}x`,
          }),
        ]),
      );
      body.append(row);
    }
    table.append(body);
    container.append(table);
  }
}
