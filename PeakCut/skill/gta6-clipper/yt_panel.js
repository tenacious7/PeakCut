// YouTube tab + Settings card. Merge into app.html's <script>. api(path, body?) is the app's helper (GET without body, POST with).
// Needs routes from ytstats.handle plus POST /api/yt/ideas {run,file} -> {titles:[5],hooks:[3],recut:""} (app.py runs ytstats.ai_prompt with the chosen AI) or {error}.
const ytEl = (t, css, txt, kids) => { const e = document.createElement(t); if (css) e.style.cssText = css; if (txt != null) e.textContent = txt; (kids || []).forEach(k => e.append(k)); return e; };
const ytBtn = (txt, fn, css) => { const b = ytEl('button', css || '', txt); b.onclick = fn; return b; };
const ytNum = n => n == null ? '-' : Math.round(n).toLocaleString();

async function renderYtPanel(container, short, api) {
  container.replaceChildren();
  const st = await api('/api/yt/status'), q = `?run=${encodeURIComponent(short.run)}&file=${encodeURIComponent(short.file)}`;
  const note = (t, c) => container.append(ytEl('div', `padding:10px 12px;border:2px solid var(--l);border-radius:var(--r);margin:8px 0;color:${c || 'var(--m)'}`, t));
  if (!st.has_key) return note('Add your YouTube API key in Settings > YouTube to see how this Short is doing.');
  const link = ytEl('input', 'flex:1;min-width:200px;min-height:44px;padding:8px 12px;background:var(--bg);color:var(--t);border:2px solid var(--l);border-radius:var(--r)');
  link.placeholder = 'Paste the YouTube link of this Short'; link.setAttribute('aria-label', 'YouTube link');
  const body = ytEl('div'), go = async () => {
    body.replaceChildren(ytEl('div', 'color:var(--m)', 'Loading...'));
    const d = await api('/api/yt/diagnose' + q); body.replaceChildren();
    if (d.error) return body.append(ytEl('div', 'color:var(--f)', d.error));
    if (!d.linked) return body.append(ytEl('div', 'color:var(--m)', d.hint + (st.channel ? ' Or post it with the same title and it links itself.' : ' Add your channel in Settings to auto-link by title.')));
    const v = d.video, m = d.median || {}, a = d.analytics || {}, tiles = ytEl('div', 'display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:12px 0');
    const tile = (label, val, med, bad) => tiles.append(ytEl('div', `background:var(--c);border:2px solid ${bad ? 'var(--f)' : 'var(--l)'};border-radius:var(--r);padding:10px;box-shadow:var(--sh)`, null,
      [ytEl('div', 'font:12px system-ui;color:var(--m)', label), ytEl('div', 'font:700 22px var(--fd)', val), ytEl('div', 'font:12px system-ui;color:var(--m)', med == null ? '' : 'your median ' + med)]));
    tile('Views', ytNum(v.views), ytNum(m.views), m.views && v.views < .5 * m.views);
    tile('Likes', ytNum(v.likes), ytNum(m.likes)); tile('Comments', ytNum(v.comments));
    if (a.averageViewPercentage != null) tile('Avg viewed', Math.round(a.averageViewPercentage) + '%', m.averageViewPercentage != null ? Math.round(m.averageViewPercentage) + '%' : null, m.averageViewPercentage && a.averageViewPercentage < .8 * m.averageViewPercentage);
    if (a.kept_at_3s != null) tile('Still there at 3 s', Math.round(a.kept_at_3s * 100) + '%', null, a.kept_at_3s < .7);
    if (a.subscribersGained != null) tile('Subs gained', ytNum(a.subscribersGained), m.subscribersGained != null ? ytNum(m.subscribersGained) : null);
    body.append(ytEl('div', 'font-weight:600', v.title), tiles);
    if (d.level === 'key') body.append(ytEl('div', 'color:var(--m);font-size:13px', st.has_oauth ? 'Sign in with Google (Settings > YouTube) to also see watch time and drop-off.' : 'Add your OAuth client in Settings > YouTube to also see watch time and drop-off in the first 3 seconds.'));
    const col = { bad: 'var(--f)', warn: 'var(--w)', info: 'var(--at)' };
    body.append(ytEl('h3', 'margin:14px 0 6px', d.reasons.length ? 'Why it may not be doing well' : 'No problems found by the rules'),
      ...d.reasons.map(r => ytEl('div', `border-left:5px solid ${col[r.level] || 'var(--l)'};padding:6px 10px;margin:6px 0;background:var(--p)`, r.text)));
    const out = ytEl('div'), ideas = ytBtn('Ideas from AI', async () => {
      out.replaceChildren(ytEl('div', 'color:var(--m)', 'Asking your AI tool...'));
      const r = await api('/api/yt/ideas', { run: short.run, file: short.file }); out.replaceChildren();
      if (r.error) return out.append(ytEl('div', 'color:var(--f)', r.error));
      const sect = (h, arr) => out.append(ytEl('h3', 'margin:12px 0 4px', h), ...arr.map(t => ytEl('div', 'display:flex;gap:8px;align-items:center;margin:4px 0', null, [ytEl('span', 'flex:1', t), ytBtn('Copy', e => { navigator.clipboard.writeText(t); e.target.textContent = 'Copied'; })])));
      sect('Titles', r.titles || []); sect('Hooks', r.hooks || []); if (r.recut) sect('Re-cut idea', [r.recut]);
    }, 'margin-top:12px;border-color:var(--a)');
    body.append(ideas, out);
  };
  container.append(ytEl('div', 'display:flex;gap:8px;flex-wrap:wrap', null, [link, ytBtn('Link', async () => { const r = await api('/api/yt/link', { run: short.run, file: short.file, url: link.value }); r.error ? note(r.error, 'var(--f)') : go(); })]), body);
  go();
}

async function renderYtSettings(container, api) {
  const draw = async () => {
    const st = await api('/api/yt/status'); container.replaceChildren();
    const inp = (ph, type) => { const i = ytEl('input', 'min-height:44px;padding:8px 12px;background:var(--bg);color:var(--t);border:2px solid var(--l);border-radius:var(--r);width:100%;box-sizing:border-box;margin:4px 0'); i.placeholder = ph; i.type = type || 'text'; i.setAttribute('aria-label', ph); i.autocomplete = 'off'; return i; };
    const key = inp('YouTube Data API key (never shown again)', 'password'), ch = inp('Your channel (@handle or UC id)'); ch.value = st.channel || '';
    const msg = ytEl('div', 'color:var(--m);margin-top:6px'), file = ytEl('input'); file.type = 'file'; file.accept = '.json'; file.setAttribute('aria-label', 'OAuth client JSON');
    container.append(ytEl('h3', 'margin:0 0 6px', 'YouTube stats'),
      ytEl('div', 'color:var(--m);font-size:13px', 'Use your own keys. They stay on this PC and are write-only here. Level 1: API key = views, likes, why-not-better rules. Level 2: OAuth client = watch time and drop-off.'),
      ytEl('div', 'margin:6px 0', st.has_key ? 'API key saved.' : 'No API key yet.'), key, ch,
      ytBtn('Save', async () => { const r = await api('/api/yt/key', { api_key: key.value, channel: ch.value }); r.error ? (msg.style.color = 'var(--f)', msg.textContent = r.error) : draw(); }),
      ytEl('div', 'margin:14px 0 4px', st.signed_in ? 'Signed in to Analytics.' : 'Optional: OAuth client JSON (Google Cloud, Desktop app).'), file,
      ytBtn('Add and sign in', async () => { if (!file.files[0]) return; const r = await api('/api/yt/oauth', { client_json: await file.files[0].text() }); if (r.error) { msg.style.color = 'var(--f)'; msg.textContent = r.error; } else { window.open(r.auth_url, '_blank', 'noopener'); msg.textContent = 'Finish the sign-in in the new tab, then reopen this card.'; } }, 'margin-left:8px'), msg);
  };
  draw();
}
