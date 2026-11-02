# Proportio – KI-Gesichtsanalyse

Eigenständige Web-App zur geometrischen Analyse von Gesichtsproportionen (inspiriert vom Funktionsumfang von Face-Rating-Apps wie FaceIQ Labs, aber mit eigenem Branding, Design und Code).

## Funktionen
- Foto-Upload (Button oder Drag & Drop) oder Aufnahme per Webcam
- Erkennung von 478 Gesichtspunkten (inkl. Iris) mit MediaPipe Face Landmarker – **komplett im Browser**, kein Server-Upload
- Gesicht wird automatisch erkannt, aufrecht gedreht (auch bei 90°/180° gedrehten Fotos) und zugeschnitten
- Scan-Animation bei der Analyse, sanfter Auto-Zoom auf die jeweilige Messstelle, Messlinien zeichnen sich ein
- Werte per ‹ › (oder Pfeiltasten) durchklicken, Stärken & Potenzial im Überblick, automatische Gesichtsform / Profiltyp
- Sichtbare Messpunkte, die sich per Drag & Drop (mit Lupe) korrigieren lassen – alle Werte rechnen live neu
- Beispielfoto zum Ausprobieren (NASA, gemeinfrei)
- Fahre über einen Wert (oder tippe ihn an): die zugehörige Messung (Linien, Winkel, Abstände) erscheint im Bild
- **Frontal** – 20 Metriken, u. a.: Symmetrie, Gesichtsdrittel, FWHR, Canthal Tilt, Augenabstand (ESR), Augen-Fünftel,
  Augenöffnung, Mittelgesichts-Ratio, Nasenbreite, Mund-/Nasenbreite, Lippenverhältnis, Kinn/Philtrum, Kieferbreite,
  Gesichtsindex, Nasenlänge, Augenbreite, Brauenneigung, Brauen-Lid-Abstand, Lippenfülle, Mund zu Pupillen
- **Seitenprofil** – 15 Metriken: nasofrontaler, nasolabialer, nasofazialer und nasomentaler Winkel, Mentolabialwinkel,
  Ricketts-E-Linie und Burstone-Linie (Ober-/Unterlippe), Nasenprojektion nach Goode, Profilkonvexität, Gonialwinkel,
  Unterkieferneigung, Kieferast/Kieferkörper, Lippe-zu-Kinn. Punkte werden automatisch
  gesetzt, wenn die KI das Profil erkennt – sonst klickt man sie geführt nacheinander an.
- Idealbereiche für männlich/weiblich, Score 0–10, Gesamtscore und Kategorie-Scores
- Personalisierte Verbesserungstipps
- Lokaler Verlauf (localStorage) und Export der Auswertung als PNG

## Starten
**Am einfachsten:** `proportio.html` herunterladen und per Doppelklick im Browser öffnen (Chrome, Edge, Firefox, Safari). Alles steckt in dieser einen Datei; Internet wird nur zum einmaligen Laden des KI-Modells gebraucht.

Hinweis: Auf github.com selbst wird HTML nur als Quelltext angezeigt – die Datei muss heruntergeladen oder über GitHub Pages bzw. einen Server geöffnet werden.

`proportio.html` wird aus `index.html`, `styles.css` und `app.js` erzeugt: `python3 tools/build.py`

Entwicklung mit lokalem Server (für die Kamera nötig):

```bash
python3 -m http.server 8000
# dann http://localhost:8000 öffnen
```

Deploybar auf GitHub Pages, Netlify, Vercel o. ä.

## Dateien
- `index.html` – Landingpage + Analyzer
- `styles.css` – Design
- `app.js` – Landmarkenerkennung, Metrik-Berechnung, Scoring, UI
- `proportio.html` – fertige Einzeldatei (generiert)
- `config.js` – Bezahl-Einstellungen (Testmodus / Live)
- `impressum.html`, `datenschutz.html`, `agb.html`, `widerruf.html` – Rechtstexte (Vorlagen)
- `assets/fonts/` – selbst gehostete Schriften (OFL)
- `.github/workflows/pages.yml` – Veröffentlichung auf GitHub Pages
- `tools/build.py` – erzeugt Einzeldatei (`proportio.html`) und Website (`--site`)
- `assets/sample-front.jpg` – Beispielfoto (NASA, gemeinfrei)

## Hinweis
Die Werte sind eine geometrische Orientierung auf Basis eines 2D-Fotos und hängen stark von Kamerawinkel, Brennweite und Mimik ab. Kein medizinischer Rat.

## Abo-Modell (Pro)

| | Kostenlos | Pro |
|---|---|---|
| Gesamtscore, Gesichtsform/Profiltyp | ✓ | ✓ |
| Basiswerte (5 frontal + 5 Profil) mit Messung im Bild | ✓ | ✓ |
| Alle 35 Messwerte | – | ✓ |
| Stärken, Potenzial & Tipps | – | ✓ |
| Export & Verlauf | – | ✓ |
| Preis | 0 € | 5 €/Monat oder 50 €/Jahr |

Gesteuert über `config.js`:

- `mode: "demo"` – **Testmodus**: simulierte Kasse (Testkarte vorausgefüllt), Testschlüssel `DEMO-…`, Kündigung und Widerruf lassen sich durchspielen. Es fließt kein Geld.
- `mode: "lemonsqueezy"` – **Live**: Kauf über Lemon Squeezy, Freischaltung per Zugangsschlüssel aus der Kauf-E-Mail.

Lemon Squeezy tritt als Verkäufer auf (Merchant of Record): EU-Umsatzsteuer, Rechnungen, Zahlungsarten
und das Kundenportal (Kündigen, Zahlungsmethode) übernimmt der Anbieter. Die Website braucht keinen eigenen Server:
Zugangsschlüssel werden direkt im Browser über die Lemon-Squeezy-License-API geprüft (alle 6 Stunden erneut, offline 7 Tage Kulanz).

## Live gehen – Checkliste

1. **Lemon Squeezy einrichten** (lemonsqueezy.com)
   - Shop anlegen, Identität und Auszahlungskonto verifizieren.
   - Produkt „Proportio Pro“ als **Abo** mit zwei Varianten: *Monatlich* 5,00 € / Monat und *Jährlich* 50,00 € / Jahr.
   - Beim Produkt **„Generate license keys“** aktivieren, Aktivierungslimit z. B. 3 Geräte, Gültigkeit an das Abo koppeln.
   - Nach dem Kauf weiterleiten auf `https://<deine-seite>/index.html?purchased=1` (öffnet automatisch die Schlüssel-Eingabe).
   - Kundenportal aktivieren.
2. **`config.js` ausfüllen**: `mode: "lemonsqueezy"`, die beiden Checkout-Links (Variante → „Share“), Varianten-IDs, Store-ID, Kontakt-E-Mail.
3. **Mit Lemon-Squeezy-Testmodus testen**: Im Test-Modus des Shops erzeugte Checkout-Links eintragen und mit der Testkarte
   `4242 4242 4242 4242` einen Kauf durchspielen: Schlüssel kommt per E-Mail → auf der Seite eingeben → Pro aktiv →
   im Kundenportal kündigen. Erst danach die Live-Links eintragen.
4. **Rechtstexte**: alle `[Platzhalter]` in `impressum.html`, `datenschutz.html`, `agb.html`, `widerruf.html` ausfüllen und
   prüfen lassen (Anwalt oder Rechtstext-Dienst). Besonders klären:
   - **Jahresabo-Verlängerung** – gegenüber Verbrauchern darf sich ein Vertrag nach der Erstlaufzeit nur auf unbestimmte Zeit
     mit maximal einem Monat Kündigungsfrist verlängern (§ 309 Nr. 9 BGB). Die automatische Verlängerung um ein weiteres Jahr ist daher problematisch.
   - **Kündigungsbutton** (§ 312k BGB) und **Widerrufsfunktion** (seit 19.06.2026) – beide sind auf der Seite eingebaut, die Abläufe mit dem Anbieter abstimmen.
   - Gewerbeanmeldung und Steuern (z. B. Kleinunternehmerregelung) mit Steuerberatung klären.
5. **Veröffentlichen** (GitHub Pages, kostenlos):
   - Diesen Branch in `main` mergen.
   - GitHub → Settings → Pages → *Source*: **GitHub Actions**.
   - GitHub → Actions → **„Website veröffentlichen“** → *Run workflow*. Der Workflow bricht ab, solange noch der Testmodus
     eingestellt ist oder Platzhalter in den Rechtstexten stehen.
   - Adresse danach: `https://niklashimmelsdorfer-pixel.github.io/Niklas-Himmelsdorfer/`
   - Eigene Domain (empfohlen): Datei `CNAME` mit der Domain anlegen und beim Domain-Anbieter auf GitHub Pages zeigen lassen.
6. **Name prüfen**: Vor dem Start klären, ob „Proportio“ markenrechtlich frei ist (DPMA/EUIPO-Recherche).

**Grenzen ohne eigenen Server:** Die Analyse läuft im Browser, daher lässt sich die Sperre der Pro-Werte von technisch
versierten Nutzer:innen umgehen. Für ein 5-€-Abo ist das üblich vertretbar. Wer das ausschließen will, braucht eine
Server-Komponente (z. B. Analyse oder Auswertung serverseitig).
