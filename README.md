# Proportio – KI-Gesichtsanalyse

Eigenständige Web-App zur geometrischen Analyse von Gesichtsproportionen (inspiriert vom Funktionsumfang von Face-Rating-Apps wie FaceIQ Labs, aber mit eigenem Branding, Design und Code).

## Funktionen
- Foto-Upload (Button oder Drag & Drop) oder Aufnahme per Webcam
- Erkennung von 478 Gesichtspunkten (inkl. Iris) mit MediaPipe Face Landmarker – **komplett im Browser**, kein Server-Upload
- Automatische Korrektur der Kopfneigung, Warnung bei seitlich gedrehtem Kopf
- 13 Metriken mit Idealbereichen (männlich/weiblich) und Score 0–10:
  Symmetrie, Gesichtsdrittel, FWHR, Canthal Tilt, Augenabstand (ESR), Augen-Fünftel, Augenöffnung,
  Mittelgesichts-Ratio, Nasenbreite, Mund-/Nasenbreite, Lippenverhältnis, Kinn/Philtrum, Kieferbreite
- Gesamtscore, Kategorie-Scores, Overlay (Mesh + Hilfslinien) auf dem Foto
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
- `tools/build.py` – erzeugt die Einzeldatei

## Hinweis
Die Werte sind eine geometrische Orientierung auf Basis eines 2D-Fotos und hängen stark von Kamerawinkel, Brennweite und Mimik ab. Kein medizinischer Rat.
