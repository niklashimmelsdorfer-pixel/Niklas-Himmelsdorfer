// Proportio – Konfiguration
// ---------------------------------------------------------------------------
// payment.mode:
//   "demo"         – Testmodus: simulierte Kasse, es wird kein Geld abgebucht.
//   "lemonsqueezy" – Live: Kauf über Lemon Squeezy, Freischaltung per Lizenzschlüssel.
//
// Für den Live-Betrieb in Lemon Squeezy ein Produkt "Proportio Pro" mit zwei
// Abo-Varianten (5 €/Monat, 50 €/Jahr) und aktivierten Lizenzschlüsseln anlegen
// und die Werte unten eintragen (siehe README → "Live gehen").
window.PROPORTIO_CONFIG = Object.assign({
  payment: {
    mode: "demo",
    priceMonthly: 5,
    priceYearly: 50,
    currency: "€",
    // Checkout-Links der beiden Varianten (Lemon Squeezy → Produkt → Variante → "Share")
    checkoutMonthly: "",
    checkoutYearly: "",
    // Varianten-IDs, um Monats- und Jahresabo zu unterscheiden (optional)
    variantMonthly: null,
    variantYearly: null,
    // Nur Schlüssel aus diesem Shop akzeptieren (Lemon Squeezy → Settings → Stores → ID)
    storeId: null,
    // Kundenportal zum Verwalten/Kündigen des Abos
    portalUrl: "https://app.lemonsqueezy.com/my-orders",
    // Kontaktadresse für Widerrufe (wie im Impressum)
    contactEmail: "kontakt@deine-domain.de",
  },
}, window.PROPORTIO_CONFIG || {});
