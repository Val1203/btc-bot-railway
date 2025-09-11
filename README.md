# BTC Bot Railway (24/7)

Bot de trading **spot** (ccxt/binance) prêt pour **Railway**, qui:
- lit sa config depuis les **Variables Railway**,
- exécute une stratégie simple (RSI + écart sous EMA),
- **journalise UNIQUEMENT** les trades exécutés (WIN/LOSS) dans Google Sheets,
- génère automatiquement un **bilan quotidien** et **hebdomadaire**.

## Déploiement express

1. Crée un Google Sheet vide et récupère son **ID** (entre `/d/` et `/edit`).
2. Crée un **Service Account** Google; copie le JSON *dans la variable* `GOOGLE_SERVICE_ACCOUNT_JSON`.
   - Partage le Google Sheet avec l'email du service account (droits Éditeur).
3. Sur Railway > Variables, ajoute exactement celles-ci (voir `.env.sample`).
4. Pousse ce repo sur GitHub, puis **Deploy from GitHub** dans Railway.
5. Assure-toi que `Procfile` est détecté et que le service se lance en **worker**.
6. Mets `TRADING_ENABLED=true` seulement quand prêt (sinon `DRY_RUN=true` pour tester).

## Sécurité & Limites
- Limites: `DAILY_MAX_LOSS_USDC`, `DAILY_LOSSES_LIMIT`, `CONSECUTIVE_LOSS_LIMIT` + `COOLDOWN_MINUTES`.
- Cap d'exposition: `MAX_CAP_USDC`.
- Pas de stockage de clés en clair (tout via variables Railway).
