name: Log trending stocks

on: { schedule: [ { cron: "15 13 * * 1-5" } ], workflow_dispatch: {} }

permissions: { contents: write }

jobs:
  log:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install alpaca-py
      - name: Log today's most actives
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
        run: python log_trending.py
      - name: Commit the updated log
        run: |
          git config user.name "trending-logger"
          git config user.email "actions@users.noreply.github.com"
          git add trending_log.csv
          git diff --cached --quiet || git commit -m "Log trending stocks"
          git push
