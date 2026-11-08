# Kipimo

*Pronounced ki-PEE-mo. Swahili for "measurement". Measure the loss. Pay the farmer.*

## How it works

When a flood ruins a farmer's crops, insurance should pay quickly. It usually doesn't, because someone has to come and measure the damage, and that takes weeks.

Kipimo sends a drone. A river forecast in the format of Google Flood Hub sees the flood coming, so nobody has to call for help, even in the middle of the night. Three AI agents on Gemini share the work: they decide where the drone looks, talk to farmers by text in their own language, and check every claim. A fixed rule sets each amount, and the insurer's actuary approves every payment. The payment goes out as a short text message the same day, even with no internet.

| Agent | Gemini model | Job |
|---|---|---|
| Field | Gemini Robotics-ER 2 | Flies the drone and decides where to look |
| Adjuster | Gemini 4 Argon | Checks every claim against written audit rules |
| Farmer | Gemini 3.8 Flash | Reads and answers farmers' texts in Swahili and English |
| Actuary | A person | Gives the final yes on every payment |

https://github.com/user-attachments/assets/dc33cce0-4071-4cac-a1e1-92df6c6922c7

[Try the dashboard](https://adeniyikayodee.github.io/kipimo-site/) · [One page technical summary](docs/technical-summary.pdf) · Built for AgentVerse 2026 (UCL AI Society), Robotics and Finance track.

## Try it

You need [uv](https://docs.astral.sh/uv/).

```
make demo
```

Then open http://localhost:8000, choose **AI agents**, and press **Start**. To run the agents on Gemini, put your key in a file called `.env` in this folder, as one line: `GEMINI_API_KEY=...`. Git ignores that file, so the key stays on your machine. Without a key, simple rules play each agent's role.

## What has been tested

| Test | Real or made up | What it showed |
|---|---|---|
| 6,872 Kenyan flood reports from Google Groundsource | **Real** | All three regions we chose flooded in the 2024 rains |
| 30 test floods in calm weather | Made up | Half the flying of covering every field. None of the 231 farmers who were owed money missed. About 1 claim in 13 paid at the wrong tier, close to covering every field |
| 60 AI test cases: calm, messy messages, tricks, storms, broken camera | Made up | No safety rule broken. Storms and camera faults miss about 1 owed farmer in 3 floods |
| A perfect reader of every farmer's message | Made up | Payments stay the same: no message, and no AI, can change who gets paid |
| Tampering, replays, losing signal | System checks | Any change to a record is caught; every payment arrives exactly once |
| Two full live missions on Gemini (Robotics-ER 2, 4 Argon, 3.8 Flash) | Made up flood, live AI | All 10 farmers who were owed got paid with zero wrong amounts, 8.9 km flown, first claim 4 minutes after the danger line. Argon challenged two borderline claims and both passed after the drone flew back. About US$0.36 to $1 of Gemini per mission |

Not tested yet: the 60 AI test cases with Gemini, the real Flood Hub and WeatherNext feeds (the forecasts are simulated in their formats), and a real drone.

## Research it builds on

| Idea | Work | How Kipimo uses it |
|---|---|---|
| Flood forecasts | Nearing et al., [Global prediction of extreme floods](https://doi.org/10.1038/s41586-024-07145-1), Nature 2024 (Google) | The forecast calls the drone |
| Weather forecasts | Price et al., [GenCast](https://doi.org/10.1038/s41586-024-08252-9), Nature 2024 (Google DeepMind) | Picks a safe time to fly |
| Flood reports from news | [Groundsource](https://research.google/blog/introducing-groundsource-turning-news-reports-into-data-with-gemini/) (Google Research, 2026) | Chose the regions and dates |
| AI for robots | [Gemini Robotics](https://arxiv.org/abs/2503.20020) (Google DeepMind, 2025) | The model behind the Field agent |
| Where to look | Krause, Singh and Guestrin, [Near-optimal sensor placements](https://jmlr.org/papers/v9/krause08a.html), JMLR 2008 | Flies where a photo would change a payment |
| Where to put drones | Church and ReVelle, [The maximal covering location problem](https://legacy.geog.ucsb.edu/~forest/G294download/MAX_COVER_RLC_CSR.pdf), 1974 | Places drone hubs like ambulance stations |
| Why fast payment matters | Janzen and Carter, [After the Drought](https://ideas.repec.org/a/oup/ajagec/v101y2019i3p651-671..html), 2019 | Payments protect families' meals and assets in Kenya |

## More

Everything else, including how each AI agent works, how we test them, the full results and all the commands, is in [docs/details.md](docs/details.md).
