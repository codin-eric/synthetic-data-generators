# Contributing

English is preferred. Issues and pull requests in Spanish are accepted.

## Ways to contribute

- **Solutions:** open a pull request that adds your solution to a challenge.
- **Bugs and improvements:** open an issue with the "Generator bug" or
  "Improvement" template. Ideas for new challenges go in "Improvement".
- **Questions and discussion:** use the
  [Discord](https://discord.com/invite/GREKfe7).

## Solutions

`<challenge>/solutions/` holds community solutions. They contain spoilers.

Put your solution in `<challenge>/solutions/<your-github-handle>/`:

- One folder per author per challenge. Organize files inside it as you want.
- Include the dependency file your tool needs (`pyproject.toml`,
  `requirements.txt`, `docker-compose.yml` or other). Any tool is accepted.
- Do not commit generated data: `data/`, `origin/`, `destination/`.
- Do not change files outside your folder.

Your folder has a `README.md` with these six sections:

1. **Tool:** the tools you used, with versions.
2. **Run:** install steps and the commands to run, in order.
3. **What I found:** the data behavior your solution handles, in your words.
4. **How it works:** your approach, and why it handles that behavior.
5. **Limits:** when it breaks or gets expensive, for example at a larger data
   size.
6. **How I checked:** the evidence that it works. If the challenge has a
   validator, include its output.

## Agents

You can use AI agents with common sense. A person writes and opens each pull
request and can explain every line of it. Issues and pull requests that are
hard to understand are closed.

## Review

Eric reviews every pull request by hand. A pull request with unticked boxes
in its template can't be reviewed.
