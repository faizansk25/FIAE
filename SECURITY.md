# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.0.x   | ✅ Yes |

## Security Overview

FIAE is a command-line tool that runs locally on your machine. It does not:
- Send data to external services
- Require authentication or credentials
- Open network ports
- Access the internet without explicit user action (e.g., URL data sources)

All data processing happens locally. Your data never leaves your machine unless you explicitly configure cloud storage access.

## Reporting a Vulnerability

If you discover a security vulnerability in FIAE, please report it responsibly.

### How to Report

Email: security@fiae.dev

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Investigation | Within 7 days |
| Fix | Within 30 days (depending on complexity) |

## Security Best Practices

When using FIAE:

1. **Keep dependencies updated** — Run `pip install --upgrade fiae` regularly
2. **Use virtual environments** — Isolate FIAE from other Python packages
3. **Review feature outputs** — Validate results before production use
4. **Limit file permissions** — Run FIAE with minimal required permissions
5. **Audit custom operators** — Review any custom feature operators before use

## Automated Security Scanning

FIAE uses automated tools to detect vulnerabilities:
- **Dependabot** — Monitors dependencies for known CVEs
- **Semgrep** — Static analysis for common security issues
- **CodeQL** — Semantic code analysis (where applicable)

## Past Security Audits

| Date | Scope | Status |
|------|-------|--------|
| — | — | No audits yet |

We plan to commission an independent security audit when funding allows. See [DONATIONS.md](DONATIONS.md) to support this effort.
