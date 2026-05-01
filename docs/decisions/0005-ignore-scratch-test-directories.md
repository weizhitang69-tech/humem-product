# 0005: Ignore local scratch test directories

Date: 2026-05-01

Status: Accepted

## Context

A previous sandboxed test run created `.tmp_tests/` with broken Windows ACLs.
The directory could not be removed by the current user, and Git tried to inspect
it before every status command.

## Decision

Add `.tmp_tests/` to `.gitignore`.

## Consequences

The repository no longer treats that local scratch directory as source material.
The ignored folder is not part of the product; it can be deleted manually once
the host file permissions are repaired.

